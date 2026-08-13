#!/usr/bin/env python3
"""
eclipse_align.py
=================

Recadre et aligne une série de photos d'éclipse solaire pour que le disque
solaire (entier ou partiellement masqué) soit toujours centré et à la même
taille apparente sur toutes les images. Permet ensuite de créer un
time-lapse.

Workflow en 3 étapes :

  1. preview  : détecte le disque solaire sur chaque photo et écrit un
                fichier tableur ODF (.ods, ouvrable dans LibreOffice Calc)
                avec le nom de fichier, le centre et le rayon détectés, que
                vous pouvez ouvrir et corriger à la main pour les photos mal
                détectées ou non détectées (typiquement les photos de
                totalité).

  2. process  : relit ce fichier .ods (après vos éventuelles corrections)
                et génère les images alignées, sans refaire de détection.

  3. video    : assemble les images alignées en vidéo.

Exemple :

    python3 eclipse_align.py preview --input photos/ --output apercu/
    # => corriger apercu/detections.ods si besoin (avec LibreOffice Calc).
    # On peut utiliser Gimp pour calculer les coordonnées du disque solaire sur l'image.
    python3 eclipse_align.py process --input photos/ --output alignees/ --detections apercu/detections.ods
    python3 eclipse_align.py video --input alignees/ --output eclipse.mp4 --fps 6

Colonnes du fichier detections.ods :

    filename | timestamp | timestamp_source | cx | cy | r | status | method

La colonne "timestamp" est extraite automatiquement de l'EXIF de la photo
(date/heure réelle de prise de vue) ; si l'EXIF est absent, la date de
modification du fichier est utilisée à la place (colonne "timestamp_source"
= "exif" ou "mtime").

Pour corriger une ligne en échec (ou une détection imprécise), remplissez
simplement les colonnes cx, cy, r (en pixels, coordonnées dans l'image
d'origine). Les lignes dont cx/cy/r sont vides seront ignorées lors du
traitement (process).
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ExifTags
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from odf.opendocument import OpenDocumentSpreadsheet, load as odf_load
    from odf.table import Table, TableRow, TableCell
    from odf.text import P as odf_P
    _HAS_ODF = True
except ImportError:
    _HAS_ODF = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DETECTIONS_FIELDS = ["filename", "timestamp", "timestamp_source", "cx", "cy", "r",
                      "status", "method"]

# Tags EXIF à essayer, dans l'ordre de préférence (date de prise de vue réelle
# d'abord, puis dates de repli)
_EXIF_DATE_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime


def get_capture_time(path):
    """
    Retourne (timestamp_str, source) où timestamp_str est au format
    'YYYY-MM-DD HH:MM:SS' et source vaut 'exif' ou 'mtime' (repli sur la
    date de modification du fichier si l'EXIF est absent/illisible).
    """
    if _HAS_PIL:
        try:
            with Image.open(path) as im:
                exif = im.getexif()
                if exif:
                    for tag_id in _EXIF_DATE_TAGS:
                        val = exif.get(tag_id)
                        if val:
                            try:
                                dt = datetime.strptime(str(val), "%Y:%m:%d %H:%M:%S")
                                return dt.strftime("%Y-%m-%d %H:%M:%S"), "exif"
                            except ValueError:
                                continue
                    # certains fichiers rangent les données EXIF détaillées
                    # dans un IFD séparé (ex: certains smartphones)
                    try:
                        exif_ifd = exif.get_ifd(0x8769)  # Exif IFD
                        for tag_id in _EXIF_DATE_TAGS:
                            val = exif_ifd.get(tag_id)
                            if val:
                                dt = datetime.strptime(str(val), "%Y:%m:%d %H:%M:%S")
                                return dt.strftime("%Y-%m-%d %H:%M:%S"), "exif"
                    except Exception:
                        pass
        except Exception:
            pass

    try:
        dt = datetime.fromtimestamp(Path(path).stat().st_mtime)
        return dt.strftime("%Y-%m-%d %H:%M:%S"), "mtime"
    except Exception:
        return "", "inconnu"


# --------------------------------------------------------------------------
# Détection du disque solaire
# --------------------------------------------------------------------------

def _fit_circle_robust(pts, max_iter=4, outlier_frac=0.03, min_points=8):
    """
    Ajuste un cercle (méthode algébrique de Kasa) à un ensemble de points 2D,
    en rejetant itérativement les points aberrants. Plus fiable que
    cv2.minEnclosingCircle sur un arc de cercle partiel (ex: croissant fin),
    car minEnclosingCircle sous-estime systématiquement le rayon et décale le
    centre quand l'arc couvre moins de 180°.

    Retourne (cx, cy, r, nb_points_utilisés) ou None.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < min_points:
        return None

    cx = cy = r = None
    for _ in range(max_iter):
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
        b = x ** 2 + y ** 2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))

        resid = np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r)
        thresh = max(2.0, outlier_frac * r)
        keep = resid < thresh
        if keep.sum() < min_points or keep.sum() == len(pts):
            break
        pts = pts[keep]

    if cx is None:
        return None
    return cx, cy, r, len(pts)


def _score_circle(gray, cx, cy, r, n_angles=360, min_gradient=8.0, tol=3.0,
                   edge_direction="falling"):
    """
    Évalue la crédibilité d'un cercle candidat : quelle fraction de sa
    circonférence correspond réellement à une transition franche clair/sombre
    (ou sombre/clair, voir edge_direction) dans l'image, et quelle est la
    plus longue portion CONTIGUË de bord valide (en degrés).

    edge_direction="falling" : bord clair(intérieur)->sombre(extérieur), cas
    du limbe solaire.

    Sert à rejeter les faux positifs (bruit, artefacts, reflets, ajustements
    mal conditionnés) qui ne correspondent à aucun vrai bord de disque.
    """
    h, w = gray.shape[:2]
    if r <= 1:
        return 0.0, 0.0

    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    rs = np.array([r - tol, r, r + tol])
    map_x = (cx + np.outer(np.cos(thetas), rs)).astype(np.float32)
    map_y = (cy + np.outer(np.sin(thetas), rs)).astype(np.float32)
    valid = (map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)
    valid_all = valid.all(axis=1)
    if valid_all.sum() < n_angles * 0.5:
        return 0.0, 0.0

    samples = cv2.remap(gray.astype(np.float32), map_x, map_y,
                         interpolation=cv2.INTER_LINEAR)
    grad = samples[:, 2] - samples[:, 0]  # variation intérieur -> extérieur
    if edge_direction == "falling":
        good = valid_all & (-grad > min_gradient)
    else:
        good = valid_all & (grad > min_gradient)

    frac = float(good.sum()) / n_angles

    # plus longue série contiguë de True (en tenant compte du bouclage 0/360°)
    doubled = np.concatenate([good, good])
    max_run = 0
    cur = 0
    for v in doubled:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    max_run = min(max_run, n_angles)
    max_run_deg = max_run * (360.0 / n_angles)

    return frac, max_run_deg


def detect_sun_circle_coarse(img, min_radius_frac=0.05, max_radius_frac=0.48,
                              fixed_threshold=None,
                              min_valid_frac=0.02, min_arc_deg=6.0):
    """
    Détection GROSSIÈRE du cercle du disque solaire (point de départ pour le
    raffinement sous-pixel qui suit). Retourne (cx, cy, r, methode) ou None.

    Deux candidats sont calculés puis évalués avec un score de crédibilité
    (fraction du contour qui correspond à un vrai bord clair/sombre, et
    longueur du plus long arc continu de bord valide) :
      - Hough Circle Transform : fonctionne bien même sur un simple arc,
        mais peut ponctuellement accrocher un faux cercle (bruit, artefacts,
        reflets).
      - Repli : plus grand contour lumineux, puis ajustement de cercle
        robuste sur son enveloppe convexe (plus fiable qu'un simple cercle
        englobant sur un croissant fin).

    Le meilleur candidat est retenu seulement s'il dépasse un seuil minimal
    de crédibilité ; sinon la détection échoue (None) plutôt que de renvoyer
    un cercle non fiable.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    min_r = int(min_radius_frac * min(h, w))
    max_r = int(max_radius_frac * min(h, w))

    candidates = []  # (frac, arc_deg, cx, cy, r, method)

    # --- Candidat 1 : Hough Circle Transform -------------------------------
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=max(h, w),
        param1=60, param2=35, minRadius=min_r, maxRadius=max_r
    )
    if circles is not None and len(circles[0]) > 0:
        cx, cy, r = (float(v) for v in circles[0][0])
        frac, arc_deg = _score_circle(gray, cx, cy, r)
        candidates.append((frac, arc_deg, cx, cy, r, "hough"))

    # --- Candidat 2 : contour + ajustement de cercle robuste sur l'enveloppe
    #     convexe (remplace l'ancien cercle englobant minimal, biaisé sur un
    #     arc partiel) --------------------------------------------------
    if fixed_threshold is not None:
        _, thresh = cv2.threshold(blur, fixed_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= np.pi * (min_r ** 2) * 0.15:
            hull = cv2.convexHull(largest).reshape(-1, 2)
            fitted = _fit_circle_robust(hull)
            if fitted is not None:
                cx, cy, r, _npts = fitted
                if min_r * 0.5 <= r <= max_r * 1.3:
                    frac, arc_deg = _score_circle(gray, cx, cy, r)
                    candidates.append((frac, arc_deg, cx, cy, r, "contour"))

    if not candidates:
        return None

    # on retient le candidat le plus crédible (plus long arc continu valide)
    candidates.sort(key=lambda c: (c[1], c[0]), reverse=True)
    frac, arc_deg, cx, cy, r, method = candidates[0]

    if frac < min_valid_frac or arc_deg < min_arc_deg:
        return None  # aucun candidat suffisamment crédible

    return cx, cy, r, method


def refine_circle_subpixel(gray, cx0, cy0, r0, band_frac=0.15, n_angles=720,
                            min_gradient=8.0, max_iter=3, outlier_px=3.0,
                            edge_direction="falling"):
    """
    Affine une estimation grossière (cx0, cy0, r0) d'un cercle (limbe
    solaire ou bord de l'ombre lunaire) avec une précision sous-pixel.

    Principe : on échantillonne l'intensité le long de centaines de rayons
    partant du centre approximatif, dans une bande étroite autour de r0. Sur
    chaque rayon on localise la transition (le vrai bord) avec une précision
    sous-pixel (interpolation parabolique du gradient). On ajuste ensuite un
    cercle par régression sur l'ensemble des points de bord trouvés, en
    rejetant itérativement les points aberrants qui ne sont pas sur le vrai
    cercle.

    edge_direction :
      - "falling" : cherche une transition clair -> sombre en s'éloignant du
        centre (cas du limbe solaire : le disque est plus clair que le ciel).

    Retourne (cx, cy, r, nb_points_utilisés) ou None si le raffinement échoue
    (pas assez de points de bord fiables trouvés).
    """
    h, w = gray.shape[:2]
    band = max(4.0, r0 * band_frac)
    rs = np.arange(-band, band + 1.0, 1.0) + r0
    n_r = len(rs)
    if n_r < 5:
        return None

    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    map_x = (cx0 + np.outer(cos_t, rs)).astype(np.float32)
    map_y = (cy0 + np.outer(sin_t, rs)).astype(np.float32)
    valid_mask = (map_x >= 1) & (map_x <= w - 2) & (map_y >= 1) & (map_y <= h - 2)

    samples = cv2.remap(gray.astype(np.float32), map_x, map_y,
                         interpolation=cv2.INTER_LINEAR)

    edge_points = []
    for i in range(n_angles):
        if valid_mask[i].sum() < n_r * 0.6:
            continue
        vals = samples[i]
        grad = np.gradient(vals)
        if edge_direction == "falling":
            idx = int(np.argmin(grad))  # transition clair -> sombre
            strength = -grad[idx]
        else:
            idx = int(np.argmax(grad))  # transition sombre -> clair
            strength = grad[idx]
        if idx <= 0 or idx >= n_r - 1:
            continue
        if strength < min_gradient:
            continue
        g0, g1, g2 = grad[idx - 1], grad[idx], grad[idx + 1]
        denom = (g0 - 2 * g1 + g2)
        offset = 0.5 * (g0 - g2) / denom if abs(denom) > 1e-6 else 0.0
        offset = float(np.clip(offset, -1, 1))
        real_r = rs[idx] + offset
        edge_points.append((cx0 + real_r * cos_t[i], cy0 + real_r * sin_t[i]))

    if len(edge_points) < max(30, n_angles * 0.10):
        return None

    pts = np.array(edge_points)
    cx, cy, r = cx0, cy0, r0
    for _ in range(max_iter):
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
        b = x ** 2 + y ** 2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        r = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))

        resid = np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r)
        keep = resid < outlier_px
        if keep.sum() < 30 or keep.sum() == len(pts):
            break
        pts = pts[keep]

    return cx, cy, r, len(pts)


def detect_sun_circle(img, min_radius_frac=0.05, max_radius_frac=0.48,
                       fixed_threshold=None, refine=True,
                       refine_band_frac=0.15, refine_min_gradient=8.0):
    """
    Détecte le disque solaire : détection grossière puis raffinement
    sous-pixel (sauf si refine=False). Retourne (cx, cy, r, methode) ou None.
    """
    coarse = detect_sun_circle_coarse(
        img, min_radius_frac=min_radius_frac, max_radius_frac=max_radius_frac,
        fixed_threshold=fixed_threshold
    )
    if coarse is None:
        return None
    cx0, cy0, r0, method = coarse

    if not refine:
        return cx0, cy0, r0, method

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    refined = refine_circle_subpixel(
        gray, cx0, cy0, r0,
        band_frac=refine_band_frac, min_gradient=refine_min_gradient
    )
    if refined is None:
        return cx0, cy0, r0, f"{method} (raffinement échoué)"

    cx, cy, r, npts = refined
    frac, arc_deg = _score_circle(gray, cx, cy, r,
                                   min_gradient=refine_min_gradient)
    if frac < 0.02 or arc_deg < 6.0:
        # le raffinement a dérivé vers un résultat non crédible : on
        # préfère l'estimation grossière (déjà validée) plutôt qu'un
        # résultat aberrant.
        return cx0, cy0, r0, f"{method} (raffinement rejeté)"

    return cx, cy, r, f"{method}+subpixel({npts}pts)"


# --------------------------------------------------------------------------
# Lecture / écriture du fichier .ods de détections
# --------------------------------------------------------------------------

def read_detections_ods(path):
    """
    Lit le fichier tableur .ods de détections. Retourne un dict :
        { filename: {"cx": float|None, "cy": float|None, "r": float|None,
                      "status": str, "method": str,
                      "timestamp": str, "timestamp_source": str} }
    Les cellules vides ou non numériques pour cx/cy/r donnent None.
    Fonctionne quel que soit l'ordre des colonnes (identifiées par leur nom
    dans la ligne d'en-tête), pour rester robuste si le fichier a été
    réenregistré par un tableur.
    """
    if not _HAS_ODF:
        print("ERREUR : le module 'odfpy' est requis pour lire/écrire les "
              "fichiers .ods. Installez-le avec : pip install odfpy "
              "(ou : pip install odfpy --break-system-packages)")
        sys.exit(1)

    path = Path(path)
    if not path.exists():
        return {}

    doc = odf_load(str(path))
    tables = doc.spreadsheet.getElementsByType(Table)
    if not tables:
        return {}
    table_rows = tables[0].getElementsByType(TableRow)
    if not table_rows:
        return {}

    def cell_text(cell):
        ps = cell.getElementsByType(odf_P)
        return "".join(str(p) for p in ps).strip() if ps else ""

    def row_values(row):
        vals = []
        for cell in row.getElementsByType(TableCell):
            repeat = int(cell.getAttribute("numbercolumnsrepeated") or 1)
            vals.extend([cell_text(cell)] * repeat)
        return vals

    header = [h.strip().lower() for h in row_values(table_rows[0])]

    result = {}
    for row in table_rows[1:]:
        vals = row_values(row)
        if not vals or not vals[0].strip():
            continue  # ligne vide (padding éventuel du tableur)
        rec = {header[i]: vals[i] for i in range(min(len(header), len(vals)))}
        fname = rec.get("filename", "").strip()
        if not fname:
            continue

        def parse_float(key):
            v = (rec.get(key) or "").strip()
            if v == "":
                return None
            try:
                return float(v.replace(",", "."))  # tolère la virgule décimale FR
            except ValueError:
                return None

        result[fname] = {
            "cx": parse_float("cx"),
            "cy": parse_float("cy"),
            "r": parse_float("r"),
            "status": (rec.get("status") or "").strip(),
            "method": (rec.get("method") or "").strip(),
            "timestamp": (rec.get("timestamp") or "").strip(),
            "timestamp_source": (rec.get("timestamp_source") or "").strip(),
        }
    return result


def write_detections_ods(path, rows):
    """
    Écrit le fichier tableur .ods de détections. `rows` est une liste de
    dicts avec les clés de DETECTIONS_FIELDS (cx/cy/r peuvent être None).
    Les colonnes numériques sont écrites comme des
    nombres (et non du texte), pour un confort d'édition dans le tableur
    (tri, format, etc.).
    """
    if not _HAS_ODF:
        print("ERREUR : le module 'odfpy' est requis pour lire/écrire les "
              "fichiers .ods. Installez-le avec : pip install odfpy "
              "(ou : pip install odfpy --break-system-packages)")
        sys.exit(1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    numeric_cols = {"cx", "cy", "r"}

    def make_cell(key, value):
        if value is None or value == "":
            return TableCell()
        if key in numeric_cols:
            fval = float(value)
            cell = TableCell(valuetype="float", value=fval)
            cell.addElement(odf_P(text=f"{fval:.2f}"))
            return cell
        cell = TableCell(valuetype="string")
        cell.addElement(odf_P(text=str(value)))
        return cell

    doc = OpenDocumentSpreadsheet()
    table = Table(name="detections")

    header_row = TableRow()
    for field in DETECTIONS_FIELDS:
        cell = TableCell(valuetype="string")
        cell.addElement(odf_P(text=field))
        header_row.addElement(cell)
    table.addElement(header_row)

    for row in rows:
        tr = TableRow()
        for field in DETECTIONS_FIELDS:
            tr.addElement(make_cell(field, row.get(field)))
        table.addElement(tr)

    doc.spreadsheet.addElement(table)
    doc.save(str(path))


# --------------------------------------------------------------------------
# Utilitaires fichiers image
# --------------------------------------------------------------------------

def list_images(input_dir):
    input_dir = Path(input_dir)
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    return files


def imread_any(path):
    """Lecture robuste (gère les chemins avec caractères spéciaux/accents)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_any(path, img, quality=95):
    ext = Path(path).suffix.lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(ext if ext else ".jpg", img, params)
    if not ok:
        raise IOError(f"Échec d'encodage pour {path}")
    buf.tofile(str(path))


# --------------------------------------------------------------------------
# Commande : preview
# --------------------------------------------------------------------------

def cmd_preview(args):
    files = list_images(args.input)
    if not files:
        print(f"Aucune image trouvée dans {args.input}")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    detections_path = Path(args.detections) if args.detections else out_dir / "detections.ods"

    # On relit un éventuel fichier existant pour PRÉSERVER les corrections
    # manuelles déjà faites par l'utilisateur (relance itérative).
    existing = read_detections_ods(detections_path)

    rows = []
    n_ok, n_fail, n_kept = 0, 0, 0

    for f in files:
        prev = existing.get(f.name)

        # Horodatage : on réutilise la valeur déjà présente dans le fichier .ods
        # (évite de rouvrir le fichier inutilement à chaque relance), sinon
        # on l'extrait de l'EXIF (ou, à défaut, de la date de modification).
        if prev and prev.get("timestamp"):
            timestamp, ts_source = prev["timestamp"], prev.get("timestamp_source", "")
        else:
            timestamp, ts_source = get_capture_time(f)

        if prev and prev["cx"] is not None and prev["cy"] is not None and prev["r"] is not None:
            # Valeur déjà présente (probablement corrigée à la main) : on la conserve.
            cx, cy, r = prev["cx"], prev["cy"], prev["r"]
            method = prev["method"] or "manuel (conservé)"
            status = "ok"
            n_kept += 1
        else:
            img = imread_any(f)
            if img is None:
                print(f"[IGNORÉ] Impossible de lire {f.name}")
                continue
            result = detect_sun_circle(
                img,
                min_radius_frac=args.min_radius_frac,
                max_radius_frac=args.max_radius_frac,
                fixed_threshold=args.threshold,
                refine=not args.no_refine,
                refine_band_frac=args.refine_band_frac,
                refine_min_gradient=args.refine_min_gradient,
            )
            if result is None:
                cx = cy = r = None
                method = "-"
                status = "echec"
                n_fail += 1
                print(f"[ÉCHEC] {f.name} : disque non détecté")
            else:
                cx, cy, r, method = result
                status = "ok"
                n_ok += 1
                print(f"[OK] {f.name} : centre=({cx:.0f},{cy:.0f}) r={r:.0f} ({method})")

        rows.append({
            "filename": f.name, "timestamp": timestamp, "timestamp_source": ts_source,
            "cx": cx, "cy": cy, "r": r,
            "status": status, "method": method,
        })

        # Image annotée pour vérification visuelle (si on a des coordonnées)
        if cx is not None:
            img = imread_any(f)
            if img is not None:
                vis = img.copy()
                cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 3)
                cv2.drawMarker(vis, (int(cx), int(cy)), (0, 0, 255),
                                cv2.MARKER_CROSS, 30, 3)
                label = f"soleil: {method} r={r:.0f}"
                cv2.putText(vis, label, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                imwrite_any(out_dir / f.name, vis)

    write_detections_ods(detections_path, rows)

    print(f"\nRésumé : {n_ok} détections auto., {n_kept} valeurs conservées "
          f"(déjà présentes), {n_fail} échecs.")
    print(f"Fichier de détections : {detections_path}")
    print(f"Images annotées        : {out_dir}")
    if n_fail:
        print(f"\n{n_fail} photo(s) sans détection (probablement des photos de "
              f"totalité, ou contraste insuffisant). Ouvrez {detections_path} "
              f"dans LibreOffice Calc (ou un autre tableur compatible .ods) et "
              f"remplissez manuellement les colonnes cx, cy, "
              f"r (en pixels, sur l'image d'origine) pour ces lignes, puis "
              f"relancez 'process' avec ce fichier.")


# --------------------------------------------------------------------------
# Annotation date/heure
# --------------------------------------------------------------------------

def draw_timestamp(img, timestamp):
    """Affiche la date et l'heure de prise de vue en bas de l'image."""
    if not timestamp:
        return img

    h, w = img.shape[:2]
    label = timestamp.replace("-", "/")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(1.5, w / 1600.0))
    thickness = max(1, int(round(font_scale * 2)))
    margin = max(20, int(round(w * 0.02)))

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x = (w - tw) // 2
    y = h - margin - baseline

    pad = max(8, int(round(font_scale * 8)))
    cv2.rectangle(
        img,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        img, label, (x, y), font, font_scale,
        (255, 255, 255), thickness, cv2.LINE_AA
    )
    return img


# --------------------------------------------------------------------------
# Commande : process
# --------------------------------------------------------------------------

def cmd_process(args):
    files = list_images(args.input)
    if not files:
        print(f"Aucune image trouvée dans {args.input}")
        return

    detections = read_detections_ods(args.detections)
    if not detections:
        print(f"Fichier de détections introuvable ou vide : {args.detections}")
        print("Lancez d'abord la commande 'preview' pour le générer.")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Étape 1 : valeurs valides + taille max des images
    valid = {}
    max_w, max_h = 0, 0
    for f in files:
        info = detections.get(f.name)
        if info is None:
            print(f"[IGNORÉ] {f.name} : absente du fichier de détections")
            continue
        if info["cx"] is None or info["cy"] is None or info["r"] is None:
            print(f"[IGNORÉ] {f.name} : cx/cy/r manquants dans le fichier "
                  f"(statut: {info['status'] or 'vide'}) -> à corriger à la main")
            continue
        img_probe = imread_any(f)
        if img_probe is None:
            print(f"[IGNORÉ] Impossible de lire {f.name}")
            continue
        h, w = img_probe.shape[:2]
        max_w, max_h = max(max_w, w), max(max_h, h)
        valid[f.name] = (info["cx"], info["cy"], info["r"])

    if not valid:
        print("Aucune image exploitable (voir les lignes du fichier .ods), arrêt.")
        return

    # Étape 2 : rayon cible et taille du canevas
    radii = [r for (_, _, r) in valid.values()]
    target_r = args.target_radius if args.target_radius else float(np.median(radii))

    # Les sorties du process sont volontairement CARRÉES. Le carré n'est
    # surtout pas ajusté au cercle solaire : on réserve une bordure sombre
    # tout autour du Soleil. Par défaut, la bordure vaut 0.5 rayon solaire,
    # soit un côté de 3 * rayon cible.
    if args.canvas:
        canvas_w, canvas_h = map(int, args.canvas.lower().split("x"))
        if canvas_w != canvas_h:
            raise ValueError(
                "--canvas doit être carré (ex: 2048x2048), "
                "car les images du process sont toujours carrées."
            )
    else:
        border_px = target_r * args.border_radius_factor
        canvas_w = canvas_h = int(round(2 * (target_r + border_px)))

    print(f"\n{len(valid)} images à traiter sur {len(files)} trouvées.")
    print(f"Rayon cible retenu : {target_r:.1f} px")
    print(f"Taille du carré     : {canvas_w}x{canvas_h}")
    print(f"Bordure autour du Soleil : "
          f"{max(0, (canvas_w / 2) - target_r):.1f} px")

    # Étape 3 : recadrage / mise à l'échelle de chaque image
    n_done = 0
    for f in files:
        if f.name not in valid:
            continue
        img = imread_any(f)
        if img is None:
            continue
        cx, cy, r = valid[f.name]
        scale = target_r / r

        M = np.array([
            [scale, 0, canvas_w / 2 - scale * cx],
            [0, scale, canvas_h / 2 - scale * cy],
        ], dtype=np.float64)

        interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        aligned = cv2.warpAffine(
            img, M, (canvas_w, canvas_h),
            flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        # Affiche la date et l'heure de prise de vue en bas de chaque image.
        timestamp = detections.get(f.name, {}).get("timestamp", "")
        aligned = draw_timestamp(aligned, timestamp)

        imwrite_any(out_dir / f.name, aligned, quality=args.quality)
        n_done += 1

    print(f"\n{n_done} images alignées enregistrées dans : {out_dir}")


# --------------------------------------------------------------------------
# Commande : video
# --------------------------------------------------------------------------

def cmd_video(args):
    files = list_images(args.input)
    if not files:
        print(f"Aucune image trouvée dans {args.input}")
        return

    first = imread_any(files[0])
    if first is None:
        print(f"Impossible de lire {files[0]}")
        return
    h, w = first.shape[:2]

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        ok = _make_video_ffmpeg(ffmpeg_path, files, args.output, args.fps, w, h)
        if ok:
            return
        print("(ffmpeg a échoué, repli sur l'encodeur intégré OpenCV...)\n")

    _make_video_opencv(files, args.output, args.fps, w, h)


def _make_video_ffmpeg(ffmpeg_path, files, output, fps, w, h):
    """
    Encode la vidéo avec ffmpeg en H.264 (libx264) + pixel format yuv420p,
    dans un conteneur MP4 avec le flag +faststart. C'est la combinaison la
    plus largement compatible (lecteurs mobiles, Google Photos, réseaux
    sociaux...) — contrairement au codec 'mp4v' d'OpenCV, souvent mal ou pas
    du tout lu par ces plateformes. Les dimensions sont forcées à des
    valeurs paires (requis par yuv420p).
    """
    with tempfile.TemporaryDirectory(prefix="eclipse_video_") as tmpdir:
        tmpdir = Path(tmpdir)
        n = 0
        for f in files:
            img = imread_any(f)
            if img is None:
                print(f"[IGNORÉ] {f.name}")
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            imwrite_any(tmpdir / f"frame_{n:06d}.jpg", img, quality=95)
            n += 1
        if n == 0:
            print("Aucune image exploitable.")
            return False

        cmd = [
            ffmpeg_path, "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%06d.jpg"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Erreur ffmpeg :")
            print(result.stderr[-2000:])
            return False

        print(f"Vidéo créée : {output} ({n} images, {fps} fps, H.264/yuv420p via ffmpeg)")
        return True


def _make_video_opencv(files, output, fps, w, h):
    """
    Repli si ffmpeg n'est pas installé sur la machine. Utilise le codec
    intégré à OpenCV (mp4v, MPEG-4 Part 2) : la vidéo produite est valide
    mais souvent mal lue par les lecteurs mobiles et certaines plateformes
    (Google Photos notamment). Installez ffmpeg pour un encodage H.264
    correctement compatible.
    """
    w_even, h_even = w - (w % 2), h - (h % 2)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (w_even, h_even))

    n = 0
    for f in files:
        img = imread_any(f)
        if img is None:
            print(f"[IGNORÉ] {f.name}")
            continue
        if img.shape[:2] != (h_even, w_even):
            img = cv2.resize(img, (w_even, h_even))
        writer.write(img)
        n += 1

    writer.release()
    print(f"Vidéo créée : {output} ({n} images, {fps} fps, codec mp4v via OpenCV)")
    print("ATTENTION : ffmpeg n'a pas été trouvé sur cette machine. Le codec "
          "mp4v utilisé ici n'est pas fiable sur Google Photos et de nombreux "
          "lecteurs mobiles. Pour un résultat largement compatible, installez "
          "ffmpeg puis relancez cette commande :")
    print("  sudo apt install ffmpeg   (Debian/Ubuntu)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Aligne des photos d'éclipse solaire (centrage + mise à "
                     "l'échelle du disque solaire) et crée un time-lapse."
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_prev = sub.add_parser(
        "preview",
        help="Détecter le disque solaire et écrire un fichier .ods corrigible à la main"
    )
    p_prev.add_argument("--input", required=True, help="Dossier des photos source")
    p_prev.add_argument("--output", required=True,
                         help="Dossier de sortie (images annotées + fichier detections.ods)")
    p_prev.add_argument("--detections", default=None,
                         help="Chemin du fichier .ods de détections "
                              "(défaut : <output>/detections.ods). S'il existe déjà, "
                              "les valeurs qu'il contient sont conservées.")
    p_prev.add_argument("--threshold", type=int, default=None,
                         help="Seuil de luminosité fixe (0-255) pour la méthode "
                              "de repli. Par défaut : seuil automatique (Otsu).")
    p_prev.add_argument("--min-radius-frac", type=float, default=0.05,
                         help="Rayon minimal du disque, en fraction de la plus "
                              "petite dimension de l'image (défaut: 0.05)")
    p_prev.add_argument("--max-radius-frac", type=float, default=0.48,
                         help="Rayon maximal du disque, en fraction de la plus "
                              "petite dimension de l'image (défaut: 0.48)")
    p_prev.add_argument("--no-refine", action="store_true",
                         help="Désactive le raffinement sous-pixel du bord "
                              "(utilise uniquement la détection grossière)")
    p_prev.add_argument("--refine-band-frac", type=float, default=0.15,
                         help="Largeur de la bande de recherche du bord autour "
                              "du rayon grossier, en fraction de ce rayon "
                              "(défaut: 0.15). Augmentez si la détection "
                              "grossière est très imprécise.")
    p_prev.add_argument("--refine-min-gradient", type=float, default=8.0,
                         help="Sensibilité minimale du contraste requis pour "
                              "considérer un bord comme valide (défaut: 8.0). "
                              "Diminuez pour un bord flou/voilé (nuages fins), "
                              "augmentez si des faux bords sont détectés.")
    p_prev.set_defaults(func=cmd_preview)

    p_proc = sub.add_parser(
        "process",
        help="Recadrer/aligner toutes les images à partir du fichier .ods de détections"
    )
    p_proc.add_argument("--input", required=True, help="Dossier des photos source")
    p_proc.add_argument("--output", required=True, help="Dossier de sortie (images alignées)")
    p_proc.add_argument("--detections", required=True,
                         help="Fichier .ods de détections généré (et éventuellement "
                              "corrigé) par la commande 'preview'")
    p_proc.add_argument("--target-radius", type=float, default=None,
                         help="Rayon cible en pixels (défaut : médiane des rayons du fichier .ods)")
    p_proc.add_argument("--canvas", default=None,
                         help="Taille du carré de sortie, ex: 2048x2048 "
                              "(doit être carré). Par défaut : 3 fois le rayon "
                              "cible sur chaque côté, avec une bordure de 0.5 rayon.")
    p_proc.add_argument("--border-radius-factor", type=float, default=0.5,
                         help="Largeur de la bordure sombre autour du Soleil, "
                              "en fraction du rayon solaire (défaut : 0.5). "
                              "Ignoré si --canvas est fourni.")
    p_proc.add_argument("--quality", type=int, default=95, help="Qualité JPEG (0-100)")
    p_proc.set_defaults(func=cmd_process)

    p_vid = sub.add_parser("video", help="Assembler les images alignées en vidéo")
    p_vid.add_argument("--input", required=True, help="Dossier des images alignées")
    p_vid.add_argument("--output", required=True, help="Fichier vidéo de sortie (ex: eclipse.mp4)")
    p_vid.add_argument("--fps", type=float, default=24, help="Images par seconde (défaut: 24)")
    p_vid.set_defaults(func=cmd_video)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
