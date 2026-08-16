# Solar_Eclipse
Recadre et aligne une série de photos d'éclipse solaire pour que le disque solaire (entier ou partiellement masqué par la Lune) soit toujours centré et à la même taille apparente sur toutes les images. Permet ensuite de créer un time-lapse.

# Installation

Le programme a besoin des bibliothèques Python suivantes : python3-opencv, python3-numpy, python3-pil, python3-odf.

Les photos sont à ranger dans un répertoire "photos".

# Utilisation en 3 étapes

<b>python3 eclipse_align.py preview --input photos/ --output apercu/</b><br>
Corriger apercu/detections.ods si besoin (avec LibreOffice Calc).<br>
On peut utiliser Gimp pour calculer à la main les coordonnées du disque solaire sur l'image.

<b>python3 eclipse_align.py process --input photos/ --output alignees/ --detections apercu/detections.ods</b><br>
Options : --flip-horizontal --rotate90-cw

<b>python3 eclipse_align.py video --input alignees/ --output eclipse.mp4 --fps 6</b>

# Remarque

Sous Linux, pour extraire des images d'une vidéo :<br>
<b>ffmpeg -i video.mp4 image_%06d.bmp</b> (ou png) (toutes les images)<br>
<b>ffmpeg -i video.mp4 -vf fps=6 image_%06d.bmp</b> (6 images par seconde)
