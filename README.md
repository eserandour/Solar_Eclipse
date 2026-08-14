# Solar_Eclipse
Recadre et aligne une série de photos d'éclipse solaire pour que le disque solaire (entier ou partiellement masqué par la Lune) soit toujours centré et à la même taille apparente sur toutes les images. Permet ensuite de créer un time-lapse.

# Installation

Le programme a besoin des bibliothèques Python suivantes : python3-opencv, python3-numpy, python3-pil, python3-odf

# Utilisation en 3 étapes

<b>python3 eclipse_align.py preview --input photos/ --output apercu/</b><br>
Corriger apercu/detections.ods si besoin (avec LibreOffice Calc).<br>
On peut utiliser Gimp pour calculer les coordonnées du disque solaire sur l'image.

<b>python3 eclipse_align.py process --input photos/ --output alignees/ --detections apercu/detections.ods</b><br>
Options : --flip-horizontal --rotate90-cw (quand les 2 options sont utilisées ensemble, la rotation est appliquée avant le flip)

<b>python3 eclipse_align.py video --input alignees/ --output eclipse.mp4 --fps 6</b>
