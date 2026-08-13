# Solar_Eclipse
Recadre et aligne une série de photos d'éclipse solaire pour que le disque solaire (entier ou partiellement masqué par la Lune) soit toujours centré et à la même taille apparente sur toutes les images. Permet ensuite de créer un time-lapse.

# Utilisation
python3 eclipse_align.py preview --input photos/ --output apercu/
# => corriger apercu/detections.ods si besoin (avec LibreOffice Calc).
# On peut utiliser Gimp pour calculer les coordonnées du disque solaire sur l'image.
python3 eclipse_align.py process --input photos/ --output alignees/ --detections apercu/detections.ods
python3 eclipse_align.py video --input alignees/ --output eclipse.mp4 --fps 6
