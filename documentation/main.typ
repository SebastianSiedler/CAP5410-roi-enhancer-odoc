#set page(paper: "us-letter", margin: 1.5cm)


We are using the REFUGE Challange dataset @src_REFUGE_dataset.
- it contains 1200 images with OD and OC annotations
- it is split into 400 training, 400 validation, and 400 test images


== Data Preparation

Aus dem G1020 Dataset haben 234 Bilder nicht alle drei Klassen in den Masken (Hintergrund, OD, OC). Bei zwei Bildern fehlen sogar beide Strukturen komplett.

Die Crops haben verschiedene Größen:
87 verschiedene Größen in nur 100 Samples!
Von 200x200 bis 645x645 Pixel
Images und Masks passen zusammen ✅
Aber für Batching müssen alle gleich groß sein ❌


Wir nutzen Albumentations um die Bilder zu Runtime zu transformieren.
Das heißt in jeder Epoche sieht beim training (nicht validation!) das Bild anders aus. 2000 trainingsbilder, 10 epochen. => 20000 verschiedene Bilder.

#bibliography("works.bib", style: "ieee")



- versucht um modell zu verbessern: 
    - cup gewichtung auf 2.0 setzen
    - mehr epochen (100 statt 50)