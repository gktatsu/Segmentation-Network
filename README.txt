Segmentierungsnetzwerk

Das Segmentierungsnetzwerk wurde entworfen, um Datensätze bestehend aus elektronenmikroskopischen Daten
auswerten und evaluieren zu können. Es verwendet eine U-Net Architektur, um eine Segmentierung aus einem
Datensatz zu erstellen, der aus Test-, Validierungs- und Trainingsdaten besteht. Das Netzwerk ist in
Python implementiert und verwendet die PyTorch-Bibliothek für Deep Learning.


Um das Netzwerk zu trainieren wird die Datei 'train.py' folgendermaßen ausgeführt:

- Lokal: python train.py
- Cluster: Verwende die run.sh Datei



Requirements (requirements.txt):

- PyTorch
- TorchVision
- scikit-learn
- tqdm
- matplotlib
- numpy
- wandb



Konfiguration:

Die Konfiguration des Netzwerks erfolgt über die Datei 'config.py'. Hier können die Hyperparameter wie die
Anzahl der Epochen, die Lernrate und die Batch-Größe festgelegt werden. Die festgelegten Werte entsprechen
denen aus dem U-Net Paper von Ronneberger et al. und eignen sich hervorragend für die Segmentierung.


Datensatz:

Das Netzwerk benötigt eine Aufteilung auf Trainings-, Test- und Validierungdaten. Der Pfad zum Datensatz
muss vor dem Training individuell angepasst werden. Das Netzwerk erwartet einen Datensatz, der in folgender
Struktur vorliegt:

Dataset/
│
├── train_images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
│
├── train_masks/
│   ├── mask1.png
│   ├── mask2.png
│   └── ...
│
├── validation_images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
│
├── validation_masks/
│   ├── mask1.png
│   ├── mask2.png
│   └── ...
│
├── test_images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
│
└── test_masks/
    ├── mask1.png
    ├── mask2.png
    └── ...


Ausgabe:
Die Ausgaben des Trainings werden in dem Verzeichnis 'output' gespeichert. Hier werden unter anderem die
trainierten Modelle sowie Logs und Metriken gespeichert.