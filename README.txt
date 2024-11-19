Segmentierungsnetzwerk

Um das Netzwerk zu trainieren wird die Datei 'train.py' folgendermaßen ausgeführt:

- Lokal: python train.py
- Cluster: Verwende die run.sh Datei



Requirements (requirements.txt / Segmentation.yaml):

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

Weitere Informationen zum Start des Codes und dem Speicherort
der Datensätze sind in Documentation.txt zu finden.