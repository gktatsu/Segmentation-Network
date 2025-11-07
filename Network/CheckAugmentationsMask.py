import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Laden der Bilddatei
image_path = r'C:\masks\img_152_0.png'
mask = Image.open(image_path)

# Konvertiere Bild in ein numpy Array
mask_np = np.array(mask)

# Anzeige der Maske
plt.imshow(mask_np)
plt.title('Segmentierungsmaske')
plt.axis('off')  # Keine Achsen anzeigen
plt.show()
