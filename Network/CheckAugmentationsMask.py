import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Laden der Bilddatei
# image_path = r'C:\Users\Pascal\PycharmProjects\ImageSelectorBA\a_m\img_8_10.png'
# image_path = r'C:\Users\Pascal\PycharmProjects\ImageSelectorBA\synthetic_train_masks\image_4.png'
# image_path = r'C:\Users\Pascal\PycharmProjects\ImageSelectorBA\TestForMasks - masks2\img_1_0.png'
image_path = r'C:\Users\Pascal\PycharmProjects\ImageSelectorBA\Dataset 1 Augmented (small) - masks\img_152_0.png'
mask = Image.open(image_path)

# Konvertiere Bild in ein numpy Array
mask_np = np.array(mask)

# Anzeige der Maske
plt.imshow(mask_np)
plt.title('Segmentierungsmaske')
plt.axis('off')  # Keine Achsen anzeigen
plt.show()