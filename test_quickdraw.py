import os
import matplotlib.pyplot as plt
import numpy as np

dataset_path = "data/npy"
class_name = "sun.npy"

file_path = os.path.join(dataset_path, class_name)

images = np.load(file_path)
print(f"Loaded {images.shape[0]} images of class {class_name}")

first_image = images[0].reshape(28,28)

plt.imshow(first_image, cmap="gray")
plt.axis('off')
plt.show()