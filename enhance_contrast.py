import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(img):
    histogram, bin_edges = np.histogram(img.flatten(), 256, [0, 256])
    cdf = histogram.cumsum()
    cdf_normalized = 255 * cdf / cdf.max() 
    img_equalized = np.interp(img.flatten(), bin_edges[:-1], cdf_normalized)
    return img_equalized.reshape(img.shape).astype('uint8')

image_path = 'assets/low-contrast-image.jpg'
image = imageio.imread(image_path)

image_equalized = histogram_equalization(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Citra Asli')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_equalized, cmap='gray', vmin=0, vmax=255)
plt.title('Citra Hasil Equalisasi Histogram')
plt.axis('off')

plt.show()
