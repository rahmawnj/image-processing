import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk melakukan histogram equalization
def histogram_equalization(img):
    histogram, bin_edges = np.histogram(img.flatten(), 256, [0, 256])
    cdf = histogram.cumsum()
    cdf_normalized = 255 * cdf / cdf.max()
    img_equalized = np.interp(img.flatten(), bin_edges[:-1], cdf_normalized)
    return img_equalized.reshape(img.shape).astype('uint8')

# Fungsi untuk menyesuaikan kontras
def adjust_contrast(img, factor):
    mean = np.mean(img)
    return np.clip((img - mean) * factor + mean, 0, 255).astype('uint8')

# Memuat citra
image_path = 'assets/low-contrast-image.jpg'  # Sesuaikan dengan jalur file citra Anda
image = imageio.imread(image_path)

# Memproses citra
image_equalized = histogram_equalization(image)
image_contrast_adjusted = adjust_contrast(image, 1.5)

# Menampilkan citra asli, hasil equalization, dan hasil penyesuaian kontras
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image_equalized, cmap='gray', vmin=0, vmax=255)
plt.title('Histogram Equalized Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image_contrast_adjusted, cmap='gray', vmin=0, vmax=255)
plt.title('Contrast Adjusted (1.5) Image')
plt.axis('off')

plt.show()
