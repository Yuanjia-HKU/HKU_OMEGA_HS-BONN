import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the two images
img_path = './ASM_simulation_results/norm2D/'
image2 = cv2.imread(img_path + '1_input_layer.tiff', cv2.IMREAD_GRAYSCALE)
image2 = image2[85:340, 60:360]
# pattern = cv2.resize(pattern, (255, 300))
# image1 = (pattern * 255).astype('uint8')
image1 = cv2.imread(img_path + '1_input_layer_5_blank_layer_2.tiff', cv2.IMREAD_GRAYSCALE)
image1 = image1[85:340, 60:360]

# equalized_image1 = cv2.equalizeHist(image1[85:340, 60:360])
equalized_image1 = cv2.equalizeHist(image1)
equalized_image2 = cv2.equalizeHist(image2)

# Flatten the arrays and remove zeros
flat_image1 = equalized_image1.flatten().astype(np.float32) / 255
flat_image2 = equalized_image2.flatten().astype(np.float32) / 255

# Remove zero values to avoid log(0)
mask = (flat_image1 > 0) & (flat_image2 > 0)
log_image1 = np.log(flat_image1[mask])
log_image2 = np.log(flat_image2[mask])

mask = (log_image1 >= -6) & (log_image2 >= -4.5)
log_image1 = log_image1[mask]
log_image2 = log_image2[mask]

# Create a scatter plot
plt.scatter(log_image1, log_image2, alpha=0.5)
plt.xlabel('log(Image 1)')
plt.ylabel('log(Image 2)')
plt.title('Log-Log Plot of Two Images (Filtered)')

# Fit a linear model
slope, intercept = np.polyfit(log_image1, log_image2, 1)

# Plot the fitted line
x_vals = np.linspace(min(log_image1), max(log_image1), 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
plt.legend()
plt.grid()

# Set axis limits to include all points
plt.xlim([min(log_image1) - 0.1, max(log_image1) + 0.1])
plt.ylim([min(log_image2) - 0.1, max(log_image2) + 0.1])

plt.show()

# Output the power-law exponent
print(f'Power-law exponent: {slope}')