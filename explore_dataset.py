import os
import cv2
import matplotlib.pyplot as plt

# Path to dataset
dataset_path = "dataset"

# Get list of disease folders
folders = os.listdir(dataset_path)

print("Disease Categories:")
print(folders)

# Select the first disease folder
first_folder = folders[0]
folder_path = os.path.join(dataset_path, first_folder)

# Select the first image in that folder
image_name = os.listdir(folder_path)[0]
image_path = os.path.join(folder_path, image_name)

# Read image using OpenCV
image = cv2.imread(image_path)

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(image)
plt.title(first_folder)
plt.axis("off")
plt.show()