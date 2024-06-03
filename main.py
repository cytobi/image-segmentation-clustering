from sklearn.cluster import KMeans
import numpy as np
import cv2

# param
n_clusters = 3
balance = 100

# load image
img_name = "nattu-adnan-vvHRdOwqHcg-unsplash"
image = cv2.imread(f"img/{img_name}.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"Image shape: {image.shape}")

x, y = image.shape[:2]

# generate linearly spaced values between 0 and 1 for x and y
x_coords = np.linspace(0, balance, x)
y_coords = np.linspace(0, balance, y)

# create a grid of coordinates
xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

# stack the grids to create the final array
coordinates = np.stack((xx, yy), axis=-1)
print(f"Coordinates shape: {coordinates.shape}")

# combine image and coordinates
data = np.concatenate((coordinates, image), axis=-1).reshape(-1, 5)
print(f"Data shape: {data.shape}")

# fit kmeans
kmeans = KMeans(n_clusters=n_clusters).fit(data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
print(f"Labels shape: {labels.shape}")

# reverse the reshaping
labels = labels.reshape(x, y)
print(f"Labels shape after reshaping: {labels.shape}")

# create the segmented image
segmented_image = np.zeros_like(image, dtype=np.float32)
for i in range(n_clusters):
    segmented_image[labels == i] = centers[i, 2:]
print(f"Segmented image shape: {segmented_image.shape}")

# write the segmented image
segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('res/segmented_image.jpg', segmented_image)
