from sklearn.cluster import KMeans
import numpy as np
import cv2
import os
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# param
balances = [1, 10, 100, 300]
n_cluster_max = 9

# logging
logging.basicConfig(level=logging.INFO)

# load image
img_name = "nattu-adnan-vvHRdOwqHcg-unsplash"
image = cv2.imread(f"img/{img_name}.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
logging.info(f"Image shape: {image.shape}")
x, y = image.shape[:2]
ratio = x / y

# make result directory
if not os.path.exists(f"res/{img_name}"):
    os.makedirs(f"res/{img_name}")

seg_images = []

for n_clusters in tqdm(range(2, n_cluster_max + 1)):
    for balance in balances:
        # generate linearly spaced values between 0 and balance for x and y
        x_coords = np.linspace(0, balance, x)
        y_coords = np.linspace(0, balance, y)

        # create a grid of coordinates
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

        # stack the grids to create the final array
        coordinates = np.stack((xx, yy), axis=-1)
        logging.debug(f"Coordinates shape: {coordinates.shape}")

        # combine image and coordinates
        data = np.concatenate((coordinates, image), axis=-1).reshape(-1, 5)
        logging.debug(f"Data shape: {data.shape}")

        # fit kmeans
        kmeans = KMeans(n_clusters=n_clusters).fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        logging.debug(f"Labels shape: {labels.shape}")

        # reverse the reshaping
        labels = labels.reshape(x, y)
        logging.debug(f"Labels shape after reshaping: {labels.shape}")

        # create the segmented image
        segmented_image = np.zeros_like(image, dtype=np.float32)
        for i in range(n_clusters):
            segmented_image[labels == i] = centers[i, 2:]
        logging.debug(f"Segmented image shape: {segmented_image.shape}")

        # append to list
        seg_images.append((segmented_image / 255, n_clusters, balance)) # normalize to [0, 1]

        # write the segmented image
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"res/{img_name}/segmented_n{n_clusters}_b{balance}.jpg", segmented_image)

logging.info("segmentation complete, plotting grid")

# plot image grid
fig = plt.figure(figsize=(len(balances)*4, (n_cluster_max - 1)*4*ratio), dpi=300)
grid = ImageGrid(fig, 111, nrows_ncols=(n_cluster_max - 1, len(balances)), axes_pad=(0.1, 0.4))

for ax, (im, n, b) in zip(grid, seg_images):
    ax.imshow(im)
    ax.axis('off')
    ax.set_title(f"n_clusters={n}, balance={b}")

# save image
plt.savefig(f"res/{img_name}_grid.jpg")
