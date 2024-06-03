# image-segmentation-clustering
cluster image pixels using k-means clustering based on rgb color and xy coordinate and segment the image using the clusters

just a simple experiment, does not work particularly well

## source images
the source images are taken from unsplash and can be seen below:

![1](img/nattu-adnan-vvHRdOwqHcg-unsplash.jpg)
![2](img/andrzej-suwara-j4glkaOX-ds-unsplash.jpg)
![3](img/chris-brignola-n7n-nkadHRM-unsplash.jpg)
![4](img/marius-spita-03sd18kcs8s-unsplash.jpg)
![5](img/rana-sawalha-IhuHLIxS_Tk-unsplash.jpg)

## results
for some images with stark differences in color the segmentation works ok, but mostly it falls short in a number of ways

the following grids show segmentation results at various numbers of clusters (increasing from top to bottom) and at various balances between the importance of color and xy coordinate (left: mostly color, right: mostly xy coordinate)

![1](res/nattu-adnan-vvHRdOwqHcg-unsplash_grid.jpg)
![2](res/andrzej-suwara-j4glkaOX-ds-unsplash_grid.jpg)
![3](res/chris-brignola-n7n-nkadHRM-unsplash_grid.jpg)
![4](res/marius-spita-03sd18kcs8s-unsplash_grid.jpg)
![5](res/rana-sawalha-IhuHLIxS_Tk-unsplash_grid.jpg)
