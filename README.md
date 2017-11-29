# Satellite-Imagery-Feature-Detection

This is a kaggle competition, which need to classify the objects in satellite image into 10 different group, which need people apply novel techniques to "train an eye in the sky",
the description and data could be get from [official website.](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection)

## Software and important package:
- Python 2.7 or Python 3.5(I use 3.5)
- Keras 2
- Theano 0.9.0

## Useful packages:
- h5py
- matplotlib
- numba
- numpy
- pandas
- rasterio
- Shapely
- scikit_image
- tifffile
- OpenCV 
- tqdm

##Run the code
If you just want to see the whole process, you can look at the project.ipynb, which including all the code. If you want to creat the 
submission file, you can download project.ipynb and unet.h5py and original dataset from website, then create dir called msk, data, subm and weights.
Put unet.h5py into weights, and all the dataset go to the data dir. Lastly, run the code in jupyter and get solution.
