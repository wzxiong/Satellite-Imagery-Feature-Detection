# Satellite-Imagery-Feature-Detection

This is a kaggle competition, which need to classify the objects in satellite image into 10 different group, which need people apply novel techniques to "train an eye in the sky",
the description and data could be get from [official website.](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection)

if you want to see the final report you can click [here](https://github.com/wzxiong/Satellite-Imagery-Feature-Detection/blob/master/satellite.pdf)

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

## Run the code
If you just want to see the whole process, you can look at the project.ipynb, which including all the code.

If you want to creat the submission file, you can download project.ipynb and unet.h5py and original dataset from website, then create dir called msk, data, subm and weights. Put unet.h5py(at release) into weights dir, and all the dataset go to the data dir. Lastly, run the code in jupyter and get solution.

If you want to train by yourself you can also use the jupyter notebook file and run train2() at the last session.

There are also .py files which have same content, preprocessing.py contain the code to process the original data, but you need to do the same things as using jupyter notebook file. train_output.py is used to provide solution. train.py contain the train process code, and water.py used to provide water submission.  Other files are supporting file or used to create some plot in the report.
