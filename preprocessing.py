from __future__ import absolute_import, division
import numpy as np
import h5py
import cv2
import pandas as pd
from tqdm import tqdm
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os
import random
from sklearn.metrics import jaccard_similarity_score
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
from collections import defaultdict

num_class = 10
DF = pd.read_csv('data/train_wkt_v4.csv')
GradSize = pd.read_csv('data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
Sample_Sub = pd.read_csv(os.path.join( 'data/sample_submission.csv'))
Image_size = 160
smooth = 1e-12
inDir = 'data'

#In this dataset that we provide, we create a set of geo-coordinates that are in the range of x = [0,1] and y = [-1,0].
#These coordinates are transformed such that we obscure the location of where the satellite images are taken from.
#The images are from the same region on Earth.
def coordinates2raster(coords, img_size, xymax):
    #For each image, you should be able to get the width (W) and height (H) from the image raster. For a 3-band image
    #that is 3391 x 3349 x 3, W is 3349, and H is 3391. Then you can scale your data as follows:
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int

def Get_XmaxXmin(grid_sizes_panda, imageId):
    # To utilize these images, we provide the grid coordinates of each image so you know how to scale them and align
    #them with the images in pixels. In grid_sizes.csv, you are given the Xmax and Ymin values for each imageId.
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def Get_Poly_List(wkt_list_pandas, imageId, cType):
    # extract polygon list from file
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList

def Convert_Contour(polygonList, raster_img_size, xymax):
    # combined function: extract polygon list, then convert to raster, and get parameter list,interior list
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = coordinates2raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = coordinates2raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def Plot_Mask(raster_img_size, contours, class_value=1):
    # plot the mask fill in the polygon, perim is 1, interior is 0, when given contours
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GradSize, wkt_list_pandas=DF):
    #generate mask for certain image and class
    xymax = Get_XmaxXmin(grid_sizes_panda, imageId)
    polygon_list = Get_Poly_List(wkt_list_pandas, imageId, class_type)
    contours = Convert_Contour(polygon_list, raster_size, xymax)
    mask = Plot_Mask(raster_size, contours, 1)
    return mask


def M(image_id):
    # read in certain image
    filename = os.path.join(inDir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img


def stretch_n(bands, lower_percent=0, higher_percent=100):
    #Contrast enhancement
    out = np.zeros_like(bands).astype(np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    
    return out.astype(np.float32)

def stick_all_train():
    print "let's stick all imGradSize together"
    s = 835
    
    x = np.zeros((5 * s, 5 * s, 8))
    y = np.zeros((5 * s, 5 * s, num_class))
    
    ids = sorted(DF.ImageId.unique())
    print len(ids)
    for i in range(5):
        for j in range(5):
            id = ids[5 * i + j]
            
            img = M(id)
            img = stretch_n(img)
            print img.shape, id, np.amax(img), np.amin(img)
            x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
            for z in range(num_class):
                y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
                                                                                           (img.shape[0], img.shape[1]), id, z + 1)[:s, :s]
    print np.amax(y), np.amin(y)
    np.save('data/x_trn_%d' % num_class, x)
    np.save('data/y_trn_%d' % num_class, y)

def get_patches(img, msk, amt=10000, aug=True):
    is2 = int(1.0 * Image_size)
    xm, ym = img.shape[0] - is2, img.shape[1] - is2
    x, y = [], []
    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
    for i in range(amt):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)
        
        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]
        
        for j in range(num_class):
            sm = np.sum(ms[:, :, j])
            if 1.0 * sm / is2 ** 2 > tr[j]:
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                        ms = ms[::-1]
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]
                x.append(im)
                y.append(ms)
    x, y = 2 * np.transpose(x, (0, 3, 1, 2)) - 1, np.transpose(y, (0, 3, 1, 2))
    print x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y)
    return x, y

def make_val():
    print "let's pick some samples for validation"
    img = np.load('data/x_trn_%d.npy' % num_class)
    msk = np.load('data/y_trn_%d.npy' % num_class)
    x, y = get_patches(img, msk, amt=3000)
    
    np.save('data/x_tmp_%d' % num_class, x)
    np.save('data/y_tmp_%d' % num_class, y)

data_path='data'
three_band_path = os.path.join(data_path, 'three_band')

file_names = []
widths_3 = []
heights_3 = []


for file_name in tqdm(sorted(os.listdir(three_band_path))):
    # TODO: crashes if there anything except tiff files in folder (for ex, QGIS creates a lot of aux files)
    image_id = file_name.split('.')[0]
    image_3 = tiff.imread(os.path.join(three_band_path, file_name))
    
    file_names += [file_name]
    _, height_3, width_3 = image_3.shape
    
    widths_3 += [width_3]
    heights_3 += [height_3]

df = pd.DataFrame({'file_name': file_names, 'width': widths_3, 'height': heights_3})

df['image_id'] = df['file_name'].apply(lambda x: x.split('.')[0])

df.to_csv(os.path.join(data_path, '3_shapes.csv'), index=False)
stick_all_train()
make_val()
