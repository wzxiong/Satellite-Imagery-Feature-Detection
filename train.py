from __future__ import absolute_import
import keras
import numpy as np
import h5py
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os
import random
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
from collections import defaultdict
from keras.layers.normalization import BatchNormalization


num_class = 10
DF = pd.read_csv('data/train_wkt_v4.csv')
GradSize = pd.read_csv('data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
Sample_Sub = pd.read_csv(os.path.join( 'data/sample_submission.csv'))
Image_size = 160
smooth = 1e-12
inDir = 'data'

def get_unet2():
    inputs = Input((8, Image_size, Image_size))
    conv1 = BatchNormalization(mode=0, axis=1)(Convolution2D(32, 3, 3, activation='elu', border_mode='same')(inputs))
    conv1 = BatchNormalization(mode=0, axis=1)(Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = BatchNormalization(mode=0, axis=1)(Convolution2D(64, 3, 3, activation='elu', border_mode='same')(pool1))
    conv2 = BatchNormalization(mode=0, axis=1)(Convolution2D(64, 3, 3, activation='elu', border_mode='same')(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = BatchNormalization(mode=0, axis=1)(Convolution2D(128, 3, 3, activation='elu', border_mode='same')(pool2))
    conv3 = BatchNormalization(mode=0, axis=1)(Convolution2D(128, 3, 3, activation='elu', border_mode='same')(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = BatchNormalization(mode=0, axis=1)(Convolution2D(256, 3, 3, activation='elu', border_mode='same')(pool3))
    conv4 = BatchNormalization(mode=0, axis=1)(Convolution2D(256, 3, 3, activation='elu', border_mode='same')(conv4))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = BatchNormalization(mode=0, axis=1)(Convolution2D(512, 3, 3, activation='elu', border_mode='same')(pool4))
    conv5 = BatchNormalization(mode=0, axis=1)(Convolution2D(512, 3, 3, activation='elu', border_mode='same')(conv5))
    
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = BatchNormalization(mode=0, axis=1)(Convolution2D(256, 3, 3, activation='elu', border_mode='same')(up6))
    conv6 = BatchNormalization(mode=0, axis=1)(Convolution2D(256, 3, 3, activation='elu', border_mode='same')(conv6))
    
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = BatchNormalization(mode=0, axis=1)(Convolution2D(128, 3, 3, activation='elu', border_mode='same')(up7))
    conv7 = BatchNormalization(mode=0, axis=1)(Convolution2D(128, 3, 3, activation='elu', border_mode='same')(conv7))
    
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = BatchNormalization(mode=0, axis=1)(Convolution2D(64, 3, 3, activation='elu', border_mode='same')(up8))
    conv8 = BatchNormalization(mode=0, axis=1)(Convolution2D(64, 3, 3, activation='elu', border_mode='same')(conv8))
    
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = BatchNormalization(mode=0, axis=1)(Convolution2D(32, 3, 3, activation='elu', border_mode='same')(up9))
    conv9 = BatchNormalization(mode=0, axis=1)(Convolution2D(32, 3, 3, activation='elu', border_mode='same')(conv9))
    
    conv10 = Convolution2D(num_class, 1, 1, activation='sigmoid')(conv9)
    
    model = Model(input=inputs, output=conv10)
    return model

def jaccard_coef(y_true, y_pred):
    # calculate jaccard coefficient
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # calculate jaccard coefficient when round every value to 0 and 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def calc_jacc(model):
    img = np.load('data/x_tmp_%d.npy' % num_class)
    msk = np.load('data/y_tmp_%d.npy' % num_class)
    
    prd = model.predict(img, batch_size=4)
    print prd.shape, msk.shape
    avg, trs = [], []
    
    for i in range(num_class):
        t_msk = msk[:, i, :, :]
        t_prd = prd[:, i, :, :]
        t_msk = t_msk.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])
        
        m, b_tr = 0, 0
        for j in range(10):
            tr = j / 10.0
            pred_binary_mask = t_prd > tr
            
            jk = jaccard_similarity_score(t_msk, pred_binary_mask)
            if jk > m:
                m = jk
                b_tr = tr
        print i, m, b_tr
        avg.append(m)
        trs.append(b_tr)
    
    score = sum(avg) / 10.0
    return score, trs

def Mask2Poly(polygons, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def get_patches(img, msk, amt=1000, aug=True):
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

def mask2polygons(mask, epsilon=5, min_area=1.):
    # convert mask to polygons
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
                                                  ((mask == 1) * 255).astype(np.uint8),
                                                  cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
                                                  # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                           shell=cnt[:, 0, :],
                           holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                                  if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def get_scalers(im_size, x_max, y_min):
    h, w = im_size
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


def train_net2():
    def jaccard_coef_loss(y_true, y_pred):
        return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)
    def save_model(model):
        json_string = model.to_json()
        if not os.path.isdir('cache'):
            os.mkdir('cache')
        json_name = 'architecture_' + '.json'
        weight_name = 'model_weights_' + '.h5'
        open(os.path.join('cache', json_name), 'w').write(json_string)
        model.save_weights(os.path.join('cache', weight_name), overwrite=True)
    print "start train net"
    x_val, y_val = np.load('data/x_tmp_%d.npy' % num_class), np.load('data/y_tmp_%d.npy' % num_class)
    img = np.load('data/x_trn_%d.npy' % num_class)
    msk = np.load('data/y_trn_%d.npy' % num_class)
    
    x_trn, y_trn = get_patches(img, msk)
    model = get_unet2()
    from keras.backend import binary_crossentropy
    from keras.optimizers import Nadam
    from keras.callbacks import History
    history = History()
    callbacks = [
                 history,
                 ]
    model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
    model.fit(x_trn, y_trn, batch_size=64, nb_epoch=50, verbose=1, shuffle=True,
                           callbacks=callbacks, validation_data=(x_val, y_val))
    save_model(model)
    return model

def predict_id(id, model, trs):
    img = M(id)
    x = stretch_n(img)
    
    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((num_class, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x
    
    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * Image_size:(i + 1) * Image_size, j * Image_size:(j + 1) * Image_size])
        
        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * Image_size:(i + 1) * Image_size, j * Image_size:(j + 1) * Image_size] = tmp[j]

    for i in range(num_class):
        prd[i] = prd[i] > trs[i]
    
    return prd[:, :img.shape[0], :img.shape[1]]


def predict_test(model, trs):
    print "predict test"
    for i, id in enumerate(sorted(set(Sample_Sub['ImageId'].tolist()))):
        msk = predict_id(id, model, trs)
        np.save('msk/10_%s' % id, msk)
        if i % 100 == 0: print i, id


def make_submit():
    print "make submission file"
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    print df.head()
    for idx, row in df.iterrows():
        id = row[0]
        kls = row[1] - 1
        
        msk = np.load('msk/10_%s.npy' % id)[kls]
        pred_polygons = mask2polygons(msk)
        x_max = GradSize.loc[GradSize['ImageId'] == id, 'Xmax'].as_matrix()[0]
        y_min = GradSize.loc[GradSize['ImageId'] == id, 'Ymin'].as_matrix()[0]
        
        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)
        
        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))
            
        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
        if idx % 100 == 0: print idx
    print df.head()
    df.to_csv('subm/submission.csv', index=False)

train_net2()

