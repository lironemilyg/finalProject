import numpy as np
import os, random
from PIL import Image
import csv
import glob
import math
import tensorflow as tf
from scipy.misc import imrotate


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w] = ((image[:,:,0])*255)

    return img


def read_original_imgs(path):
    files = glob.glob(os.path.join(path,'*.bmp'))
    imgs = [np.asarray(Image.open(file))[:,:,:1]/255. for file in files]
    return imgs

def read_imgs_with_labels(path):
    files = glob.glob(os.path.join(path,'*B-mode.bmp'))
    imgs = [np.asarray(Image.open(file))[:,:,:1]/255. for file in files]
    Labels = [int(os.path.split(file)[1].split('.')[1].split('_')[0]) for file in files]
    return imgs, Labels, files


def split_train_test(imgs,labels,img_files, test_imgs_per_class = 5):
    np_labels = np.array(labels)
    t = np.random.permutation(np.where(np_labels==0)[0])
    test_index = t[:test_imgs_per_class]
    train_index = t[test_imgs_per_class:]
    t = np.random.permutation(np.where(np_labels==1)[0])
    test_index = np.concatenate([test_index,t[:test_imgs_per_class]])
    train_index = np.concatenate([train_index,t[test_imgs_per_class:]])
    train_imgs = [imgs[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    train_imgs_files = [img_files[i] for i in train_index]
    test_imgs = [imgs[i] for i in test_index]
    test_labels = [labels[i] for i in test_index]
    test_imgs_files = [img_files[i] for i in test_index]
    return train_imgs, train_labels, test_imgs, test_labels, test_imgs_files, train_imgs_files

def get_next_random_batch(imgs, img_size, batch_size):
    num_of_imgs = imgs.__len__()
    indexes = np.random.permutation(np.arange(0,num_of_imgs))[:batch_size]
    #imrotate()
    batch = []
    for idx in indexes:
        while True:
            try:
                img = imgs[idx]
                h = img.shape[0] - img_size
                w = img.shape[1] - img_size
                if h > 0 and w > 0:
                    p1 = random.randint(0, h)
                    p2 = random.randint(0, w)
                    batch.append(img[p1:p1+img_size, p2:p2+img_size, :])
                    break
                else:
                    idx = np.random.randint(0, num_of_imgs)
            except:
                idx = np.random.randint(0, num_of_imgs)

    return np.stack(batch, 0)

def get_next_random_batch_with_labels(imgs, labels, img_size, batch_size, image_pixel_data, image_files):
    num_of_imgs = imgs.__len__()
    indexes = np.random.permutation(np.arange(0,num_of_imgs))[:batch_size]
    #imrotate()
    #delimiter = "\\" #windows
    delimiter = 't/' #linux
    batch = []
    label = []
    for idx in indexes:
        while True:
            try:
                img = imgs[idx]
                h = img.shape[0] - img_size
                w = img.shape[1] - img_size
                if h > 0 and w > 0:

                    image_name = image_files[idx].split(delimiter)[1]

                    x = image_pixel_data[image_name][0]
                    if(float(x) - img_size/2 < 0):
                        p1 = 0
                    elif (float(x) + img_size/2 > img.shape[1]):
                        p1 = w
                    else:
                        p1 = float(x) - img_size / 2
                    y = image_pixel_data[image_name][1]
                    if (float(y) - img_size / 2 < 0):
                        p2 = 0
                    elif (float(y) + img_size / 2 > img.shape[0]):
                        p2 = h
                    else:
                        p2 = float(y) - img_size / 2
                    p1 = int(p1)
                    p2 = int(p2)
                    p1, p2 = p2, p1
                    batch.append(img[p1:p1+img_size, p2:p2+img_size, :])

                    label.append(labels[idx])
                    break
                else:
                    idx = np.random.randint(0, num_of_imgs)
            except:
                idx = np.random.randint(0, num_of_imgs)

    return np.stack(batch, 0),np.stack(label)

def get_next_nonrandom_batch(imgs, labels, img_size, batch_size):
    num_of_imgs = imgs.__len__()
    indexes = np.random.permutation(np.arange(0,num_of_imgs))[:batch_size]
    batch_lable = []
    batch = []
    for idx in indexes:
        batch.append(imgs[idx])
        batch_lable.append(labels[idx])
    return np.stack(batch, 0), np.array(batch_lable)


def get_test_batch(imgs, labels, img_size, batch_size):
    num_of_imgs = imgs.__len__()
    batch_lable = []
    batch = []
    for idx in range(num_of_imgs):
        batch.append(imgs[idx])
        batch_lable.append(labels[idx])
    return np.stack(batch, 0)/255., np.array(batch_lable)




def shift_batch(data, offset, constant=0):
    """
    Shifts the array in two dimensions while setting rolled values to constant
    :param data: 3d (batch,x,y)
    :param offset: 2d (batch,xy)
    :param constant: The constant to replace rolled values with
    :return: The shifted array with "constant" where roll occurs
    """
    for i in range(data.shape[0]):
        tImg = data[i,:,:,:]
        tImg = np.roll(tImg, offset[i,0], axis=1)
        if offset[i,0] < 0:
            tImg[:, offset[i,0]:,:] = constant
        elif offset[i,0] > 0:
            tImg[:, 0:np.abs(offset[i,0]),:] = constant

            tImg = np.roll(tImg, offset[i,1], axis=0)
        if offset[i,1] < 0:
            tImg[offset[i,1]:, :] = constant
        elif offset[i,1] > 0:
            tImg[0:np.abs(offset[i,1]), :] = constant
        data[i, :, :, :] = tImg
    return data


def load_data_from_csv(path):
    dict = {}
    with open(path, newline = '') as data_file:
        data = csv.DictReader(data_file)
        for row in data:
            dict[row['filename']] = [row['x'], row['y'], row['height'], row['width']]
    return dict

def CreateCenerMassFile(path):
    import matplotlib.pyplot as plt
    files = os.listdir(path)
    files = sorted([file for file in files if 'B-mode' in file])
    plt.ion()
    plt.figure()
    with open(os.path.join(path,'Benchmark2.csv'), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        wr.writerow(["filename", "x", "y", "height", "width"])
        for i in range(files.__len__()):
            f = files[i]
            mylist = [f]
            img = Image.open(os.path.join(path, f))
            plt.imshow(img)
            plt.title(f)
            center = plt.ginput(1)
            width = plt.ginput(2)
            width = abs(width[1][0] - width[0][0])
            print(width)
            height = plt.ginput(2)
            height = abs(height[1][1] - height[0][1])
            print(height)

            #mylist = mylist+list([center[0],height,width])
            mylist.extend(list(center[0]) + [height] + [width])
            wr.writerow(mylist)


#CreateCenerMassFile(r'./dataset')


