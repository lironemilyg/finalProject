import numpy as np
import os, random
from PIL import Image
import csv
import glob
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
    return imgs, Labels

def split_train_test(imgs,labels,test_imgs_per_class = 5):
    np_labels = np.array(labels)
    t = np.random.permutation(np.where(np_labels==0)[0])
    test_index = t[:test_imgs_per_class]
    train_index = t[test_imgs_per_class:]
    t = np.random.permutation(np.where(np_labels==1)[0])
    test_index = np.concatenate([test_index,t[:test_imgs_per_class]])
    train_index = np.concatenate([train_index,t[test_imgs_per_class:]])
    train_imgs = [imgs[i] for i in train_index]
    trian_labels = [labels[i] for i in train_index]
    test_imgs = [imgs[i] for i in test_index]
    test_labels = [labels[i] for i in test_index]
    return train_imgs, trian_labels, test_imgs, test_labels

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

def get_next_random_batch1(imgs,labels, img_size, batch_size):
    num_of_imgs = imgs.__len__()
    indexes = np.random.permutation(np.arange(0,num_of_imgs))[:batch_size]
    #imrotate()
    batch = []
    label = []
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

def CreateCenerMassFile(path):
    import matplotlib.pyplot as plt
    files = os.listdir(path)
    files = [file for file in files if 'B-mode' in file]
    plt.ion()
    plt.figure()
    with open(os.path.join(path,'Benchmark1.csv'), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        for i in range(58,files.__len__()):
            f = files[i]
            mylist = [f]
            img = Image.open(os.path.join(path, f))
            plt.imshow(img)
            plt.title(f)
            a = plt.ginput(1)
            mylist = mylist+list(a[0])
            wr.writerow(mylist)

