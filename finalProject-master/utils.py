import numpy as np
import tensorflow as tf
import os, random
from PIL import Image

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w] = ((image[:,:,0])*255)

    return img


def ReadImgs(path,GreyFlag):
    files = os.listdir(path)
    if GreyFlag:
        Imgs = [np.asarray(Image.open(os.path.join(path, file)))[:,:,:1] for file in files]
    else:
        Imgs = [np.asarray(Image.open(os.path.join(path, file))) for file in files]
    return Imgs


def ReadImgs1(path,GreyFlag):
    files = os.listdir(path)
    if GreyFlag:
        Imgs = [np.asarray(Image.open(os.path.join(path, file)))[:,:,:1] for file in files if file != ".DS_Store"]
    else:
        Imgs = [np.asarray(Image.open(os.path.join(path, file))) for file in files if file != ".DS_Store"]
    Labels = [int(file.split('.')[1].split('_')[0]) for file in files if file != ".DS_Store"]
    return Imgs, Labels

def NextBatch(Imgs,ImgSize,batch_size):
    NumImgs = Imgs.__len__()
    idxs = np.random.permutation(np.arange(0,NumImgs))[:batch_size]
    batch = []
    for idx in idxs:
        while True:
            try:
                img = Imgs[idx]
                h = img.shape[0] - ImgSize
                w = img.shape[1] - ImgSize
                if h > 0 and w > 0:
                    p1 = random.randint(0, h)
                    p2 = random.randint(0, w)
                    batch.append(img[p1:p1+ImgSize, p2:p2+ImgSize, :])
                    break
                else:
                    idx = np.random.randint(0, NumImgs)
            except:
                idx = np.random.randint(0, NumImgs)

    return np.stack(batch, 0)/255.

def NextBatch1(Imgs,labels,ImgSize,batch_size):
    NumImgs = Imgs.__len__()
    idxs = np.random.permutation(np.arange(0,NumImgs))[:batch_size]
    batchLable = []
    batch = []
    for idx in idxs:
        batch.append(Imgs[idx])
        batchLable.append(labels[idx])
    return np.stack(batch, 0)/255., np.array(batchLable)

def testBatch(Imgs,labels,ImgSize,batch_size):
    NumImgs = Imgs.__len__()
    batchLable = []
    batch = []
    for idx in range(NumImgs):
        batch.append(Imgs[idx])
        batchLable.append(labels[idx])
    return np.stack(batch, 0)/255., np.array(batchLable)
