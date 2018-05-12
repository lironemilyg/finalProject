import numpy as np
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


def read_original_imgs(path, is_grey):
    files = os.listdir(path)
    if is_grey:
        imgs = [np.asarray(Image.open(os.path.join(path, file)))[:,:,:1] for file in files]
    else:
        imgs = [np.asarray(Image.open(os.path.join(path, file))) for file in files]
    return imgs


def read_imgs_with_labels(path, is_grey):
    files = os.listdir(path)
    if is_grey:
        imgs = [np.asarray(Image.open(os.path.join(path, file)))[:,:,:1] for file in files if file != ".DS_Store"]
    else:
        imgs = [np.asarray(Image.open(os.path.join(path, file))) for file in files if file != ".DS_Store"]
    Labels = [int(file.split('.')[1].split('_')[0]) for file in files if file != ".DS_Store"]
    return imgs, Labels


def get_next_random_batch(imgs, img_size, batch_size):
    num_of_imgs = imgs.__len__()
    indexes = np.random.permutation(np.arange(0,num_of_imgs))[:batch_size]
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

    return np.stack(batch, 0)/255.


def get_next_nonrandom_batch(imgs, labels, img_size, batch_size):
    num_of_imgs = imgs.__len__()
    indexes = np.random.permutation(np.arange(0,num_of_imgs))[:batch_size]
    batch_lable = []
    batch = []
    for idx in indexes:
        batch.append(imgs[idx])
        batch_lable.append(labels[idx])
    return np.stack(batch, 0)/255., np.array(batch_lable)


def get_test_batch(imgs, labels, img_size, batch_size):
    num_of_imgs = imgs.__len__()
    batch_lable = []
    batch = []
    for idx in range(num_of_imgs):
        batch.append(imgs[idx])
        batch_lable.append(labels[idx])
    return np.stack(batch, 0)/255., np.array(batch_lable)
