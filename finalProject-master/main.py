import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *

class LatentAttention():
    def __init__(self):
        self.path = r'/Users/lirongazit/Documents/finalProject/finalProject/finalProject-master/dataset'
        self.IsImgsGrey = True #grey or RGB
        self.ImgSize = 128
        self.num_of_steps = 150000
        self.factor = 1

        self.Imgs = ReadImgs(self.path,self.IsImgsGrey)
        self.n_samples = self.Imgs.__len__()

        self.n_z = 100
        self.batchsize = 50

        if self.IsImgsGrey:
            self.images = tf.placeholder(tf.float32, [self.batchsize, self.ImgSize, self.ImgSize, 1])
        else:
            self.images = tf.placeholder(tf.float32, [self.batchsize, self.ImgSize, self.ImgSize, 3])

        z_mean = self.recognition(self.images)

        self.generated_images = self.generation(z_mean)

        self.generation_loss = tf.reduce_sum((self.images-self.generated_images)**2)
        # self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + self.generated_images) + (1-self.images) * tf.log(1e-8 + 1 - self.generated_images),1)


        self.cost = tf.reduce_mean(self.generation_loss)
        # self.cost = tf.reduce_mean(self.generation_loss)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, self.factor*16, "d_h1"))# 128x128x1 -> 64x64x16
            h2 = lrelu(conv2d(h1, self.factor*16, self.factor*32, "d_h2")) # 64x64x16 -> 32x32x32
            h3 = lrelu(conv2d(h2, self.factor*32, self.factor*64, "d_h3")) # 32x32x32 -> 16x16x64
            h4 = lrelu(conv2d(h3, self.factor*64, self.factor*128, "d_h4")) # 16x16x64 -> 8x8x128
            h5 = lrelu(conv2d(h4, self.factor*128, self.factor*256, "d_h5")) # 8x8x128 -> 4x4x256
            h5_flat = tf.reshape(h5, [self.batchsize, -1])

            w_mean = dense(h5_flat, h5_flat.shape[1].value, self.n_z, "w_mean")

        return w_mean

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 4*4*128*self.factor, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 4, 4, 128*self.factor]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 8, 8, 64*self.factor], "g_h1"))
            h2 = tf.nn.relu(conv_transpose(h1, [self.batchsize, 16, 16, 32*self.factor], "g_h2"))
            h3 = tf.nn.relu(conv_transpose(h2, [self.batchsize, 32, 32, 16*self.factor], "g_h3"))
            h4 = tf.nn.relu(conv_transpose(h3, [self.batchsize, 64, 64, 8*self.factor], "g_h4"))
            h5 = conv_transpose(h4, [self.batchsize, 128, 128, 1], "g_h5")
            h5 = tf.nn.sigmoid(h5)
        return h5

    def save_diff(self, i, sess, batch):
        rec_imgs = sess.run(self.generated_images, feed_dict={self.images: batch})
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(np.reshape(batch[i], [128, 128]))
        plt.subplot(1, 2, 2)
        plt.imshow(rec_imgs[i, :, :, 0])
        fig.savefig('149900_{i}_fig.jpg'.format(i=i))

    def train(self):
        visualization = NextBatch(self.Imgs,self.ImgSize, self.batchsize)
        ims('./results/base.jpg',merge(visualization[:49],[7,7]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # try:
            #     saver.restore(sess, tf.train.latest_checkpoint('training/'))
            #     self.num_of_steps = 5
            # except:
            #     pass
            for step in range(self.num_of_steps):
                batch = NextBatch(self.Imgs,self.ImgSize,self.batchsize)
                _, gen_loss = sess.run((self.optimizer, self.generation_loss),
                                                 feed_dict={self.images: batch})
                # dumb hack to print cost every epoch
                if step % 100 == 0:
                    print("step %d: genloss %f" % (step, np.mean(gen_loss)))
                    generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                    ims("results/" + str(step) + ".jpg", merge(generated_test[:49], [7, 7]))
                if step % 5000 == 0:
                    saver.save(sess, os.getcwd() + "/training/train", global_step=step)
                if step == 149900:
                    self.save_diff(0, sess, batch)
                    self.save_diff(14, sess, batch)





model = LatentAttention()
model.train()
