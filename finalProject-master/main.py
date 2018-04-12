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
        self.ImgSize = 64
        self.num_of_steps = 100000
        self.factor = 1

        self.Imgs = ReadImgs(self.path,self.IsImgsGrey)
        self.n_samples = self.Imgs.__len__()

        self.n_z = 100
        self.batchsize = 50

        if self.IsImgsGrey:
            self.images = tf.placeholder(tf.float32, [self.batchsize, self.ImgSize, self.ImgSize, 1])
        else:
            self.images = tf.placeholder(tf.float32, [self.batchsize, self.ImgSize, self.ImgSize, 3])

        z_mean, z_stddev = self.recognition(self.images)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        # guessed_z = z_mean + (z_stddev * samples)
        guessed_z = z_mean

        self.generated_images = self.generation(guessed_z)

        self.generation_loss = tf.reduce_sum((self.images-self.generated_images)**2)
        # self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + self.generated_images) + (1-self.images) * tf.log(1e-8 + 1 - self.generated_images),1)

        self.latent_loss = 0 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        # self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        # self.cost = tf.reduce_mean(self.generation_loss)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, self.factor*16, "d_h1")) # 64x64x1 -> 32x32x16
            h2 = lrelu(conv2d(h1, self.factor*16, self.factor*32, "d_h2")) # 32x32x16 -> 16x16x32
            h3 = lrelu(conv2d(h2, self.factor*32, self.factor*64, "d_h3")) # 16x16x32 -> 8x8x64
            h4 = lrelu(conv2d(h3, self.factor*64, self.factor*128, "d_h4")) # 8x8x64 -> 4x4x128
            h4_flat = tf.reshape(h4, [self.batchsize, -1])

            w_mean = dense(h4_flat, h4_flat.shape[1].value, self.n_z, "w_mean")
            w_stddev = dense(h4_flat, h4_flat.shape[1].value, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 4*4*128*self.factor, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 4, 4, 128*self.factor]))
            h0 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 8, 8, 64*self.factor], "g_h0"))
            h1 = tf.nn.relu(conv_transpose(h0, [self.batchsize, 16, 16, 32*self.factor], "g_h1"))
            h2 = tf.nn.relu(conv_transpose(h1, [self.batchsize, 32, 32, 16*self.factor], "g_h2"))
            h3 = conv_transpose(h2, [self.batchsize, 64, 64, 1], "g_h3")
            h3 = tf.nn.sigmoid(h3)
        return h3

    def train(self):
        visualization = NextBatch(self.Imgs,self.ImgSize,self.batchsize)
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
                _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss),
                                                 feed_dict={self.images: batch})
                # dumb hack to print cost every epoch
                if step % 100 == 0:
                    print("step %d: genloss %f latloss %f" % (step, np.mean(gen_loss), np.mean(lat_loss)))
                    generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                    ims("results/" + str(step) + ".jpg", merge(generated_test[:49], [7, 7]))
                if step % 5000 == 0:
                    saver.save(sess, os.getcwd() + "/training/train", global_step=step)

model = LatentAttention()
model.train()
