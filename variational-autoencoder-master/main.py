import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *

class LatentAttention():
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.n_hidden = 500
        self.n_z = 100
        self.batchsize = 100

        self.images = tf.placeholder(tf.float32, [None, 16384])
        image_matrix = tf.reshape(self.images,[-1, 128, 128, 1])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 128*128])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 128x128x1 -> 64x64x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 64x64x16 -> 32x32x32
            h3 = lrelu(conv2d(h2, 32, 64, "d_h3")) # 32x32x32 -> 16x16x64
            h3_flat = tf.reshape(h3, [self.batchsize, 16*16*64])

            w_mean = dense(h3_flat, 16*16*64, self.n_z, "w_mean")
            w_stddev = dense(h3_flat, 16*16*64, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 16*16*64, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 16, 16, 64]))
            h0 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 32, 32, 32], "g_h0"))
            h1 = tf.nn.relu(conv_transpose(h0, [self.batchsize, 64, 64, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batchsize, 128, 128, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        reshaped_vis = visualization.reshape(self.batchsize,128,128)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            num_of_steps = 3000
            # try:
            #     saver.restore(sess, tf.train.latest_checkpoint('training/'))
            #     num_of_steps = 5
            # except:
            #     pass
            for epoch in range(num_of_steps):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss),
                                                     feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                        if num_of_steps != 1:
                            saver.save(sess, os.getcwd() + "/training/train", global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize, 128, 128)
                        ims("results/" + str(epoch) + ".jpg", merge(generated_test[:64], [8, 8]))

model = LatentAttention()
model.train()
