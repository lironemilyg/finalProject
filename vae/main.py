import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *
import logging


class LatentAttention():
    def __init__(self):
        self.original_dataset_path = r'./dataset'
        self.train_dataset_path = r'./trainImages'
        self.test_dataset_path = r'./testImages'
        self.is_img_grey = True #grey or RGB
        self.img_size = 128
        self.num_of_steps = 150000
        self.factor = 1

        self.original_imgs = read_original_imgs(self.original_dataset_path, self.is_img_grey)
        self.train_imgs_cropped, self.train_labels = read_imgs_with_labels(self.train_dataset_path, self.is_img_grey)
        self.test_imgs_cropped, self.test_labels = read_imgs_with_labels(self.test_dataset_path, self.is_img_grey)
        self.num_of_original_imgs = self.original_imgs.__len__()

        self.n_z = 100

        if self.is_img_grey:
            self.images = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
        else:
            self.images = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3])

        self.tf_labels = tf.placeholder(tf.float32, [None])
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        z_mean = self.recognition(self.images)
        z_mean = tf.layers.batch_normalization(z_mean, training=self.is_training, trainable=False)

        self.classifier_estimated = tf.squeeze(self.classifier_net(z_mean))
        # self.classifier_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tf_labels, logits=self.classifier_estimated)
        # self.classifier_loss = tf.nn.weighted_cross_entropy_with_logits(targets=self.tf_labels, logits=self.classifier_estimated, pos_weight=5)
        self.classifier_loss = tf.nn.weighted_cross_entropy_with_logits(targets=self.tf_labels, logits=self.classifier_estimated, pos_weight=1.5)

        self.generated_images = self.generation(z_mean)
        self.generation_loss = tf.reduce_mean((self.images-self.generated_images)**2)

        # self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + self.generated_images) + (1-self.images) * tf.log(1e-8 + 1 - self.generated_images),1)

        t_vars = tf.trainable_variables()
        self.e_vars = [var for var in t_vars if 'recognition' in var.name]
        self.d_vars = [var for var in t_vars if 'generation' in var.name]
        self.c_vars = [var for var in t_vars if 'classifier_net' in var.name]

        self.Loss1 = tf.reduce_mean(self.generation_loss)
        self.Loss2 = tf.reduce_mean(self.classifier_loss)# + 0.1 * tf.nn.l2_loss(self.c_vars[0])
        # self.cost = tf.reduce_mean(self.generation_loss)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.Loss1,var_list=[self.e_vars+self.d_vars])
            self.optimizer2 = tf.train.AdamOptimizer(0.001).minimize(self.Loss2,var_list=[self.c_vars])

    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = conv2d(input_images, 1, self.factor*16, "d_h1")# 128x128x1 -> 64x64x16
            # h1 = batch_norm(h1, self.is_training)
            h1 = lrelu(h1)
            h2 = conv2d(h1, self.factor*16, self.factor*32, "d_h2")# 128x128x1 -> 64x64x16
            # h2 = batch_norm(h2, self.is_training)
            h2 = lrelu(h2)
            h3 = conv2d(h2, self.factor*32, self.factor*64, "d_h3")# 128x128x1 -> 64x64x16
            # h3 = batch_norm(h3, self.is_training)
            h3 = lrelu(h3)
            h4 = conv2d(h3, self.factor*64, self.factor*128, "d_h4")# 128x128x1 -> 64x64x16
            # h4 = batch_norm(h4, self.is_training)
            h4 = lrelu(h4)
            h5 = conv2d(h4, self.factor*128, self.factor*256, "d_h5")# 128x128x1 -> 64x64x16
            # h5 = batch_norm(h5, self.is_training)
            h5 = lrelu(h5)
            h5_flat = tf.reshape(h5, [-1, 4*4*self.factor*256])

            w_mean = dense(h5_flat, h5_flat.shape[1].value, self.n_z, "w_mean")

        return w_mean

    def classifier_net(self, z):
        with tf.variable_scope("classifier_net"):
            w_mean = dense(z, self.n_z, 1, "classifier")

        return w_mean

    # decoder
    def generation(self, z):
        batch_size = tf.shape(z)[0]
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 4*4*128*self.factor, scope='z_matrix')
            #z_develop = batch_norm(z_develop, self.is_training)
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [batch_size, 4, 4, 128 * self.factor]))
            h1 = conv_transpose(z_matrix, [batch_size, 8, 8, 64 * self.factor], "g_h1")
            # h1 = batch_norm(h1, self.is_training)
            h1 = tf.nn.relu(h1)
            h2 = conv_transpose(h1, [batch_size, 16, 16, 32 * self.factor], "g_h2")
            # h2 = batch_norm(h2, self.is_training)
            h2 = tf.nn.relu(h2)
            h3 = conv_transpose(h2, [batch_size, 32, 32, 16 * self.factor], "g_h3")
            # h3 = batch_norm(h3, self.is_training)
            h3 = tf.nn.relu(h3)
            h4 = conv_transpose(h3, [batch_size, 64, 64, 8 * self.factor], "g_h4")
            # h4 = batch_norm(h4, self.is_training)
            h4 = tf.nn.relu(h4)
            h5 = conv_transpose(h4, [batch_size, 128, 128, 1], "g_h5")
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
        # generation_test_batch, _ = get_test_batch(self.test_imgs_cropped, self.test_labels, self.img_size, self.batch_size)
        classifier_test_batch, classifier_test_labels_batch = get_test_batch(self.test_imgs_cropped, self.test_labels, self.img_size, 16)
        generation_test_batch = classifier_test_batch
        ims('./results/base.jpg', merge(generation_test_batch[:15], [4, 4]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # try:
            #     saver.restore(sess, tf.train.latest_checkpoint('training/'))
            #     self.num_of_steps = 5
            # except:
            #     pass
            generation_loss_list = []
            classifier_train_loss_list = []
            classifier_test_loss_list = []
            for step in range(self.num_of_steps):
                self.batch_size = 100
                random_batch = get_next_random_batch(self.original_imgs, self.img_size, self.batch_size)
                nonrandom_batch, nonrandom_labels = get_next_nonrandom_batch(self.train_imgs_cropped, self.train_labels, self.img_size, self.batch_size)
                _, session_generation_loss = sess.run((self.optimizer, self.generation_loss),
                                       feed_dict={self.images: random_batch,self.is_training:True})
                if (step > -1):
                    _, session_classifier_loss = sess.run((self.optimizer2, tf.nn.sigmoid(self.classifier_loss)),
                                                           feed_dict={self.images: nonrandom_batch,self.tf_labels:nonrandom_labels,self.is_training:True})
                # dumb hack to print cost every epoch
                if step % 100 == 0:
                    self.batch_size = 16
                    if(step > -1):
                        classification_test_labels = sess.run(tf.nn.sigmoid(self.classifier_estimated), feed_dict={self.images: classifier_test_batch, self.tf_labels: classifier_test_labels_batch, self.is_training: False})
                        print("step %d: genloss %f, classifier loss %f, test class loss %f" % (step, np.mean(np.abs(nonrandom_labels-session_generation_loss)), np.mean(np.abs(nonrandom_labels-session_classifier_loss)), np.mean(classification_test_labels)))
                    else:
                        print("step %d: genloss %f" % (step, np.mean(session_generation_loss)))
                    generation_test = sess.run(self.generated_images, feed_dict={self.images: generation_test_batch, self.is_training: False})
                    ims("results/" + str(step) + ".jpg", merge(generation_test[:15], [4, 4]))
                    generation_loss_list.append(np.mean(session_generation_loss))
                    if (step > -1):
                        classifier_train_loss_list.append(np.mean(session_classifier_loss))
                        classifier_test_loss_list.append(np.mean(classification_test_labels))
                        logging.info('step is {d}'.format(d=step))
                        logging.info('    Loss gen = {g}, class = {c}, test_class = {tc}'.format(g=session_generation_loss, c=session_classifier_loss,tc=classification_test_labels))

                        real_vs_estimated_labels = [(classifier_test_labels_batch[i], classification_test_labels[i], classifier_test_labels_batch[i] - classification_test_labels[i]) for i in range(self.batch_size)]
                        for tup in real_vs_estimated_labels:
                            logging.info('\t' + str(tup))
                        logging.info('##########################################################')
                if step % 5000 == 0:
                    saver.save(sess, os.getcwd() + "/training/train", global_step=step)
                if step == 149900:
                    self.save_diff(0, sess, random_batch)
                    self.save_diff(14, sess, random_batch)
            # logging.info('###############################################################################################')
            # for i in range(len(generation_loss_list)):
            #     logging.info('        i = {i} Loss gen = {g}, class = {c}, test_class = {tc}'.format(i=i, g=generation_loss_list[i],
            #                                                                                          c=classifier_train_loss_list[i],
            #                                                                                          tc=classifier_test_loss_list[i]))


try:
    os.remove(r'results/classifier_results.log')
except:
    pass
logging.basicConfig(filename=r'results/classifier_results.log', level=logging.DEBUG)
model = LatentAttention()
model.train()
