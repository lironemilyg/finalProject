import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *
import logging
from model import *


class LatentAttention():
    def __init__(self):
        self.original_dataset_path = r'./dataset'
        self.benchmark_path = r'./Benchmark2.csv'
        self.img_size = 128
        self.num_of_steps = 15000
        self.factor = 1

        #CreateCenerMassFile(self.original_dataset_path)

        self.unsupervised_imgs = read_original_imgs(self.original_dataset_path)
        self.supervised_imgs, self.supervised_imgs_labels, self.image_files = read_imgs_with_labels(self.original_dataset_path)
        self.train_imgs, self.trian_labels, self.test_imgs, self.test_labels, self.test_img_files, self.train_img_files = split_train_test(self.supervised_imgs, self.supervised_imgs_labels,  self.image_files, test_imgs_per_class=5)
        self.image_pixel_data = load_data_from_csv(self.benchmark_path)
        self.images = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])


        self.tf_labels = tf.placeholder(tf.float32, [None])
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        z_mean = recognition(self.images)
        z_mean = tf.layers.batch_normalization(z_mean, training=self.is_training, trainable=False)

        self.classifier_estimated = tf.squeeze(classifier_net(z_mean))
        class_w = (np.array(self.trian_labels)==0).sum()/(np.array(self.trian_labels)==1).sum()
        self.classifier_loss = tf.nn.weighted_cross_entropy_with_logits(targets=self.tf_labels, logits=self.classifier_estimated, pos_weight=class_w)

        self.generated_images = generation(z_mean)
        self.generation_loss = tf.reduce_mean((self.images - self.generated_images) ** 2)


        t_vars = tf.trainable_variables()
        self.e_vars = [var for var in t_vars if 'recognition' in var.name]
        self.d_vars = [var for var in t_vars if 'generation' in var.name]
        self.c_vars = [var for var in t_vars if 'classifier_net' in var.name]

        self.Loss1 = tf.reduce_mean(self.generation_loss)
        self.Loss2 = tf.reduce_mean(self.classifier_loss) + 0.007 * tf.nn.l2_loss(self.c_vars[0])

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.Loss1,var_list=[self.e_vars+self.d_vars])
            self.optimizer2 = tf.train.GradientDescentOptimizer(0.1).minimize(self.Loss2,var_list=[self.c_vars])



    def train(self):
        #visu_imgs_AE = get_next_random_batch(self.unsupervised_imgs, self.img_size, 64)
        #ims('./results/base.jpg', merge(visu_imgs_AE, [8, 8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        self.batch_size = 100
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            generation_loss_list = []
            classifier_train_loss_list = []
            classifier_test_loss_list = []
            # train autoencoder
            '''
            for step in range(int(self.num_of_steps/25)):
                random_batch = get_next_random_batch(self.unsupervised_imgs, self.img_size, self.batch_size)
                _, session_generation_loss = sess.run((self.optimizer, self.generation_loss),
                                       feed_dict={self.images: random_batch,self.is_training:True})
                print(session_generation_loss)

                if step % 100 == 0:
                    # self.batch_size = 16
                    # if(step > 40000):
                    #     classification_test_labels = sess.run(tf.nn.sigmoid(self.classifier_estimated), feed_dict={self.images: classifier_test_batch, self.tf_labels: classifier_test_labels_batch, self.is_training: False})
                    #     print("step %d: genloss %f, classifier loss %f, test class loss %f" % (step, np.mean(np.abs(nonrandom_labels-session_generation_loss)), np.mean(np.abs(nonrandom_labels-session_classifier_loss)), np.mean(classification_test_labels)))
                    # else:
                    #     print("step %d: genloss %f" % (step, np.mean(session_generation_loss)))
                    generation_test = sess.run(self.generated_images, feed_dict={self.images: visu_imgs_AE, self.is_training: True})
                    ims("results/" + str(step) + ".jpg", merge(generation_test, [8, 8]))
            '''
            # train classifier
            self.batch_size = 10
            test_batch, test_labels = get_next_random_batch_with_labels(self.test_imgs, self.test_labels, self.img_size,
                                                              self.batch_size, self.image_pixel_data,
                                                              self.test_img_files)
            ims('./results/base.jpg', merge(test_batch, [5, 2]))
            for step in range(int(self.num_of_steps)):
                self.batch_size = 100
                random_batch = get_next_random_batch(self.unsupervised_imgs, self.img_size, self.batch_size)
                _, session_generation_loss = sess.run((self.optimizer, self.generation_loss),
                                                      feed_dict={self.images: random_batch, self.is_training: True})
                print(session_generation_loss)
                if (step > 6000):
                    batch, labels = get_next_random_batch_with_labels(self.train_imgs, self.trian_labels, self.img_size,
                                                                      self.batch_size, self.image_pixel_data, self.train_img_files)
                    _, session_classifier_loss,train_label = sess.run((self.optimizer2, self.Loss2,tf.nn.sigmoid(self.classifier_estimated)),
                                                                      feed_dict={self.images: batch,self.tf_labels:labels,self.is_training:True})
                    print("train classifier loss " + str(session_classifier_loss))

                #_, session_classifier_loss = sess.run((self.optimizer2, self.Loss2),
                #                                       feed_dict={self.images: batch,self.tf_labels:labels,self.is_training:False})

                if step % 100 == 0:
                    if(step > 6000):
                        logging.info('##########################################################')
                        logging.info('step is {d}'.format(d=step))
                        logging.info('########################TRAIN###########################')
                        real_vs_estimated_labels = [
                            (labels[i], train_label[i], int(round(abs(labels[i] - train_label[i]))))
                            for i in range(self.batch_size)]
                        for tup in real_vs_estimated_labels:
                            logging.info('\t' + str(tup))
                        logging.info('##########################################################')
                    self.batch_size = 10
                    if(step>4000):
                        session_classifier_loss, test_label_result = sess.run((self.Loss2, tf.nn.sigmoid(self.classifier_estimated)),
                                                                           feed_dict={self.images: test_batch, self.tf_labels: test_labels, self.is_training: False})
                        logging.info('########################TEST###########################')
                        real_vs_estimated_labels = [(test_labels[i], test_label_result[i],
                                                     int(round(abs(test_labels[i] - test_label_result[i])))) for i in
                                                    range(self.batch_size)]
                        for tup in real_vs_estimated_labels:
                            logging.info('\t' + str(tup))
                        logging.info('##########################################################')
                        logging.info("test classifier loss " + str(session_classifier_loss))
                        logging.info('##########################################################')
                        print("test classifier loss " + str(session_classifier_loss))

                    generation_test = sess.run(self.generated_images, feed_dict={self.images: test_batch, self.is_training: False})
                    ims("results/" + str(step) + ".jpg", merge(generation_test, [5, 2]))


            # dumb hack to print cost every epoch

                #     generation_loss_list.append(np.mean(session_generation_loss))
                #     if (step > 40000):
                #         classifier_train_loss_list.append(np.mean(session_classifier_loss))
                #         classifier_test_loss_list.append(np.mean(classification_test_labels))
                #         logging.info('step is {d}'.format(d=step))
                #         logging.info('    Loss gen = {g}, class = {c}, test_class = {tc}'.format(g=session_generation_loss, c=session_classifier_loss,tc=classification_test_labels))
                #
                #         real_vs_estimated_labels = [(classifier_test_labels_batch[i], classification_test_labels[i], classifier_test_labels_batch[i] - classification_test_labels[i]) for i in range(self.batch_size)]
                #         for tup in real_vs_estimated_labels:
                #             logging.info('\t' + str(tup))
                #         logging.info('##########################################################')
                # if step % 5000 == 0:
                #     saver.save(sess, os.getcwd() + "/training/train", global_step=step)


try:
    os.remove(r'results/classifier_results.log')
except:
    pass
logging.basicConfig(filename=r'results/classifier_results.log', level=logging.DEBUG)
model = LatentAttention()
model.train()
