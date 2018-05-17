
for step in range(100):
    nonrandom_batch, nonrandom_labels = get_next_nonrandom_batch(self.train_imgs_cropped, self.train_labels, self.img_size, self.batch_size)
    _, session_classifier_loss = sess.run((self.optimizer2, tf.nn.sigmoid(self.classifier_loss)),
                                          feed_dict={self.images: nonrandom_batch,self.tf_labels:nonrandom_labels,self.is_training:False})
    print("step %d: classifier loss %f" % (step, np.mean(np.abs(nonrandom_labels-session_classifier_loss))))
