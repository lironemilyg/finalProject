plt.imshow(np.reshape(batch[0],[100,100]))
rec_imgs = sess.run(self.generated_images,
                                 feed_dict={self.images: batch})


i=1
def f(i):
    rec_imgs = sess.run(self.generated_images, feed_dict={self.images: batch})
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(batch[i],[64,64]))
    plt.subplot(1,2,2)
    plt.imshow(rec_imgs[i,:,:,0])


    def save_diff(self, i, sess, batch):
        rec_imgs = sess.run(self.generated_images, feed_dict={self.images: batch})
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(np.reshape(batch[i], [64, 64]))
        plt.subplot(1, 2, 2)
        plt.plot(rec_imgs[i, :, :, 0])