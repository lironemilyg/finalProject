import tensorflow as tf
from ops import *


factor = 1
# encoder
def recognition(input_images):
    with tf.variable_scope("recognition"):
        h1 = conv2d(input_images, 1, factor * 16, ker_size=5,stride=2, name="d_h1")  # 128x128x1 -> 64x64x16
        # h1 = batch_norm(h1, self.is_training)
        h1 = lrelu(h1)
        h2 = conv2d(h1, factor * 16, factor * 32, ker_size=5,stride=2, name="d_h2")  # 64x64x16 -> 32x32x32
        # h2 = batch_norm(h2, self.is_training)
        h2 = lrelu(h2)
        h3 = conv2d(h2, factor * 32, factor * 64, ker_size=5,stride=2, name="d_h3")  # 32x32x32 -> 16x16x64
        h3 = lrelu(h3)
        h4 = conv2d(h3, factor * 64, 1, ker_size=1,stride=1, name="d_h4")  # 16x16x64 -> 8x8x128
        # h3 = batch_norm(h3, self.is_training)
        # h4_flat = tf.reshape(h4, [-1, 16 * 16 * factor * 64])

    return h4


def classifier_net(z):
    with tf.variable_scope("classifier_net"):
        z = tf.reshape(z, [-1, 16 * 16])
        w_mean = dense(z, 16*16, 10*10, "classifier")
        w_mean = tf.nn.relu(w_mean)
        w_mean_2 = dense(w_mean, 10*10, 4*4, "classifier_2")
        w_mean_2 = tf.nn.relu(w_mean_2)
        w_mean_3 = dense(w_mean_2, 4 * 4, 1, "classifier_3")
        #w_mean_3 = tf.nn.relu(w_mean_3)
        #w_mean_4 = dense(w_mean_3, 4 * 4, 2 * 2, "classifier_4")
        #w_mean_4 = tf.nn.relu(w_mean_4)
        #w_mean_5 = dense(w_mean_4, 2 * 2, 1, "classifier_5")
    return w_mean_3


# decoder
def generation(z):
    batch_size = tf.shape(z)[0]
    with tf.variable_scope("generation"):
        # h3 = batch_norm(h3, self.is_training)
        h1 = conv_transpose(z, [batch_size, 32, 32, 8 * factor], "g_h1")
        # h4 = batch_norm(h4, self.is_training)
        h1 = tf.nn.relu(h1)
        h2 = conv_transpose(h1, [batch_size, 64, 64, 16 * factor], "g_h2")
        h2 = tf.nn.relu(h2)
        h3 = conv_transpose(h2, [batch_size, 128, 128, 1], "g_h3")
        h3 = tf.nn.sigmoid(h3)
    return h3
