import tensorflow as tf
import ccifar10

dtype = tf.float32 # tf.float32
height = 24
width = 24
batch_size = 128
NUM_CLASSES = 10
weight_decay = []

Data = ccifar10.Data()

input_images = tf.placeholder(dtype=dtype,shape=[batch_size, 32, 32, 3])
input_label = tf.placeholder(dtype='int32',shape=[batch_size])
# Image processing for training the network. Note the many random
# distortions applied to the image.

# # Randomly crop a [height, width] section of the image.
# distorted_image = tf.random_crop(images, [height, width, 3])
#
# # Randomly flip the image horizontally.
# distorted_image = tf.image.random_flip_left_right(distorted_image)
#
# # Because these operations are not commutative, consider randomizing
# # the order their operation.
# # NOTE: since per_image_standardization zeros the mean and makes
# # the stddev unit, this likely has no effect see tensorflow#1458.
# distorted_image = tf.image.random_brightness(distorted_image,
#                                              max_delta=63)
# distorted_image = tf.image.random_contrast(distorted_image,
#                                            lower=0.2, upper=1.8)
#
# # Subtract off the mean and divide by the variance of the pixels.
# float_image = tf.image.per_image_standardization(distorted_image)

with tf.variable_scope('conv1') as scope:
    kernel = tf.get_variable(name='weights',shape=[5, 5, 3, 64],dtype=dtype
                             ,initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=dtype)) #wd=0

    conv = tf.nn.conv2d(input_images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases',shape=[64],dtype=dtype,
                             initializer=tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

# pool1
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME', name='pool1')
# norm1
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                  name='norm1')

# conv2
with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable(name='weights',shape=[5, 5, 64, 64],dtype=dtype
                             ,initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=dtype)) #wd=0

    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases',shape=[64],dtype=dtype,
                             initializer=tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

# norm2
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                  name='norm2')
# pool2
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1], padding='SAME', name='pool2')

# local3
with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = tf.get_variable(name='weights',shape=[dim, 384],dtype=dtype
                             ,initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=dtype))
    wd  =0.004
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
    biases = tf.get_variable(name='biases', shape=[384], dtype=dtype,
                             initializer=tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

# local4
with tf.variable_scope('local4') as scope:
    weights = tf.get_variable(name='weights', shape=[384, 192], dtype=dtype
                              , initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=dtype))
    wd = 0.004
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
    biases = tf.get_variable(name='biases', shape=[192], dtype=dtype,
                             initializer=tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

# linear layer(WX + b),
# We don't apply softmax here because
# tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
# and performs the softmax internally for efficiency.
with tf.variable_scope('softmax_linear') as scope:
    weights = tf.get_variable(name='weights', shape=[192, NUM_CLASSES], dtype=dtype
                              , initializer=tf.truncated_normal_initializer(stddev=1 / 192.0, dtype=dtype))

    biases = tf.get_variable(name='biases', shape=[NUM_CLASSES], dtype=dtype,
                             initializer=tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax_linear, labels=input_label))
optim = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(10000):
        next_imgs, next_labels = Data.next_batch(batch_size)
        sc, _ = sess.run([loss, optim], feed_dict={input_images:next_imgs,input_label:next_labels})
        print(sc)

