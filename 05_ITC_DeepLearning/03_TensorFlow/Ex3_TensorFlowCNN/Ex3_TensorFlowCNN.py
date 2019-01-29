import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
print('Training.images shape: ', mnist.train.images.shape)
print('Training.labels shape: ', mnist.train.labels.shape)
print('Shape of an image: ', mnist.train.images[0].shape)
print('Example label: ', mnist.train.labels[0])

# Review a few images
image_list = mnist.train.images[0:9]
image_list_labels = mnist.train.labels[0:9]

fig = plt.figure(1, (5., 5.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                 axes_pad=0.3,  # pad between axes in inch.
                 )

for i in range(len(image_list)):
    image = image_list[i].reshape(28, 28)
    grid[i].imshow(image)
    grid[i].set_title('Label: {0}'.format(image_list_labels[i].argmax()))

plt.show()

print('abc')

sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Create placeholders nodes for images and label inputs
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])  # mnist image comes in as 784 vector

# Conv layer 1 - 32x5x5
W1 = weight_variable([5, 5, 1, 32])
b1 = bias_variable([32])
x_conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME'), b1))
x_pool1 = tf.nn.max_pool(x_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Conv layer 2 - 64x5x5
W2      = weight_variable([5, 5, 32, 64])
b2      = bias_variable([64])
x_conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_pool1, W2, strides=[1, 1, 1, 1], padding='SAME'), b2))
x_pool2 = tf.nn.max_pool(x_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Flatten
x_flat = tf.reshape(x_pool2, [-1, 7*7*64])

# Dense fully connected layer   7 * 7 * 64 --> 1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
x_fc1 = tf.nn.relu(tf.add(tf.matmul(x_flat, W_fc1), b_fc1))

# Regularization with dropout
keep_prob = tf.placeholder(tf.float32)
x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

# Classification layer   1024 --> 10
W_fc2  = weight_variable([1024, 10])
b_fc2  = bias_variable([10])
y_est  = tf.add(tf.matmul(x_fc1_drop, W_fc2), b_fc2)


# Probabilities - output from model (not the same as logits)
y = tf.nn.softmax(y_est)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_est))
Optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

# Setup to test accuracy of model
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_est, 1), tf.argmax(y_, 1)), tf.float32))
# TODO  use:  y_est ,  y_  tf.equal and tf.reduce_mean

# Initilize all global variables
sess.run(tf.global_variables_initializer())

# Train model
# Run once to get the model to a good confidence level
for i in range(1000):
    batch = mnist.train.next_batch(100)
    imgs = batch[0]
    lbls = batch[1]

    if i%200 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: imgs, y_: lbls, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

    #sess.run([Optimizer], feed_dict={x: imgs, y_: lbls})
    Optimizer.run(feed_dict={x: imgs, y_: lbls, keep_prob: 1.0})

# Run trained model against test data
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images[0:500],
                                                    y_: mnist.test.labels[0:500], keep_prob: 1.0}))


def plot_predictions(image_list, output_probs=False, adversarial=False):
    prob = y.eval(feed_dict={x: image_list, keep_prob: 1.0})

    pred_list = np.zeros(len(image_list)).astype(int)
    pct_list = np.zeros(len(image_list)).astype(int)

    # Setup image grid
    import math
    cols = 3
    rows = math.ceil(image_list.shape[0] / cols)
    fig = plt.figure(1, (12., 12.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     )

    # Get probs, images and populate grid
    for i in range(len(prob)):
        pred_list[i] = np.argmax(prob[i])  # for mnist index == classification
        pct_list[i] = prob[i][pred_list[i]] * 100

        image = image_list[i].reshape(28, 28)
        grid[i].imshow(image)

        grid[i].set_title('Label: {0} \nCertainty: {1}%' \
                          .format(pred_list[i],
                                  pct_list[i]))

        # Only use when plotting original, partial deriv and adversarial images
        if (adversarial) & (i % 3 == 1):
            grid[i].set_title("Adversarial \nPartial Derivatives")

    plt.show()

    return prob if output_probs else None


# Get 10 2s [:,2] from top 500 [0:500], nonzero returns tuple, get index[0], then first 10 [0:10]
index_of_2s = np.nonzero(mnist.test.labels[0:500][:, 2])[0][0:10]
x_batch = mnist.test.images[index_of_2s]

plot_predictions(x_batch)

print("Done")
