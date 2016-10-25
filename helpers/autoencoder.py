from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile
from tensorflow.contrib.layers.python.layers import fully_connected, convolution2d, convolution2d_transpose, max_pool2d, flatten
from collections import OrderedDict
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 5
batch_size = 256
display_step = 1
examples_to_show = 10


# Building the encoder
def encoder(x):
#     layer_1 = fully_connected(x, params['n_hidden_1'])
#     conv1 = convolution2d(x, 32, [5,5], [2,2], padding='SAME')
#     conv1 = tf.nn.dropout(conv1, keep_prob)
#     conv2 = convolution2d(conv1, 16, [5,5], [2,2], padding='SAME')
#     flat1 = conv2.flatten()
#     fc1 = fully_connected(flat1, 100, None)
    
    e_conv1 = convolution2d(x, 32, [5,5], [2,2], padding='SAME')
    e_conv2 = convolution2d(e_conv1, 16, [5,5], [2,2], padding='SAME')
    e_flat1 = flatten(e_conv2)
    e_fc1 = fully_connected(e_flat1, 200, None)
    e = e_fc1
    return e


# Building the decoder
def decoder(x):
#     fc1 = fully_connected()
    # Encoder Hidden layer with sigmoid activation #1
#     conv_trans1 = convolution2d_transpose(x, 32, [5,5], [2,2], padding='SAME')
#     conv_trans2 = convolution2d_transpose(conv_trans1, 3, [5,5], [2,2], padding='SAME')
    
    d_fc1 = fully_connected(x, 1024)
    d_unflat1 = tf.reshape(d_fc1, [-1,8,8,16])
    d_conv1 = convolution2d_transpose(d_unflat1, 32, [5,5], [2,2], padding='SAME')
    d_conv2 = convolution2d_transpose(d_conv1, 3, [5,5], [2,2], padding='SAME')
    d = d_conv2
    return d

tf.reset_default_graph()
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 32, 32, 3])
# Network Parameters
params = OrderedDict([('n_hidden_1', 128), ('n_hidden_2', 32), ('n_input', 784)])

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()
tf.add_to_collection('y_pred', y_pred)
tf.add_to_collection('X', X)
tf.add_to_collection('encoder_op', encoder_op)


