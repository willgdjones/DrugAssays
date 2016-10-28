import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile
from tensorflow.contrib.layers.python.layers import fully_connected, convolution2d, convolution2d_transpose, max_pool2d, flatten
from collections import OrderedDict
import tensorflow as tf
import cv2
import numpy as np
import os
from multiprocessing import Pool


class Assay():
    def __init__(self, ID, data_dir):
        self.ID = ID
        self.imageIDs = self.get_imageIDs(self.ID)
        self.label = 0
        self.data_dir = data_dir
        self.p_images = self.get_p_images(self.imageIDs, self.data_dir)
        
    def get_imageIDs(self, ID):
        prefixes = ['{}_{}'.format(ID,x) for x in ['s1','s2','s3','s4']]
        image_ids = [(x + '_w1.tif', x + '_w2.tif') for x in prefixes]
        return image_ids
    
    def get_p_images(self, images, data_dir):
         return [self.process_image(os.path.join(data_dir,image[0]),os.path.join(data_dir,image[1])) for image in images]


    def process_image(self,r_filepath, g_filepath):
        image1 = cv2.imread(r_filepath)
        image2 = cv2.imread(g_filepath)
        R = np.array(image1[:,:,0])
        G = np.array(image2[:,:,0])
        B = np.zeros(R.shape)
        raw_image = np.zeros([2160,2160,3])
        raw_image[:,:,0], raw_image[:,:,1], raw_image[:,:,2] = R, G, B
        n_image = normalize(raw_image)
        p_image = multiply_with_overflow(n_image, [3,3,1])
        return p_image
    
    
    
class AutoEncoder():
    def __init__(self, params):
        self.params = params
    # Building the encoder
    def encoder(self,x):
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
    def decoder(self,x):
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
    
       
    def fit(self, train):
        
        self.train = train
        tf.reset_default_graph()
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, 32, 32, 3])

        # Construct model
        encoder_op = self.encoder(X)
        decoder_op = self.decoder(encoder_op)

        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = X

        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(self.params['learning_rate']).minimize(cost)

        # Initializing the variables
        init = tf.initialize_all_variables()
        #saver = tf.train.Saver()
        #tf.add_to_collection('y_pred', y_pred)
        #tf.add_to_collection('X', X)
        #tf.add_to_collection('encoder_op', encoder_op)

        total_batch = len(train)
        nbatches = total_batch / self.params['batch_size']

        with tf.Session() as sess:
            sess.run(init)
            for ep in range(self.params['training_epochs']):
                for i in range(int(nbatches)):
                    batch = train[self.params['batch_size']*i: self.params['batch_size']*(i+1)]
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch})
                if ep % self.params['display_step'] == 0:
                    print "Epoch:" + '%04d' % (ep+1) + ', ' + "cost=" + "{:.9f}".format(c)
            encoded, decoded = sess.run(
                [encoder_op, decoder_op], feed_dict={X: train})
        self.encoded = encoded
        self.decoded = decoded
        return (encoded, decoded)
    
    def display(self):
        f, a = plt.subplots(2,5, figsize=(20,10))
        for i in range(5):
            a[0][i].set_title("Training {}".format(i))
            a[0][i].imshow(self.train[i], interpolation='none')
            a[1][i].set_title("Decoded {}".format(i))
            a[1][i].imshow(multiply_with_overflow(self.decoded[i],[1,1,1]), interpolation='none')
        return f, a




def multiply_with_overflow(image, factor):
    m_image = np.zeros_like(image)

    m_imageR = cv2.multiply(image[:,:,0], factor[0])
    m_imageR[m_imageR > 1] = 1

    m_imageG = cv2.multiply(image[:,:,1], factor[1])
    m_imageG[m_imageG > 1] = 1

    m_imageB = cv2.multiply(image[:,:,2], factor[2])
    m_imageB[m_imageB > 1] = 1

    m_image[:,:,0] = m_imageR
    m_image[:,:,1] = m_imageG
    m_image[:,:,2] = m_imageB

    return m_image

def normalize(image):
    image = np.array(image)
    max_r = max(image[:,:,0].flatten()) if max(image[:,:,0].flatten()) != 0 else 1
    max_g = max(image[:,:,1].flatten()) if max(image[:,:,1].flatten()) != 0 else 1
    max_b = max(image[:,:,2].flatten()) if max(image[:,:,2].flatten()) != 0 else 1

    n_image = np.zeros_like(image)
    n_image[:,:,0] = image[:,:,0] / max_r
    n_image[:,:,1] = image[:,:,1] / max_g
    n_image[:,:,2] = image[:,:,2] / max_b
    return n_image

def random_chunk2(width, resize):
    X = width
    canvas = np.zeros(image.shape[0:2], dtype= "uint8")

    w = image.shape[0]
    r = int(np.floor(np.sqrt(2) * X))

    center_range = w - 2*np.ceil(r)

    x = np.random.randint(center_range)
    y = np.random.randint(center_range)

    C_mask = cv2.circle(canvas, (r + x,r + y), r, (255,255,255), -1)
    O = cv2.bitwise_and(image,image,mask=C_mask)

    theta = np.random.randint(360)

    M = cv2.getRotationMatrix2D((r + x,r + y), theta, 1.0)

    rotated = cv2.warpAffine(O, M, (w,w))

    chunk = rotated[y+r-X:y+r+X,x+r-X:x+r+X]
    resized_chunk = cv2.resize(chunk, (resize,resize), interpolation=cv2.INTER_CUBIC)

    di = np.random.randint(-1,3)

    if di == 2:
        return resized_chunk

    else:
        return cv2.flip(resized_chunk, di)

def create_assay(ID, data_dir='Run01test/'):
    a = Assay(ID, data_dir)
    return a


def generate_assays(IDs):
    p = Pool(30)
    assays = p.map(create_assay, IDs)
    return assays
