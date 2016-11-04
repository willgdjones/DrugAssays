import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile
from tensorflow.contrib.layers.python.layers import fully_connected, convolution2d, convolution2d_transpose, max_pool2d, flatten
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import cv2
import numpy as np
import os
import pdb
from time import time
import pickle


class Assay():
    def __init__(self, ID, data_dir):
        self.ID = ID
        self.imageIDs = self.get_imageIDs(self.ID)
        self.label = 0
        self.data_dir = data_dir
        self.processed_images = None
        self.random_samples = None
        
    def get_imageIDs(self, ID):
        prefixes = ['{}_{}'.format(ID,x) for x in ['s1','s2','s3','s4']]
        image_ids = [(x + '_w1.tif', x + '_w2.tif') for x in prefixes]
        return image_ids
    
    def create_processed_images(self):
         return [self.process_image(os.path.join(self.data_dir,image[0]),os.path.join(self.data_dir,image[1])) for image in self.imageIDs]


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
    
    def write_processed(self):
        d = os.path.join('data/processed', self.ID)
        if not os.path.exists(d):
            os.makedirs(d)
        p = self.create_processed_images()
        for i in range(len(p)):
            I = np.round(p[1] * 255).astype(np.int16)
            cv2.imwrite(os.path.join(d, 'r{}.tif'.format(i+1)), I)
            
    def read_processed_images(self):
        d = os.path.join('data/processed', self.ID)
        if not os.path.exists(d):
            print "Not written yet"
        images = [cv2.imread(os.path.join(d,'r{}.tif').format(i+1)) for i in range(4)]
        normalized_images = [normalize(image) for image in images]
        self.processed_images = normalized_images
        return normalized_images

    
    def display(self):
        f, a = plt.subplots(1,4, figsize=(20,5))
        f.suptitle('Assay ID: {}'.format(self.ID),size=20)
        for i in range(4):
            a[i].set_title('Replicate {}'.format(i+1))
            a[i].imshow(self.processed_images[i], interpolation='none')
        return f, a
    
    def generate_random_samples(self, number_per_replicate=10):
        random_samples = []
        for p_im in self.processed_images:
            random_samples.extend([random_chunk2(p_im, 200, 64) for k in range(number_per_replicate)])
        self.random_samples = random_samples
        return random_samples

    def display_random_samples(self):
        f, a = plt.subplots(4,len(self.random_samples) / 4, figsize=(20,8))
        for k in range(4):
            for i in range(10):
                a[k][i].imshow(self.random_samples[i*k + i])

    
    
    
class AutoEncoder():
    def __init__(self, params,):
        self.params = params
        self.modelID = '{}-{}-{}'.format(self.params['training_epochs'],self.params['final_layer'], self.params['keep_prob'])
    # Building the encoder
    def encoder(self,x, keep_prob):


        e_conv1 = convolution2d(x, 32, [5,5], [2,2], padding='SAME')
        #print e_conv1
        e_conv2 = convolution2d(e_conv1, 16, [5,5], [2,2], padding='SAME')
        #print e_conv2
        e_flat1 = flatten(e_conv2)
        #print e_flat1
        #e_flat1 = tf.nn.dropout(e_flat1 keep_prob)
        e_fc1 = fully_connected(e_flat1, self.params['final_layer'], None)
        e_fc1 = tf.nn.dropout(e_fc1, keep_prob)
        return e_fc1


    # Building the decoder
    def decoder(self,x,keep_prob):
        d_fc1 = fully_connected(x, 4096, None)
        d_fc1 = tf.nn.dropout(d_fc1, keep_prob)
        #print d_fc1
        d_unflat1 = tf.reshape(d_fc1, [-1,16,16,16])
        #print d_unflat1
        d_conv1 = convolution2d_transpose(d_unflat1, 32, [5,5], [2,2], padding='SAME')
        #print d_conv1
        d_conv2 = convolution2d_transpose(d_conv1, 3, [5,5], [2,2], padding='SAME')
        #print d_conv2
        return d_conv2

    
       
    def fit(self, train_data, test_data, train_labels, test_labels):
        
        self.train_data = np.array(train_data)
        self.train_labels = np.array(train_labels)
        self.test_data = np.array(test_data)
        self.test_labels = np.array(test_labels)
        tf.reset_default_graph()
        # tf Graph input (only pictures)
        X = tf.placeholder(tf.float32, [None, 64, 64, 3])
        keep_prob = tf.placeholder(tf.float32)
        #encoded_placeholder = tf.placeholder(tf.float32, [None, self.params['final_layer']] )

        # Construct model
        encoder_op = self.encoder(X, keep_prob)
        decoder_op = self.decoder(encoder_op, keep_prob)
        
        # Allow for on the fly decodings
        #decode = self.decoder(encoded_placeholder, keep_prob)

        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = X

        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(self.params['learning_rate']).minimize(cost)
        
        
        decoding_weights = [u'Conv/weights:0',
        u'Conv/biases:0',
        u'Conv_1/weights:0',
        u'Conv_1/biases:0',
        u'fully_connected/weights:0',
        u'fully_connected/biases:0']

        # Initializing the variables
        init = tf.initialize_all_variables()
        saver1 = tf.train.Saver()
        saver2 = tf.train.Saver(
        [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if var.name in decoding_weights])
        tf.add_to_collection('y_pred', y_pred)
        tf.add_to_collection('X', X)
        tf.add_to_collection('encoder_op', encoder_op)
        tf.add_to_collection('keep_prob', keep_prob)
        #tf.add_to_collection('encoded_placeholder', encoded_placeholder)
        #tf.add_to_collection('decode', decode)

        total_batch = len(self.train_data)
        nbatches = total_batch / self.params['batch_size']
        
        
        t = time()
        s = time()
        with tf.Session() as sess:
            sess.run(init)
            for ep in range(self.params['training_epochs']):
                for i in range(int(nbatches)):
                    batch = self.train_data[self.params['batch_size']*i: self.params['batch_size']*(i+1)]
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch, keep_prob: self.params['keep_prob']})
                if ep % self.params['display_step'] == 0:

                    t_, tc = sess.run([optimizer, cost], feed_dict={X: self.test_data, keep_prob:1.})
                    print "Epoch: {}, train cost= {:.9f}, test cost= {:.9f}, speed: {:.4f} chunks p/s".format(ep+1, c, tc, self.params['display_step'] * total_batch / (time() - s))
                    s = time()
                    
            encoded_train, decoded_train = sess.run(
                [encoder_op, decoder_op], feed_dict={X: self.train_data, keep_prob:1.})            
            encoded_test, decoded_test = sess.run(
                [encoder_op, decoder_op], feed_dict={X: self.test_data, keep_prob:1.})
            print ("Saving model {}".format(self.modelID))
            os.system('mkdir -p models/{}'.format(self.modelID))
            saver1.save(sess, './models/{}/vars'.format(self.modelID))
            saver2.save(sess, './models/{}/decoder_vars'.format(self.modelID))
            
        
        

        print ("Training time: {}".format(time() - t))
        
        
        
        
        self.encoded_train = encoded_train
        self.decoded_train = decoded_train
        self.encoded_test = encoded_test
        self.decoded_test = decoded_test

    
    def display_train(self):
        f, a = plt.subplots(2,5, figsize=(20,10))
        idx = np.random.choice(range(len(self.encoded_train)), 5)
        for i in range(5):
            a[0][i].set_title("Training {}, carried through: {}".format(i, self.train_labels[idx][i]))
            a[0][i].imshow(self.train_data[idx][i], interpolation='none')
            a[1][i].set_title("Decoded {}".format(i))
            a[1][i].imshow(multiply_with_overflow(self.decoded_train[idx][i],[1,1,1]), interpolation='none')
            
    def display_test(self):
        f, a = plt.subplots(2,5, figsize=(20,10))
        idx = np.random.choice(range(len(self.encoded_test)), 5)
        for i in range(5):
            a[0][i].set_title("Training {}, carried through: {}".format(i, self.test_labels[idx][i]))
            a[0][i].imshow(self.test_data[idx][i], interpolation='none')
            a[1][i].set_title("Decoded {}".format(i))
            a[1][i].imshow(multiply_with_overflow(self.decoded_test[idx][i],[1,1,1]), interpolation='none')

#tf.reset_default_graph()
#with tf.Session() as sess:
#    saver = tf.train.import_meta_graph('./models/{}/vars.meta'.format(modelID))
#    saver.restore(sess, './models/{}/vars'.format(modelID))
#    y_pred = tf.get_collection('y_pred')[0]
#    X = tf.get_collection('X')[0]
#    keep_prob = tf.get_collection('keep_prob')[0]
#    encoder_op = tf.get_collection('encoder_op')[0]
#    encoded_placeholder = tf.get_collection('encoded_placeholder')[0]
#    decode = tf.get_collection('decode')[0]
#    decoded = sess.run(decode, feed_dict={encoded_placeholder: encode[0:10], keep_prob: 1})
#    variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]


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
    n_image = image.astype(np.float32)
    n_image[:,:,0:2] = n_image[:,:,0:2] / np.max(n_image[:,:,0:2], (0,1))
    return n_image

def random_chunk2(image, width, resize):
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


def generate_assays(IDs,f):
    from multiprocessing import Pool
    p = Pool(30)
    assays = p.map(f, IDs)
    return assays

def extract_data(IDs, labels):
    chunk_data = []
    chunk_labels = []
    for i in range(len(IDs)):
        a = Assay(IDs[i], 'data/Run01test/')
        a.read_processed_images()
        chunk_data.extend(a.generate_random_samples(10))
        chunk_labels.extend([labels[i] for x in range(4*10)])
    print ('data extracted, labels added')
    return chunk_data, chunk_labels



def generate_data(positive_IDs, negative_IDs):
    IDs = positive_IDs + negative_IDs
    labels = [1 for x in positive_IDs] + [0 for x in negative_IDs]

    train_IDs, test_IDs, train_labels, test_labels  = train_test_split(IDs,labels, test_size=0.2)

    train_chunk_data, train_chunk_labels = extract_data(train_IDs, train_labels)
    test_chunk_data, test_chunk_labels = extract_data(test_IDs, test_labels)

    pickle.dump([train_chunk_data, test_chunk_data, train_chunk_labels, test_chunk_labels], open('data.py','wb'))
    return [train_chunk_data, test_chunk_data, train_chunk_labels, test_chunk_labels]
    
