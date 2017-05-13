'''
Summary of experiment:
- uses minibatch discrimination, inspired by this blog post:
http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
- batchsize 8
- Gen width 1200
- Layers 2
- Gen activation ReLU
- Disc activation sigmoid
'''
import train

import time
import numpy as np
import tensorflow as tf
import os
import sys
from tensorflow.contrib.layers.python import layers


tf.flags.DEFINE_string("config","default", "The name of the hyperparameter configuration.")
tf.flags.DEFINE_string("exp_name", "", "The name of the experiment. controls the output dir.")
FLAGS = tf.flags.FLAGS


WIDTH = 20 # Image width (X,Y axes)

def generator(z, y):
    '''
    args:
        z: random vector
    returns:
        net: generator network
    '''
    zf = layers.fully_connected(z, 1000, scope='fcz')
    yf = layers.fully_connected(y, 200, scope='fcy')
    net = tf.concat(1, [zf, yf])
    net = layers.fully_connected(net, 1200, activation_fn=tf.nn.relu, scope='fc1')
    net = layers.fully_connected(net, 1200, activation_fn=tf.nn.relu, scope='fc2')
    net = layers.fully_connected(net, WIDTH*WIDTH, scope='fc3')
    return net


def compare_to_minibatch(input, num_kernels=5, kernel_dim=3):
    '''
    take output of intermediate layer of the discriminator,
    and compare individual samples within the minibatch
    '''
    #multiply discriminator layer by 3D tensor to produce matrix
    x = layers.fully_connected (input, num_kernels * kernel_dim)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))

    #compute L1-distance between rows of matrix
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)

    #apply negative exponential
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat(1, [input, minibatch_features])

def discriminator(x,y):
    '''
    args:
        x: data vector
    returns:
        net: discriminator network
    '''
    xf = layers.fully_connected(x, 1000, scope='fcx')
    yf = layers.fully_connected(y, 200, scope='fcy') 
    net = tf.concat(1, [xf, yf])
    net = layers.dropout(net, 0.2, scope='do1') 
    net = layers.fully_connected(net, 1200, activation_fn=tf.nn.sigmoid, scope='fc1')
    net = layers.dropout(net, 0.5, scope='do2')
    net = layers.fully_connected(net, 1200, activation_fn=tf.nn.sigmoid, scope='fc2')
    net = compare_to_minibatch(net)
    # no activation function because it's in the cost function used later.
    net = layers.fully_connected(net, 1, scope='fc3', activation_fn=None) 
    return net


class DefaultConfig(object):
    batch_size = 8 #Small so that compare_to_minibatch works
    samples_batch_size = 16 #number of samples of each particle type and momentum bin.
    z = 100
    max_epochs = 100
    lr = 0.0002
    beta1=0.5
    momentum = 0.5
    decay_steps = 100000
    decay_factor = 0.96
    train_dir = '/scratch/cdg356/udon/exp/'
    slice_idx = 12
    data_partition = [80, 10, 10]
    test = False
    generator_hits = 4
    save_freq = 10000

# fewer epochs and bigger batch size so tests are kinda fast. 
class TestConfig(object):
    batch_size = 16
    samples_batch_size = 10 #number of samples of each particle type and momentum bin.
    z = 100
    max_epochs = 5
    lr = 0.0002
    beta1=0.5
    momentum = 0.5
    decay_steps = 100000
    decay_factor = 0.96
    train_dir = '../exp/'
    slice_idx = 12
    data_partition = [80, 10, 10]
    test = True  # This is what tells it to use the test data
    generator_hits = 1
    save_freq = 5

configs_dict = {
    'default': DefaultConfig(),
    'test': TestConfig()
}


def main(argv=None):  # pylint: disable=unused-argument
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

   
    config = configs_dict[FLAGS.config]
    if FLAGS.exp_name:
        exp_dir = os.path.join(config.train_dir, FLAGS.exp_name)
        if os.path.isdir(exp_dir):
            print 'output dir {} already exists!'.format(exp_dir)
            sys.exit(1)
        config.train_dir = os.path.join(config.train_dir, FLAGS.exp_name)
    else:
        config.train_dir = os.path.join(config.train_dir,
                                        'exp_' + time.strftime("%Y.%m.%d.%H.%M.%S") + '/')
        
    train.build_train_2d(config, generator, discriminator)

if __name__ == '__main__':
    tf.app.run()
