import sys
sys.path.append('..') # this allows it to find data_loader
sys.setrecursionlimit(90000)
from keras.models import Sequential,Model
from keras.layers import Activation,Convolution2D,LocallyConnected2D,Flatten,Dense,Input,Reshape,Embedding,Merge,merge
from keras.layers.convolutional import UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
import keras.backend as K
K.set_image_dim_ordering('th') # NOTE: channel is first dim after batch.

import numpy as np

import data_loader
import train


# Creates a randomized num_rows by num_cols grid of floats.
# The total energy of the grid will be 18.87*input_energy - 183.7, which is the
# regression that maps input to output energy, as determined by the data.
# Each row of the grid is half of a gaussian curve, because that's kind of what
# an unrolled ecal looks like.
def gauss_z(input_energy, num_rows, num_cols):
    # constants derived from input/output energy chart
    output_energy = 18.87*input_energy - 183.7
    row_energy = output_energy / num_rows
    def gauss(x):
        mu = 0.0
        # sigma = 6.0 because then the curve goes to zero around 20, like the data does.
        sig = 6.0
        return 1./(np.sqrt(2.*np.pi)*sig) * np.exp(-np.power((x - mu)/sig, 2.)/2)
    nonrandom_row = [2 * row_energy * gauss(col) for col in xrange(num_cols)]
    # In order to make the row's sum equal row_energy, the first element must halved,
    # since half of it belongs to the negative side of the gaussian.
    nonrandom_row[0] *= 0.5
    nonrandom_grid = np.repeat([nonrandom_row], num_rows, axis=0)
    noise_grid = np.random.randn(num_rows, num_cols)
    grid = nonrandom_grid + noise_grid
    return grid


def gauss_discriminator(args):
    DIMS = (args['cropped_width']*args['cropped_width'], data_loader.DATA_DIM[2])
    
    d_input =  Input(shape=(1, DIMS[0], DIMS[1]), name='d_input')
    d_input_particle = Input(shape=(1,), name='d_input_particle')    
    d_input_energy = Input(shape=(1,), name='g_input_energy')
    
    hid = Convolution2D(32, 5, 5, border_mode='same')(d_input)
    hid = Flatten()(hid)
    hid = merge([hid, d_input_particle, d_input_energy], mode='concat')

    hid = Dense(8, activation='relu')(hid)
    
    d_proba = Dense(1, activation='linear', name='d_proba')(hid)    

    discriminator = Model(input=[d_input, d_input_particle, d_input_energy], output=d_proba)
    discriminator.compile(optimizer=train.optimizer(), loss=train.wasser_loss)
    print 'Discriminator'
    print discriminator.summary() 
    return discriminator


# TODO iterate on this with upsamling, locally connected layers, etc.
def gauss_generator(args):
    DIMS = (args['cropped_width']*args['cropped_width'], data_loader.DATA_DIM[2])
    
    FLAT_LENGTH = DIMS[0]*DIMS[1]
    # particle
    g_input_particle = Input(shape=(1,), name='g_input_particle')
    particle_embed = Embedding(2, FLAT_LENGTH)(g_input_particle)
    # input energy
    g_input_energy = Input(shape=(1,), name='g_input_energy')
    input_energy_hid = Dense(FLAT_LENGTH)(g_input_energy)
    input_energy_hid = Reshape((1, FLAT_LENGTH))(input_energy_hid)
    # noise
    g_input_noise = Input(shape=(1, DIMS[0], DIMS[1]), name='g_input_noise')
    reshaped_noise = Reshape((1, FLAT_LENGTH))(g_input_noise)
    # merge them by multiplying
    merged = merge([particle_embed, input_energy_hid, reshaped_noise], mode='mul')

    hid =  Dense(FLAT_LENGTH, activation='relu')(merged)
    generated = Reshape((1, DIMS[0], DIMS[1]))(hid)

    generator = Model(input=[g_input_particle, g_input_energy, g_input_noise], output=generated)
    print 'Generator'
    print generator.summary()
    return generator


def gauss_gan(G, D, args):
    '''
    For both G and D we pass the correct particle-type as label.
    '''
    DIMS = (args['cropped_width']*args['cropped_width'], data_loader.DATA_DIM[2])
    
    gan_input_particle = Input(shape=(1,), name='gan_input_particle')
    gan_input_energy = Input(shape=(1,), name='gan_input_energy')
    gan_input_noise = Input(shape=(1, DIMS[0], DIMS[1]),name='gan_input_noise')
    fake = G([gan_input_particle, gan_input_energy, gan_input_noise])
    d_proba = D([fake, gan_input_particle, gan_input_energy])

    GAN = Model(input=[gan_input_particle, gan_input_energy, gan_input_noise], output=d_proba)
    # temporarily making D untrainable so the summary shows that D won't be trainale during the GAN.
    # We reset it during training.
    train.make_trainable(D, False)
    GAN.compile(optimizer=train.optimizer(), loss=train.wasser_loss)
    print 'GAN'
    print GAN.summary()
    train.make_trainable(D, True)
    return GAN


if __name__ == '__main__':
    # parse args
    path2config = sys.argv[1]
    args = train.parse_config(path2config)
    G = gauss_generator(args)
    D = gauss_discriminator(args)
    GAN = gauss_gan(G, D, args)

    train.train(G, D, GAN, args)
