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

import data_loader
import train

def small_discriminator(args):
    DIMS = (args['cropped_width']*args['cropped_width'], data_loader.DATA_DIM[2])
    
    d_input =  Input(shape=(1, DIMS[0], DIMS[1]), name='d_input')
    d_input_particle = Input(shape=(1,), name='d_input_particle')
    d_input_energy = Input(shape=(1,), name='d_input_energy')
    
    hid = Convolution2D(32, 5, 5, border_mode='same')(d_input)
    hid = Flatten()(hid)
    hid = merge([hid, d_input_particle, d_input_energy], mode='concat')
#    hid = merge([hid, d_input_energy], mode='concat')

    hid = Dense(8, activation='relu')(hid)
    
    d_proba = Dense(1, activation='linear', name='d_proba')(hid)
    discriminator = Model(input=[d_input, d_input_particle, d_input_energy], output=d_proba)
    discriminator.compile(optimizer=train.optimizer(), loss=train.wasser_loss)
    print 'Discriminator'
    print discriminator.summary() 
    return discriminator

def small_generator(args):
    DIMS = (args['cropped_width']*args['cropped_width'], data_loader.DATA_DIM[2])
    
    # particle
    g_input_particle = Input(shape=(1,), name='g_input_particle')
    # input energy
    g_input_energy = Input(shape=(1,), name='g_input_energy')
    # noise
    g_input_noise = Input(shape=(1, args['latent_size']), name='g_input_noise')

    # initial transform
    g_particle_embed = Embedding(2, args['latent_size'])(g_input_particle)
    input_energy_hid = Dense(args['latent_size'])(g_input_energy)
    input_energy_hid = Reshape((1, args['latent_size']))(input_energy_hid)
    # merge them by multiplying
    g_input_noisy = merge([g_particle_embed, input_energy_hid, g_input_noise], mode='mul')

    # Model
    # TODO iterate on this with upsamling, locally connected layers, etc.
    hid =  Dense(DIMS[0]*DIMS[1], activation='relu')(g_input_noisy)
    generated = Reshape((1, DIMS[0], DIMS[1]))(hid)

    generator = Model(input=[g_input_particle, g_input_energy, g_input_noise], output=generated)
    # TODO I'm a little afraid that I should have compiled this, but I'm pretty sure that would be
    # superfluous.
    print 'Generator'
    print generator.summary()
    return generator


def gan(G, D, args):
    '''
    For both G and D we pass the correct particle-type as label.
    '''
    gan_input_particle = Input(shape=(1,), name='gan_input_particle')
    gan_input_energy = Input(shape=(1,), name='gan_input_energy')
    gan_input_noise = Input(shape=(1, args['latent_size']),name='gan_input_noise')
    
    generated_image = G([gan_input_particle, gan_input_energy, gan_input_noise])
    d_proba = D([generated_image, gan_input_particle, gan_input_energy])

    GAN = Model(input=[gan_input_particle, gan_input_energy, gan_input_noise], output=d_proba)

    # temporarily making D untrainable so the summary shows that D won't be trainale during the GAN.    
    # We reset it during training, and the freeze it every cycle.
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
    G = small_generator(args)
    D = small_discriminator(args)
    GAN = gan(G, D, args)

    train.train(G, D, GAN, args)
