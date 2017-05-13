import sys
sys.path.append('..') # this allows it to find data_loader
sys.setrecursionlimit(90000)
from keras.models import Sequential,Model
from keras.layers import Activation,Convolution2D,LocallyConnected2D,Flatten,Dense,Input,Reshape,Embedding,Merge,merge,Lambda
from keras.layers.convolutional import UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
import keras.backend as K
K.set_image_dim_ordering('th') # NOTE: channel is first dim after batch.

import data_loader
import train


def local_discriminator(args):
    leaky_alpha = 0.03

    d_input = Input(shape=(1, 25, 25), name='d_input')
    hid = Convolution2D(32, 5, 5, border_mode='same')(d_input)
    hid = LeakyReLU(alpha=0.3)(hid)

    hid = LocallyConnected2D(8, 5, 5, border_mode='valid')(hid)
    hid = LeakyReLU(alpha=leaky_alpha)(hid)
    hid = BatchNormalization(axis=1)(hid) # Note the axis, since the channel is first

    hid = LocallyConnected2D(8, 5, 5, border_mode='valid')(hid)
    hid = LeakyReLU(alpha=leaky_alpha)(hid)
    hid = BatchNormalization(axis=1)(hid) # Note the axis, since the channel is first

    hid = LocallyConnected2D(8, 5, 5, border_mode='valid')(hid)
    hid = LeakyReLU(alpha=leaky_alpha)(hid)
    hid = BatchNormalization(axis=1)(hid) # Note the axis, since the channel is first

    hid = AveragePooling2D(pool_size=(2, 2), border_mode='valid')(hid)
    hid = Flatten()(hid)

    d_proba = Dense(1, activation='linear', name='d_proba')(hid)

    discriminator = Model(input=d_input, output=d_proba)

    print 'Discriminator'
    print discriminator.summary()
    return discriminator


def local_generator(args):
    leaky_alpha = 0.03

    g_input_particle =  Input(shape=(1,), name='g_input_particle')
    g_input_energy = Input(shape=(1,), name='g_input_energy')
    g_input_noise = Input(shape=(1, args['latent_size']), name='g_input_noise')

    g_particle_embed = Embedding(2, args['latent_size'])(g_input_particle)

    input_energy_hid = Dense(args['latent_size'])(g_input_energy)
    input_energy_hid = Reshape((1, args['latent_size']))(input_energy_hid)

    hid = merge([g_particle_embed, input_energy_hid, g_input_noise], mode='mul')

    hid = Dense(6272)(hid)
    hid = Reshape((128, 7, 7))(hid)
    hid = Convolution2D(32, 5, 5, border_mode='same')(hid)

    hid = UpSampling2D(size=(2,2))(hid)

    hid = LocallyConnected2D(6, 5, 5, border_mode='valid')(hid)
    hid = LeakyReLU(alpha=leaky_alpha)(hid)
    hid = BatchNormalization(axis=1)(hid) # Note the axis, since the channel is first

    hid = UpSampling2D(size=(2,2))(hid)

    hid = LocallyConnected2D(6, 3, 3, border_mode='valid')(hid)
    hid = LeakyReLU(alpha=leaky_alpha)(hid)
    hid = BatchNormalization(axis=1)(hid) # Note the axis, since the channel is first

    hid = LocallyConnected2D(1, 2, 2, border_mode='valid', bias=False, activation='relu')(hid)

    generator = Model(input=[g_input_particle, g_input_energy, g_input_noise], output=hid)
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
    # This model has 5x5 hard coded
    assert args['cropped_width'] == 5
    
    G = local_generator(args)
    D = local_discriminator(args)
    GAN = gan(G, D, args)

    train.train(G, D, GAN, args)
