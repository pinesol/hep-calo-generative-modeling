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


# TODO doesn't work
# TODO using Cropping2D instead of lambda layers
def split_generator(args):
    FLAT_LENGTH = train.FLAT_CROPPED_DIMS[0]*train.FLAT_CROPPED_DIMS[1]
    # particle
    g_input_particle = Input(shape=(1,), name='g_input_particle')
    particle_embed = Embedding(2, FLAT_LENGTH)(g_input_particle)
    # input energy
    g_input_energy = Input(shape=(1,), name='g_input_energy')
    input_energy_hid = Dense(FLAT_LENGTH)(g_input_energy)
    input_energy_hid = Reshape((1, FLAT_LENGTH))(input_energy_hid)
    # noise
    g_input_noise = Input(shape=(1, train.FLAT_CROPPED_DIMS[0], train.FLAT_CROPPED_DIMS[1]), name='g_input_noise')
    reshaped_noise = Reshape((1, FLAT_LENGTH))(g_input_noise)
    # merge them by multiplying
    merged = merge([particle_embed, input_energy_hid, reshaped_noise], mode='mul')

    LEFT_HEIGHT = 25
    RIGHT_HEIGHT = train.FLAT_CROPPED_DIMS[0] - LEFT_HEIGHT
    
    def left_slice(x):
        return x[:, :, 0:train.FLAT_CROPPED_DIMS[1]*LEFT_HEIGHT]
    left = Lambda(left_slice, output_shape=(1, train.FLAT_CROPPED_DIMS[1]*LEFT_HEIGHT))(merged)

    #left_dense = Dense(train.FLAT_CROPPED_DIMS[1]*LEFT_HEIGHT, activation='relu')(left) # TODO changee
#    left = Reshape((1, LEFT_HEIGHT, train.FLAT_CROPPED_DIMS[1]))(left)
#    left = LocallyConnected2D(3, 5, 5, border_mode='valid')(left) # -4,-4 TODO this doesn't work!!!
#    left = LeakyReLU(alpha=0.03)(left)
    left = Dense(train.FLAT_CROPPED_DIMS[1]*LEFT_HEIGHT, activation='relu')(left)
    left = Reshape((1, LEFT_HEIGHT, train.FLAT_CROPPED_DIMS[1]))(left)    

    def right_slice(x):
        return x[:, :, train.FLAT_CROPPED_DIMS[1]*LEFT_HEIGHT:]
    right = Lambda(right_slice, output_shape=(1, train.FLAT_CROPPED_DIMS[1]*RIGHT_HEIGHT))(merged)
    right = Reshape((1, RIGHT_HEIGHT, train.FLAT_CROPPED_DIMS[1]))(right)
    right = Convolution2D(8, 5, 5, border_mode='same')(right)
    right = LeakyReLU(alpha=0.03)(right)
    right = Convolution2D(1, 5, 5, border_mode='same', activation='relu')(right)
#    right_dense = Dense(train.FLAT_CROPPED_DIMS[1]*RIGHT_HEIGHT, activation='relu')(right) # TODO changee

    hid = merge([left, right], mode='concat', concat_axis=2) # TODO replace with Merge() layer?
    generated = Reshape((1, train.FLAT_CROPPED_DIMS[0], train.FLAT_CROPPED_DIMS[1]))(hid)

    generator = Model(input=[g_input_particle, g_input_energy, g_input_noise], output=generated)
    print 'Generator'
    print generator.summary()
    return generator


# TODO old sauce
# def local_discriminator(args):
#     DIMS = (args['cropped_width']*args['cropped_width'], data_loader.DATA_DIM[2])
    
#     d_input =  Input(shape=(1, DIMS[0], DIMS[1]), name='d_input')
#     d_input_particle = Input(shape=(1,), name='d_input_particle')
#     d_input_energy = Input(shape=(1,), name='d_input_energy')
    
#     hid = Convolution2D(32, 5, 5, border_mode='same')(d_input)
#     hid = Flatten()(hid)
#     hid = merge([hid, d_input_particle, d_input_energy], mode='concat')

#     hid = Dense(8, activation='relu')(hid)
    
#     d_proba = Dense(1, activation='linear', name='d_proba')(hid)
#     discriminator = Model(input=[d_input, d_input_particle, d_input_energy], output=d_proba)
# #    discriminator = Model(input=d_input, output=d_proba)
#     discriminator.compile(optimizer=train.optimizer(), loss=train.wasser_loss)
#     print 'Discriminator'
#     print discriminator.summary() 
#     return discriminator


def local_discriminator(args):
    leaky_alpha = 0.03
    
    DIMS = (args['cropped_width']*args['cropped_width'], data_loader.DATA_DIM[2])

    d_input = Input(shape=(1, DIMS[0], DIMS[1]), name='d_input')
    d_input_particle = Input(shape=(1,), name='d_input_particle') #TODO unused atm
    d_input_energy = Input(shape=(1,), name='d_input_energy') #TODO unused atm

    d_particle_embed = Embedding(2, DIMS[0]*DIMS[1])(d_input_particle)
    d_particle_embed = Reshape((1, DIMS[0], DIMS[1]))(d_particle_embed)
    
    input_energy_hid = Dense(DIMS[0]*DIMS[1])(d_input_energy)
    input_energy_hid = Reshape((1, DIMS[0], DIMS[1]))(input_energy_hid)
    
    d_input_noisy = merge([d_particle_embed, input_energy_hid, d_input], mode='mul')

#TODO    hid = Convolution2D(32, 5, 5, border_mode='same')(d_input)
    hid = Convolution2D(32, 5, 5, border_mode='same')(d_input_noisy)
    hid = LeakyReLU(alpha=leaky_alpha)(hid)
#    hid = BatchNormalization(axis=1)(hid) #Note the axis, since the channel is first

    hid = LocallyConnected2D(8, 5, 5, border_mode='valid', subsample=(2,2))(hid)
    hid = LeakyReLU(alpha=leaky_alpha)(hid)
    # hid = BatchNormalization(axis=1)(hid) #Note the axis, since the channel is first

    hid = LocallyConnected2D(8, 5, 5, border_mode='valid', subsample=(1,1))(hid)
    hid = LeakyReLU(alpha=leaky_alpha)(hid)
    # hid = BatchNormalization(axis=1)(hid) #Note the axis, since the channel is first

    hid = AveragePooling2D(pool_size=(2, 2), border_mode='valid')(hid)
    hid = Flatten()(hid)

#    hid = merge([hid, d_input_particle, d_input_energy], mode='concat') # TODO this is experimental

    d_proba = Dense(1, activation='linear', name='d_proba')(hid)
    
    discriminator = Model(input=[d_input, d_input_particle, d_input_energy], output=d_proba)
    discriminator.compile(optimizer=train.optimizer(), loss=train.wasser_loss)

    print 'Discriminator'
    print discriminator.summary()
    return discriminator


def local_generator(args):
    DIMS = (args['cropped_width']*args['cropped_width'], data_loader.DATA_DIM[2])
    
    g_input_particle = Input(shape=(1,), name='g_input_particle')
    g_input_energy = Input(shape=(1,), name='g_input_energy')
    g_input_noise = Input(shape=(1, args['latent_size']), name='g_input_noise')

    g_particle_embed = Embedding(2, args['latent_size'])(g_input_particle)
    input_energy_hid = Dense(args['latent_size'])(g_input_energy)
    input_energy_hid = Reshape((1, args['latent_size']))(input_energy_hid)
    g_input_noisy = merge([g_particle_embed, input_energy_hid, g_input_noise], mode='mul')

    hid = Reshape((1, 16, 16))(g_input_noisy)
    hid = Convolution2D(16, 7, 7)(hid)
    hid = LeakyReLU(alpha=0.03)(hid)
    hid = BatchNormalization(axis=1)(hid) # Note the axis, since the channel is first
    hid = UpSampling2D(size=(2,2))(hid)
    hid = LocallyConnected2D(6, 5, 5)(hid)
    hid = LeakyReLU(alpha=0.03)(hid)
    hid = BatchNormalization(axis=1)(hid) # Note the axis, since the channel is first
    hid = UpSampling2D(size=(2,2))(hid)
    hid = LocallyConnected2D(1, 8, 8, activation='relu')(hid)
    generated = Reshape((1, DIMS[0], DIMS[1]))(hid)
    
    generator = Model(input=[g_input_particle, g_input_energy, g_input_noise], output=generated)

    print 'Generator'
    generator.summary()

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
#    d_proba = D(generated_image)
    
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
    G = local_generator(args)
    D = local_discriminator(args)
    GAN = gan(G, D, args)

    train.train(G, D, GAN, args)
