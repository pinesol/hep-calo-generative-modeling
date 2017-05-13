'''
Learning the Particle Physics By Example discriminator
by Israel
'''

import numpy as np
import os
import sys
from ConfigParser import SafeConfigParser
import data_loader
from keras.models import Sequential
from keras.layers import Activation,Convolution1D,Convolution2D,LocallyConnected2D,Flatten,Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
import keras.backend as K
K.set_image_dim_ordering('th') #this is Theano ordering.  batch size, then channel

def parse_config(path_to_config):
    parser = SafeConfigParser()
    parser.read(path_to_config)
    args = {}
    for section_name in parser.sections():
        for name, value in parser.items(section_name):
            args[name] = value
            
    # formatting args
    args['test'] = int(args['test'])
    args['splits'] = [int(a) for a in args['splits'].split(',')]
    args['patience'] = int(args['patience'])
    args['batch_size'] = int(args['batch_size'])
    args['slice_start'] = int(args['slice_start'])
    args['slice_end'] = int(args['slice_end'])


    # display
    for k,v in args.iteritems():
        print k,':',v,';',type(v)

    return args

def my_generator(split,batch_size,slice_start=0,slice_end=24,n_epochs=1,test=False,partition=0):
    '''Wrapper on data_loader.DataLoader to yield batches that only grab particular slice&target
       This wrapper plays nicely with Keras' model.fit_generator() functionality
       Notes:
       1) split=[train, valid, test] where train+valid+test=100, can have arbitrary number of partitions
       2) partition_batch_iter(N_batch, N_epochs,parition_index) ; set N_epochs to 1 if using Keras' fit_generator'''
    dl = data_loader.DataLoader(split,test=test)
    for ecal,target in dl.batch_iter(partition,batch_size,n_epochs):
        X = np.array([np.expand_dims(x[:,:,slice_start:slice_end+1].mean(axis=2),axis=0) for x in ecal])

        #save the particle type, which is y[0], not the momentum
        Y = np.array([y[0] for y in target])
        yield (X,Y)

def make_model():
    print "compiling model"
    leaky_alpha = 0.03
    input_size = 20

    model = Sequential()
    model.add(Convolution2D(32,5,5, border_mode='same', input_shape=(1,input_size,input_size)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(LocallyConnected2D(8, 5, 5, border_mode='valid'))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(BatchNormalization())
    model.add(LocallyConnected2D(8, 5, 5, border_mode='valid'))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(BatchNormalization())
    model.add(LocallyConnected2D(8, 5, 5, border_mode='valid'))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(BatchNormalization())
    model.add(LocallyConnected2D(8, 3, 3, border_mode='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['binary_accuracy'])
    print model.summary()   
    return model

def train(model,splits,slice_start,slice_end,batch_size,patience,test,logs_path,name):
    print "training starting now"
    total_per_epoch = 100 if test else 99000
    train_per_epoch = (splits[0]/100.0)*total_per_epoch
    valid_per_epoch = (splits[1]/100.0)*total_per_epoch
    
    gen_train = my_generator(splits,batch_size,slice_start=slice_start,slice_end=slice_end,partition=0,n_epochs=100,test=test)
    gen_valid = my_generator(splits,batch_size,slice_start=slice_start,slice_end=slice_end,partition=1,n_epochs=100,test=test)
    
    # callbacks
    saver = CSVLogger(os.path.join(logs_path,name+'_log.csv'))
    stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    ckpt = ModelCheckpoint(os.path.join(logs_path,name+'_model.ckpt'),verbose=0,save_best_only=True,
                           save_weights_only=False,mode='auto',period=1)  
    
    model.fit_generator(gen_train,train_per_epoch,100,
                        validation_data = gen_valid,
                        nb_val_samples = valid_per_epoch,
                        callbacks = [saver, stopper, ckpt],
                        max_q_size = 1)

def main(args):
    model = make_model()
    train(model,
          args['splits'],
          args['slice_start'],
          args['slice_end'],
          args['batch_size'],
          args['patience'],
          args['test'],
          args['logs_path'],
          args['name'])
    

if __name__=='__main__':
    ptc = sys.argv[1]
    args = parse_config(ptc)
    args['name'] = os.path.basename(ptc).split('.')[0]
    main(args)
    
    
