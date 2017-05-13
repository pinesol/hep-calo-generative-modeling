import numpy as np
import os
import sys
sys.setrecursionlimit(10000)
from ConfigParser import SafeConfigParser
import data_loader
from keras.models import Sequential,Model
from keras.layers import Dense,GRU,LSTM,Input,Lambda,Activation,Embedding,Merge,merge,RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
import keras.backend as K
K.set_image_dim_ordering('th')

def rnn_generator(split,batch_size,n_epochs=1,test=0,partition=0):
    '''
    Put data into a format for RNN training or prediction
    '''
    dl = data_loader.DataLoader(split,test=test)
    for ecal,target in dl.batch_iter(partition,batch_size,n_epochs):
        flat = np.array([[x[:,:,i].flatten() for i in range(x.shape[-1])] for x in ecal])
        X = np.log(1+flat[:,:-1,:])
        Y = np.log(1+flat[:,1:,:])
        P = np.zeros((Y.shape[0],2))
        P[np.arange(P.shape[0]),np.array([int(t[0]) for t in target])] = 1
        M = np.array([t[1] for t in target])
        I = np.tile(np.arange(24),X.shape[0]).reshape(X.shape[0],24)
        X_dict = {'X_input': X,
                  'P_input': P,
                  'M_input': M,
                  'I_input':I}
        Y_dict ={'output': Y}
        yield (X_dict,Y_dict)

def generator_data_pull(batch_size):
    '''
    produce inputs to generator
    '''
    latent_size = 200
    gen_particle_types = np.random.randint(0,2,size=(batch_size,1))
    random_list = []
    for b in range(batch_size):
        mean = np.zeros(latent_size)
        cov = np.eye(latent_size)
        random_num = np.random.multivariate_normal(mean,cov)
        random_list.append(random_num)
    gen_noise = np.expand_dims(np.stack(random_list, axis=0),axis=1)
    gen_labels = -1*np.ones(batch_size)
    return gen_particle_types,gen_noise,gen_labels

def save_samples(G,N,batch_size,save_path):
    '''
    save N generated calorimeter images
    '''
    gen_particle_types,gen_noise,gen_labels = generator_data_pull(N)
    images = G.predict([gen_particle_types,gen_noise],batch_size=batch_size)
    np.savez(save_path,particle=gen_particle_types,image=images)

def index_representation(rep_dim):
    m = Input(shape=(24,))
    embedding = Embedding(24, rep_dim,name='embedding')(m)
    model = Model(m,embedding,name='index_rep')
    print model.summary()
    return model
             
def momentum_representation(rep_dim):
    m = Input(shape=(1,))
    m1 = Dense(rep_dim,activation='sigmoid')(m)
    m2 = Dense(rep_dim,activation='sigmoid')(m1)
    model = Model(m,m2,name='momentum_rep')
    print model.summary()
    return model

def particle_representation(rep_dim):
    m = Input(shape=(2,))
    m1 = Dense(rep_dim,activation='sigmoid')(m)
    m2 = Dense(rep_dim,activation='sigmoid')(m1)
    model = Model(m,m2,name='particle_rep')
    print model.summary()
    return model 

def make_model():
    embed_P = particle_representation(10)
    embed_M = momentum_representation(10)
    embed_I = index_representation(10)
    X = Input(shape=(24,400), name='X_input')
    P = Input(shape=(2,), name='P_input')
    M = Input(shape=(1,), name='M_input')
    I = Input(shape=(24,), name='I_input')
    P_emb = RepeatVector(24)(embed_P(P))
    M_emb = RepeatVector(24)(embed_M(M))
    I_emb = embed_I(I)
    concat = merge([X,P_emb,M_emb,I_emb],mode='concat',concat_axis=-1,name='pre_rnn')
    seq = LSTM(800,return_sequences=True)(concat)
    out = TimeDistributed(Dense(600,activation='sigmoid'))(seq)
    out =  TimeDistributed(Dense(400,activation='relu',bias=False),name='output')(out)
    model  = Model([X,P,M,I],out)
    model.compile(optimizer ='Adam',loss = 'mean_squared_error') 
    print model.summary()
    return model
    
def parse_config(path_to_config):
    parser = SafeConfigParser()
    parser.read(path_to_config)

    args = {}
    for section_name in parser.sections():
        args[section_name] = {}
        for name, value in parser.items(section_name):
            args[name] = value

    # format types
    args['n_epochs'] = int(args['n_epochs'])
    args['batch_size'] = int(args['batch_size'])
    args['splits'] = [int(s) for s in args['splits'].split(',')]
    args['test'] = int(args['test'])
    args['sample_freq'] = int(args['sample_freq'])
    args['patience'] = int(args['patience'])


    #print
    for k,v in args.iteritems():
        print k,'\t',v

    return args    
if __name__ == '__main__':
    # parse args
    path2config = sys.argv[1]
    args = parse_config(path2config)
    # make model
    model = make_model()
    # data generator
    train_generator = rnn_generator(args['splits'],args['batch_size'],n_epochs=10**4,test=args['test'],partition=0)
    valid_generator = rnn_generator(args['splits'],args['batch_size'],n_epochs=10**4,test=args['test'],partition=1)
    # sample size calculations
    N = 900000 if args['test']==0 else 100
    train_batches = N*(args['splits'][0]/100.0)*(1/float(args['batch_size']))
    valid_batches = N*(args['splits'][1]/100.0)*(1/float(args['batch_size']))
    # checkpoints
    CSVLogger = CSVLogger(os.path.join(args['save_path'],args['name']+'_log'))
    Stopper = EarlyStopping(patience=args['patience'])
    Saver = ModelCheckpoint(filepath=os.path.join(args['save_path'],args['name']+'_model'), save_best_only=True)
    # fit generator
    model.fit_generator(train_generator,
                        steps_per_epoch=train_batches,
                        epochs=args['n_epochs'],
                        validation_data=valid_generator,
                        validation_steps=valid_batches,
                        callbacks=[CSVLogger,Stopper,Saver])