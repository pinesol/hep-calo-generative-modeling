import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import numpy as np
import os
import pandas as pd
import datetime
import sys
sys.path.append('..') # this allows it to find data_loader
sys.setrecursionlimit(90000)
import time
import cPickle as pickle
from ConfigParser import SafeConfigParser
import data_loader
from keras.models import Sequential,Model
from keras.optimizers import RMSprop
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
import keras.backend as K
K.set_image_dim_ordering('th') # NOTE: channel is first dim after batch.

# The mean never varies by much, regardless of the cropping. It's always pretty much zero.
TRAIN_MEAN = 0.0


# Key: tuple(cropped_width, logged, averaged)

# no log, no average
STDDEV_MAP = {}
STDDEV_MAP[(5, False, False)] = 25.995
STDDEV_MAP[(10, False, False)] = 13.018
STDDEV_MAP[(data_loader.DATA_DIM[0], False, False)] = 6.511
# no log, average
STDDEV_MAP[(5, True, False)] = 1.470
STDDEV_MAP[(10, True, False)] = 0.789
STDDEV_MAP[(data_loader.DATA_DIM[0], True, False)] = 0.401

# no log, average
STDDEV_MAP[(5, False, True)] = 18.055
STDDEV_MAP[(10, False, True)] = 9.037
STDDEV_MAP[(data_loader.DATA_DIM[0], False, True)] = 4.519
# log, average
STDDEV_MAP[(5, True, True)] = 1.589
STDDEV_MAP[(10, True, True)] = 0.834
STDDEV_MAP[(data_loader.DATA_DIM[0], True, True)] = 0.419


def parse_config(path_to_config):
    parser = SafeConfigParser()
    parser.read(path_to_config)

    args = {}
    for section_name in parser.sections():
        args[section_name] = {}
        for name, value in parser.items(section_name):
            args[name] = value

    # format types
    args['cropped_width'] = int(args['cropped_width']) # 0 means no cropping
    args['log_data'] = bool(int(args['log_data']))
    args['n_epochs'] = int(args['n_epochs'])
    args['batch_size'] = int(args['batch_size'])
    assert args['batch_size'] > 0
    args['update_ratio'] = int(args['update_ratio'])
    assert args['update_ratio'] > 0    
    args['latent_size'] = int(args['latent_size'])
    assert args['latent_size'] > 0        
    args['splits'] = [int(s) for s in args['splits'].split(',')]
    args['test'] = bool(int(args['test']))
    args['num_samples'] = int(args['num_samples'])
    assert args['num_samples'] > 0
    args['averaged_data'] = bool(int(args['averaged_data']))
    args['use_gauss'] = bool(int(args['use_gauss']))
    
    #print
    for k,v in args.iteritems():
        print k,'\t',v

    return args


def data_generator(args, partition=0):
    splits = args['splits']
    batch_size =  args['batch_size']
    test = args['test']
    cropped_width = args['cropped_width']
    log_data = args['log_data'] # TODO
    averaged = args['averaged_data']

    # compute normalization stddev based on width, logging, and averaging
    stddev = STDDEV_MAP[cropped_width, log_data, averaged]
    
    if averaged:
        data_loader._SCRATCH_DIR = data_loader._AVG_SCRATCH_DIR
        data_loader._FILENAME_REGEX = data_loader._AVG_FILENAME_REGEX
    if test:
        assert not averaged, 'averaged should not be true when testing'
        dl = data_loader.DataLoader(splits, test=True, local_test_data_dir='../..')
    else:
        dl = data_loader.DataLoader(splits, test=False)

    for ecals, targets in dl.batch_iter(partition, batch_size, num_epochs=1):
        if cropped_width < data_loader.DATA_DIM[0]:
            ecals = data_loader.truncate_ecals(ecals, (cropped_width, cropped_width))
        if log_data:
            ecals = data_loader.log_ecals(ecals)
        # normalize using the stddev computed above based on width+logging+averaging
        ecals = data_loader.normalize_ecals(ecals, TRAIN_MEAN, stddev)
        ecals = data_loader.unroll_ecals(ecals)
        # NOTE this is needed to make it fit with the model architecture
        ecals = np.expand_dims(ecals, axis=1)
        particle_types = np.array([y[0] for y in targets])
        input_energies = np.array([y[1] for y in targets])        
        yield (ecals, particle_types, input_energies)


def make_trainable(net, val):
    '''
    makes model (net) trainable/not-trainable based on input (val)
    '''
    net.trainable = val
    for l in net.layers:
        l.trainable = val

# https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/WassersteinGAN/src/model
def wasser_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def clip_weights(network, c=0.01):
    ws = network.get_weights()
    ws = [np.clip(w, -c, c) for w in ws]
    network.set_weights(ws)


def plot_losses(args, timestamp, epoch, d_real_losses, d_gen_losses, g_losses):
    print 'Graphing Losses'

    # In the discriminator, the real image labels are negated, so I do it again here to undo that.
    d_real_losses = -np.array(d_real_losses)
    d_gen_losses = np.array(d_gen_losses)
    # the discriminator loss is real loss minus the generated loss
    d_losses = d_real_losses - d_gen_losses
    # In the generator, the generated image labels are negated, so I do it again here to undo that.
    g_losses = -np.array(g_losses)
    
    print 'Epoch {} D real loss: {:f}'.format(epoch, np.mean(d_real_losses))
    print 'Epoch {} D gen loss: {:f}'.format(epoch, np.mean(d_gen_losses))
    print 'Epoch {} D loss: {:f}'.format(epoch, np.mean(d_losses))        
    print 'Epoch {} G loss: {:f}'.format(epoch, np.mean(g_losses))

    def plot_loss_graph(losses, filepath, title):
        smooth_steps = 50 if not args['test'] else 2

        # http://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394#27681394
        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            return (cumsum[N:] - cumsum[:-N]) / N 
        smoothed_losses = running_mean(losses, smooth_steps)
        
        ax = pd.DataFrame({
            'Loss': smoothed_losses, # TODO can you remove the legend?
        }).plot(title=title)        
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Loss")
        fig = ax.get_figure()
        print('Writing {} loss graph to {}'.format(title, filepath))    
        fig.savefig(filepath)

    d_real_filepath = os.path.join(args['save_path'],'{}_{}_e{}_loss_d_real.png'.format(args['name'], timestamp, epoch))
    d_real_title = 'Real image discriminator loss through epoch {}'.format(epoch+1)
    plot_loss_graph(d_real_losses, d_real_filepath, d_real_title)
    d_gen_filepath = os.path.join(args['save_path'],'{}_{}_e{}_loss_d_gen.png'.format(args['name'], timestamp, epoch))
    d_gen_title = 'Generated image discriminator loss through epoch {}'.format(epoch+1)
    plot_loss_graph(d_gen_losses, d_gen_filepath, d_gen_title)
    d_filepath = os.path.join(args['save_path'],'{}_{}_e{}_loss_d.png'.format(args['name'], timestamp, epoch))
    d_title = 'Net discriminator loss through epoch {}'.format(epoch+1)
    plot_loss_graph(d_losses, d_filepath, d_title)
    g_filepath = os.path.join(args['save_path'],'{}_{}_e{}_loss_g.png'.format(args['name'], timestamp, epoch))
    g_title = 'Generator loss through epoch {}'.format(epoch+1)
    plot_loss_graph(g_losses, g_filepath, g_title)


def save_samples(G, num_to_generate, batch_size, save_path, args):
    print 'generating {} samples...'.format(num_to_generate)

    cropped_width = args['cropped_width']
    
    gen_particle_types, gen_input_energies, gen_noise, gen_labels = generator_data_pull(num_to_generate, args)
    images = G.predict([gen_particle_types, gen_input_energies, gen_noise], batch_size=batch_size)
    
    images = np.squeeze(images) # get rid of the extra dim added by expand_dims
    assert images.shape == (num_to_generate, cropped_width*cropped_width, data_loader.DATA_DIM[2]), images.shape
    images = data_loader.roll_ecals(images) # make it back into cubes
    stddev = STDDEV_MAP[cropped_width, args['log_data'], args['averaged_data']]
    images = data_loader.denormalize_ecals(images, TRAIN_MEAN, stddev)
    if args['log_data']:
        images = data_loader.unlog_ecals(images)
    if cropped_width < data_loader.DATA_DIM[0]:
        images = data_loader.untruncate_ecals(images)
    assert images.shape == (num_to_generate, data_loader.DATA_DIM[0], data_loader.DATA_DIM[1], data_loader.DATA_DIM[2]), images.shape
    
    # get rid of the extra dim added by expand_dims
    gen_particle_types = np.squeeze(gen_particle_types)
    gen_input_energies = np.squeeze(gen_input_energies)
    
    targets = np.vstack((gen_particle_types, gen_input_energies)).T
    output_tuple = (images, targets)
    
    with open(save_path, 'wb') as f:
        print('Writing output tuple to {}'.format(save_path))
        pickle.dump(output_tuple, f, protocol=pickle.HIGHEST_PROTOCOL)   


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


def generator_data_pull(num_to_generate, args):
    gen_particle_types = np.random.randint(0, 2, size=(num_to_generate, 1))
    # sample momentum from [10,500)
    gen_input_energies = 490 * np.random.random((num_to_generate, 1)) + 10
    if args['use_gauss']:
        flat_height = args['cropped_width']*args['cropped_width']
        gen_noise = np.expand_dims(np.stack(
            [gauss_z(input_energy[0], flat_height, data_loader.DATA_DIM[2])
             for input_energy in gen_input_energies], axis=0), axis=1)
    else:
        gen_noise = np.expand_dims(np.stack(
            [np.random.multivariate_normal(np.zeros(args['latent_size']), np.eye(args['latent_size']))
             for b in range(num_to_generate)], axis=0), axis=1)
    gen_labels = np.ones(num_to_generate)
    return gen_particle_types, gen_input_energies, gen_noise, gen_labels


def optimizer():
    learning_rate = 5E-5  # From the wasserstein paper
    return RMSprop(lr=learning_rate)


def train_epoch(data_generator, G, D, GAN, epoch, args):
    seen_batches = 0
    
    d_gen_losses = []
    d_real_losses = []    
    g_losses = []

    last_epoch = False
    
    while not last_epoch:
        seen_batches += 1
        
        # Train discriminator 'update_ratio' times for every one time the generator is trained.
        d_update_step = 0        
        for real_images, real_particle_types, real_input_energies in data_generator:
            d_update_step += 1

            # The batch size is set in the data generator, but the number of examples in a file won't
            # divide evenly by the batch size, so the actual batch size may be smaller.
            batch_size = real_images.shape[0]

            # Train the discriminator on real images
            # add an extra dimension to real data to fit model
            real_particle_types = np.expand_dims(real_particle_types, axis=1)
            real_input_energies = np.expand_dims(real_input_energies, axis=1)
            # NOTE: In wasserstein, this the real images get negative lavels in the discriminator
            # (so when the cost function is minimized, it maximizes instead).        
            real_labels = -np.ones(batch_size)
            # Train the discriminator on real images
            d_real_loss = D.train_on_batch([real_images, real_particle_types, real_input_energies], real_labels)
            d_real_losses.append(d_real_loss)

            # Train the discriminator on generated images
            # NOTE: In wasserstein, when training the discriminator, the generated images get a positive
            # label, so the cost function minimizes them.            
            gen_particle_types, gen_input_energies, gen_noise, gen_labels = generator_data_pull(batch_size, args)
            gen_images = G.predict([gen_particle_types, gen_input_energies, gen_noise])
            d_gen_loss = D.train_on_batch([gen_images, gen_particle_types, gen_input_energies], gen_labels)
            d_gen_losses.append(d_gen_loss)

            # Clip the discriminator's weights
            clip_weights(D)
            
            if d_update_step >= args['update_ratio']:
                break
        else:
            # The else clause in a python for-loop executes when the loop finishes normally, and
            # doesn't exit due to break.
            # Here, that means we've run out of data for the epoch, and should stop training.
            last_epoch = True

        # Train generator
        batch_size = args['batch_size']
        gen_particle_types, gen_input_energies, gen_noise, gen_labels = generator_data_pull(batch_size, args)
        # Freeze D when training the GAN.
        # Also, when training the generator in wasserstein, the generates images get negative labels, so
        # that minimizing the cost function causes a maximization.
        make_trainable(D, False)
        g_loss = GAN.train_on_batch([gen_particle_types, gen_input_energies, gen_noise], -gen_labels)
        make_trainable(D, True)
        g_losses.append(g_loss)
            
    return d_real_losses, d_gen_losses, g_losses 


def train(G, D, GAN, args):
    if not args['save_path']:
        print 'Save path not set! Exiting...'
        sys.exit(1)
    if not os.path.exists(args['save_path']):
        print 'Save path {} not found. Creating directory...'.format(args['save_path'])
        os.makedirs(args['save_path'])

    timestamp = '{:%Y%m%d%H%M}'.format(datetime.datetime.now())

    d_real_losses = []
    d_gen_losses = []
    g_losses = []

    # epoch loop
    for epoch in range(args['n_epochs']):
        print 'starting epoch {}'.format(epoch)
        start_time = time.time()
        
        train_generator = data_generator(args, partition=0)
        
        d_epoch_real_loss, d_epoch_gen_loss, g_epoch_loss = train_epoch(train_generator, G, D, GAN, epoch, args)

        epoch_hours = (time.time() - start_time) / 3600
        print 'Epoch {} finished in {:f} hours'.format(epoch, epoch_hours)

        d_real_losses.extend(d_epoch_real_loss)
        d_gen_losses.extend(d_epoch_gen_loss)
        g_losses.extend(g_epoch_loss)
        
        plot_losses(args, timestamp, epoch, d_real_losses, d_gen_losses, g_losses)
        
        samples_save_path = os.path.join(args['save_path'], '{}_{}_e{}_samples.pkl'.format(args['name'], timestamp, epoch))
        num_to_generate = args['num_samples']
        batch_size = args['batch_size']
        save_samples(G, num_to_generate, batch_size, samples_save_path, args)

    print 'Training complete'
