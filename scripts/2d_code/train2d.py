import data_loader
import eval

import datetime
import numpy as np
import cPickle as pkl
import tensorflow as tf
import os
from scipy.misc import imsave
from tensorflow.contrib.metrics import accuracy


WIDTH = 20  # Image width (X,Y axes)
N_SAMPLE_BINS= 10 #Right now we have 5 different momentum bins (100..500) x two particle types. 


def accuracy_(score, labels):
    '''
    assumes score will be logits (ie unbounded real numbers) 
    '''
    pred = tf.cast(tf.round(tf.sigmoid(score)), 'int8')
    _labels = tf.cast(labels, 'int8')
    return accuracy(pred, _labels)

def build_train_2d(config, generator, discriminator):
    """Builds a 2D GAN graph, trains it, and saves it to a file, along with output samples.

    Args:
      config: An object with a bunch of config files. TODO standardize the fields!
      generator: A function that takes a 1d tensor of random values, and returns a network that
        generates a 1d tensor or size WIDTH*WIDTH
      discriminator: A function that takes a 1d tensor of size WIDTH*WIDTH representing a 2d slice
        of 3d data, and returns a scalar tensor between zero and one. One for real and zero for
        fake (I think?).
    """  
    config.sample_dir = os.path.join(config.train_dir, 'samples')
    os.makedirs(config.sample_dir)
    checkpoint_path = os.path.join(config.train_dir, 'model.ckpt')
    
    print("Setting up graph")
    # One slice will be 24x24.  None implies variable batch size.    
    x = tf.placeholder(tf.float32, [None, WIDTH * WIDTH], name='x_input')
    # y: particle type and momentum
    y = tf.placeholder(tf.float32, [None, 2], name='y_input')
    # random dist to sample from (config.z = 100)
    z = tf.placeholder(tf.float32, [None, config.z], name='z_input')

    #y inputs to generate samples
    y_samples = tf.placeholder(tf.float32, [None, 2])

    #random numbers to generate samples
    z_samples = tf.placeholder(tf.float32, [None, config.z], name='z_samples')
    
    # This builds the graphs.  Shared variables: first call creates the network, 
    # second call to the net will share the same variables and weights
    with tf.variable_scope('generator'):
        G = generator(z,y)
    # These samples are implicitly reusing the same Variables as G since they're in the same scope.
    # They're only used for showing examples later.
    with tf.variable_scope('generator', reuse=True):
        samples = generator(z_samples,y_samples)
    with tf.variable_scope('discriminator'):
        D_x = discriminator(x,y)
    with tf.variable_scope('discriminator', reuse=True):
        D_g = discriminator(G,y)

    # cost and accuracy
    # computes cross entropy cost of D_x, and takes mean across the batch.
    # 1 means "real output", 0 means generated.
    # Cost here means how far from the correct guess for real vs. fake.
    # "logits": the result of a matrix multiplication w/o a non-linearity.
    # "ones_like": creates tensor of ones in the shape of the given tensor.

    # Cost of the discriminator on the real values.
    real_labels = tf.ones_like(D_x)
    d_cost_x = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_x, real_labels))
    d_accuracy_x = accuracy_(D_x, real_labels)

    # Cost of the discriminator on the generated values. Notice it uses zeros_like here.
    generated_labels = tf.zeros_like(D_g)
    d_cost_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_g, generated_labels))
    d_accuracy_g = accuracy_(D_g, generated_labels)
    d_cost = d_cost_x + d_cost_g
    d_accuracy = (d_accuracy_x + d_accuracy_g) / 2.0
    # Cost function of the generator. Compares D_g to a tensor of ones, since it's trying
    # to get the discriminator to output ones.
    g_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_g, tf.ones_like(D_g)))
    print("Defining summary objects...")
    # summary
    # log the values every batch or so
    
    tf.scalar_summary('d_cost_x', d_cost_x)
    tf.scalar_summary('d_cost_g', d_cost_g)
    tf.scalar_summary('d_cost', d_cost)
    tf.scalar_summary('g_cost', g_cost)
    tf.scalar_summary('d_accuracy_x', d_accuracy_x)
    tf.scalar_summary('d_accuracy_g', d_accuracy_g)
    tf.scalar_summary('d_accuracy', d_accuracy)
    
    # Collect list of all the discrim variables
    d_params = filter(lambda x: x.name.startswith('discriminator'),
                      tf.trainable_variables())
    # Collect list of all the generator variables
    g_params = filter(lambda x: x.name.startswith('generator'),
                      tf.trainable_variables())    
    
    # train ops
    # 'step':= mini batch training step.
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(config.lr, global_step,
                                           config.decay_steps,
                                           config.decay_factor, staircase=True)
    train_op_d = tf.train.AdamOptimizer(learning_rate).minimize(d_cost,
                                                        var_list=d_params,
                                                        global_step=global_step)
    # Note: I removed the second global step update, so there is only one 
    # step increment per batch. 
    train_op_g = tf.train.AdamOptimizer(learning_rate).minimize(g_cost,
                                                        var_list=g_params,
                                                        global_step=None)  
    
    tf.scalar_summary('learning_rate', learning_rate)
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver(tf.all_variables())

    init_op = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter(config.train_dir, sess.graph)
        print("Running initialization operation...")
        sess.run(init_op)

        current_step = 0
        
        data_loader_obj = data_loader.DataLoader(config.data_partition, config.test)
        
        # Plot input_energy -> sum(output_energy) data
        print('Loading validation data...')
        energy_eval_iter = data_loader_obj.batch_iter(partition_index=1, batch_size=config.batch_size,
                                                      num_epochs=1)
        p0_energy_data, p1_energy_data = eval.prepare_energy_data(energy_eval_iter)
        eval.plot_energy_data(p0_energy_data, config.train_dir)
        eval.plot_energy_data(p1_energy_data, config.train_dir)

        # Training
        print('Training beginning...')
        train_batch_iter = data_loader_obj.train_batch_iter(batch_size=config.batch_size,
                                                            num_epochs=config.max_epochs)
        for ecals, targets in train_batch_iter:
            # slice out a single Z slice, since this is the 2d GAN.
            train_x = ecals[:,:,:,config.slice_idx]
            train_x = np.reshape(train_x,(-1,WIDTH*WIDTH))
            train_y = targets
            discrim_z = np.random.uniform(-1, 1, size=(train_y.shape[0], config.z))

            discrim_feed_dict = {x: train_x, y: train_y, z: discrim_z}

            _, d_loss, d_acc_x, d_acc_g = sess.run([train_op_d, d_cost, d_accuracy_x, d_accuracy_g],
                                                   discrim_feed_dict)
            g_losses = []  # Collect generator losses, and then average

            for gen_hit in range(config.generator_hits):
                gen_z = np.random.uniform(-1, 1, size=(train_y.shape[0], config.z))
                gen_feed_dict = {y: train_y, z: gen_z} 
                _, g_loss = sess.run([train_op_g, g_cost], gen_feed_dict)
                g_losses.append(g_loss)

            #get the current step
            current_step = tf.train.global_step(sess, global_step)
            summary_str = sess.run(summary_op, discrim_feed_dict)
            summary_writer.add_summary(summary_str, current_step)
            avg_g_loss = np.mean([g_losses])
            print("d_acc_x: {0:.4f}, d_acc_g: {1:.4f}, d_loss: {2:.4f}, g_loss: {3:.4f}".format(
                d_acc_x, d_acc_g, d_loss, avg_g_loss))

            if current_step % config.save_freq == 0:
                print 'saving the model... at step {}'.format(current_step)
                saver.save(sess, checkpoint_path, global_step=current_step)
                save_samples(sess, samples, y_samples, z_samples, current_step, config)
                # Plot energy data points
                p0_energy_w_gan_data = eval.add_gen_data(p0_energy_data, sess, samples, y_samples, z_samples, config.z)
                p1_energy_w_gan_data = eval.add_gen_data(p1_energy_data, sess, samples, y_samples, z_samples, config.z)
                eval.plot_energy_data(p0_energy_w_gan_data, config.train_dir, step=current_step)
                eval.plot_energy_data(p1_energy_w_gan_data, config.train_dir, step=current_step)
                # Plot histograms
                eval.plot_energy_histogram(p0_energy_w_gan_data, p1_energy_w_gan_data, config.train_dir, step=current_step)

        print 'Training is complete. Saving the model...'
        saver.save(sess, checkpoint_path, global_step=current_step)
        save_samples(sess, samples, y_samples, z_samples, current_step, config)
        
    print 'All done!'


def save_samples(sess, samples, y_samples, z_samples, current_step, config):
    '''
    save samples as numpy arrays
    '''

    #create 10 different sample types: 5 different momentum values and 2 different particle types
    momentum = range(100,600,100)*2
    n_momentums = len(set(momentum))
    particle_type = [0]*5+[1]*5
    n_particles = len(set(particle_type))

    y = np.array(zip(particle_type,momentum)*config.samples_batch_size)
    z = np.random.uniform(-1, 1, size=(N_SAMPLE_BINS*config.samples_batch_size, config.z))
    print 'Saving samples images...' 
    gen_samples = sess.run(samples, {z_samples: z, y_samples: y})

    #reshape samples into grid
    gen_samples = gen_samples.reshape(config.samples_batch_size, n_particles, n_momentums, WIDTH, WIDTH)
    y = y.reshape(config.samples_batch_size,n_particles,n_momentums,n_particles)

    checkpoint_num = current_step / config.save_freq

    #save sample
    fname = os.path.join(config.sample_dir,str(checkpoint_num))+'.pkl'
    pkl.dump( {'gen_samples':gen_samples,'y':y}, open( fname, "wb" ) )

    #save latest sample for easy reference
    latest = os.path.join(config.sample_dir,'latest.pkl')
    pkl.dump( {'gen_samples':gen_samples,'y':y}, open( latest, "wb" ) )
