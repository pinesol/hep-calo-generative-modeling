# NOTE!
# matplotlib doesn't like virtualenv.
# execute this in bash:

# function frameworkpython {
#   if [[ ! -z "$VIRTUAL_ENV" ]]; then PYTHONHOME=$VIRTUAL_ENV /usr/bin/python "$@";
#   else /usr/bin/python "$@";
#   fi;
# }
# then use 'frameworkpython eval.py' instead of just 'python eval.py'

import data_loader

import collections
import matplotlib
# This next line is required in order for plt.plot() to run on a remote server.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys


# EnergyData: namedtuple representing input energy -> sum(output energy). Should include
# both real and GAN-generated data.
# input: numpy array of floats representing input energy values
# output: numpy array of floats representing output energy values
# dummy: numpy array of zeros and ones, one for each data point. A one indicates real data, a
#        zero indicates gan data.
# type: The particle type (int, zero or 1)
EnergyData = collections.namedtuple('EnergyData', ['type', 'input', 'output', 'dummy'])

RegressionCoeffs = collections.namedtuple('RegressionCoeffs', ['slope', 'bias'])

# Takes a EnergyData object, finds two 1d linear regressions,
# Makes a plot and saves it to disk, saves the coefficients to disk in a text file.
def energy_regression(energy_data, output_dir):
    input_energy_vals = np.array(energy_data.input)
    output_energy_vals = np.array(energy_data.output)
    dummy_vals = np.array(energy_data.dummy)
    
    # Parameters
    learning_rate = 1.0
    batch_size = 512
    training_epochs = 1000
    display_epoch = 100

    X = tf.placeholder(tf.float32, name='input_energy')
    y = tf.placeholder(tf.float32, name='total_output_energy')
    G = tf.placeholder(tf.float32, name='gan_dummy')

    W = tf.Variable(np.random.randn(), name="weight")
    b = tf.Variable(np.random.randn(), name="bias")
    Wg = tf.Variable(np.random.randn(), name="gan_weight")
    bg = tf.Variable(np.random.randn(), name="gan_bias")

    pred = tf.mul(X, W) + b + tf.mul(tf.mul(X, G), Wg) + tf.mul(bg, G)
    cost = tf.reduce_mean(tf.square(pred-y))

    # NOTE: this blows up with a regular gradient descent optimizer, interestingly
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize variable op
    init = tf.initialize_all_variables()

    # Run the model
    print 'Training linear regression...'
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            for i in range(0, len(input_energy_vals), batch_size):
                input = input_energy_vals[i:(i+batch_size)]
                output = output_energy_vals[i:(i+batch_size)]
                dummy = dummy_vals[i:(i+batch_size)]
                sess.run(optimizer, feed_dict={X: input, y: output, G: dummy})

            if epoch % display_epoch == 0:
                # Evaluate the linear regression on 5x the batch size. Not
                # worrying about train/test split here.
                cost_val = sess.run(cost,
                                    feed_dict={X: input_energy_vals[:5*batch_size],
                                               y: output_energy_vals[:5*batch_size],
                                               G: dummy_vals[:5*batch_size]})
                print "Epoch: {0:}, cost={1:.4f}".format(epoch, cost_val)
                
        print 'Optimization complete'
        cost_val, W_val, b_val, Wg_val, bg_val = sess.run(
            [cost, W, b, Wg, bg],
            feed_dict={X: input_energy_vals, y: output_energy_vals, G: dummy_vals})
        print "Final values: cost={}, W={}, b={}, Wg={}, bg={}".format(
            cost_val, W_val, b_val, Wg_val, bg_val)

        real_coeffs = RegressionCoeffs(slope=W_val, bias=b_val)
        gan_coeffs = RegressionCoeffs(slope=W_val + Wg_val, bias=b_val + bg_val)
        plot_energy_data(energy_data, output_dir, real_coeffs, gan_coeffs)
        
        # Save the regression coefficients
        filename = 'energy_linreg_p' + str(energy_data.type) + '_coeffs.txt'
        with open(os.path.join(output_dir, filename), 'w') as f:
            real_slope = W_val
            real_bias = b_val
            gan_slope = W_val + Wg_val
            gan_bias = b_val + bg_val
            slope_diff = real_slope - gan_slope
            bias_diff = real_bias - gan_bias
            line0 = '---Particle type {} regression coefficients---'.format(energy_data.type)
            line1 = 'Real data regression: slope={0:.4f}, bias={1:.4f}'.format(real_slope, real_bias)
            line2 = 'GAN data regression: slope={0:.4f}, bias={1:.4f}'.format(gan_slope, gan_bias)
            line3 = 'Difference: slope={0:.4f}, bias={1:.4f}'.format(slope_diff, bias_diff)
            f.write('\n'.join([line0, line1, line2, line3]))


# Returns two EnergyData objects, one for each particle type.
def prepare_energy_data(data_batch_iter):
    CALORIMETER_SLICE_INDEX = 12

    output_tuple = (EnergyData(type=0, input=[], output=[], dummy=[]),
                    EnergyData(type=1, input=[], output=[], dummy=[]))
        
    for ecals, targets in data_batch_iter:
        assert ecals.shape[0] == targets.shape[0]
        for i in range(ecals.shape[0]):
            input_energy = targets[i][1]
            ecals_slice = ecals[i, :, :, CALORIMETER_SLICE_INDEX]
            output_energy = np.sum(ecals_slice, axis=None)
            particle = int(targets[i][0]) # either zero or one
            output_tuple[particle].input.append(input_energy)
            output_tuple[particle].output.append(output_energy)
            output_tuple[particle].dummy.append(0)
    return output_tuple


# Returns a copy of energy_data with GAN data added.
def add_gan_data(energy_data, session, samples_op,
                 y_samples_ph, z_samples_ph, num_randos):
    print('Generating {} data points from GAN for particle type {} for plotting'.format(
            len(energy_data.input), energy_data.type))
    new_energy_data = EnergyData(type=energy_data.type, input=list(energy_data.input),
                                 output=list(energy_data.output), dummy=list(energy_data.dummy))
    for i in range(len(energy_data.input)):
        input_energy = energy_data.input[i]
        gan_output_energy = np.sum(session.run(
            samples_op, {z_samples_ph: [np.random.uniform(-1, 1, size=num_randos)],
                         y_samples_ph: [[energy_data.type, input_energy]]}), axis=None)
        new_energy_data.input.append(input_energy)
        new_energy_data.output.append(gan_output_energy)
        new_energy_data.dummy.append(1)
    return new_energy_data


# Given energy data, saves a plot to disk. Only includes GAN data if it's present.
# step should be the current training step number
def plot_energy_data(energy_data, output_dir, real_coeffs=None, gan_coeffs=None, step=None):
    print('Plotting energy data...')
    input_energy_vals = np.array(energy_data.input)
    output_energy_vals = np.array(energy_data.output)
    dummy_vals = np.array(energy_data.dummy)
    
    real_inputs = input_energy_vals[(dummy_vals == 0)]
    real_outputs = output_energy_vals[(dummy_vals == 0)]    
    gan_inputs = input_energy_vals[(dummy_vals == 1)]
    gan_outputs = output_energy_vals[(dummy_vals == 1)]
    
    plt.plot(real_inputs, real_outputs, 'ro', label='Real data',
             alpha=0.4, markersize=3, markeredgewidth=0.0)
    
    if len(gan_inputs) > 0:
        plt.plot(gan_inputs, gan_outputs, 'go', label='GAN data',
                 alpha=0.4, markersize=3, markeredgewidth=0.0)
        step_str = ', step {}'.format(step) if step else ''
        plt.title('Particle {} Energy, with GAN{}'.format(energy_data.type, step_str))
        plt.legend(loc='upper left')
    else:
        step_str = ', step {}'.format(step) if step else ''
        plt.title('Particle {} Real and GAN Energy{}'.format(energy_data.type, step_str))

    if real_coeffs:
        real_regression_vals = real_coeffs.slope * real_inputs + real_coeffs.bias
        plt.plot(real_inputs, real_regression_vals, 'r', label='Real line')
    if len(gan_inputs) > 0 and gan_coeffs:
        gan_regression_vals = gan_coeffs.slope * gan_inputs + gan_coeffs.bias
        plt.plot(gan_inputs, gan_regression_vals, 'g', label='GAN line')

    # Determine file name
    step_str = '_step_{}'.format(step) if step else ''
    gan_str = '_w_gan' if len(gan_inputs) > 0 else ''
    reg_str = '_linreg' if real_coeffs else ''
    filename = 'energy{}{}{}_p{}_plot.png'.format(step_str, gan_str, reg_str, energy_data.type)
        
    plt.xlabel('Input Energy')
    plt.ylabel('Total Output Energy')
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path)

    #also save to latest
    latestname = 'energy_latest_{}{}_p{}_plot.png'.format(gan_str, reg_str, energy_data.type)
    latest_path = os.path.join(output_dir, latestname)
    plt.savefig(latest_path)

    plt.clf() # Clears matplotlib
    print 'Saved energy data plot to {}.'.format(full_path)


# Create histograms for both real and GAN energy data, where the value to plot
# is the ratio of output energy to the input energy.
def create_histogram(energy_data, output_dir, step=None):
    print('Creating energy data histogram...')
    input_energy_vals = np.array(energy_data.input)
    output_energy_vals = np.array(energy_data.output)
    dummy_vals = np.array(energy_data.dummy)
    
    real_inputs = input_energy_vals[(dummy_vals == 0)]
    real_outputs = output_energy_vals[(dummy_vals == 0)]    
    gan_inputs = input_energy_vals[(dummy_vals == 1)]
    gan_outputs = output_energy_vals[(dummy_vals == 1)]

    real_ratios = np.array([1.0 * output / input
                            for input, output in zip(real_inputs, real_outputs)])
    gan_ratios = np.array([1.0 * output / input
                           for input, output in zip(gan_inputs, gan_outputs)])

    fig = plt.figure(1)
    step_str = ', step {}'.format(step) if step else ''
    st = fig.suptitle('Particle {} Output-to-Input Energy Ratio{}'.format(energy_data.type, step_str),
                      fontsize="x-large")
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(real_ratios, bins='auto', color='red', alpha=0.7)
    ax1.set_title('Real data')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
    legend = '$\mu=%.2f$\n$\sigma=%.2f$'%(real_ratios.mean(), real_ratios.std())
    ax1.text(0.95, 0.95, legend, transform=ax1.transAxes, horizontalalignment='right',
             verticalalignment='top')

        
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(gan_ratios, bins='auto', color='green', alpha=0.7)
    ax2.set_title('GAN data')
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)
    legend = '$\mu=%.2f$\n$\sigma=%.2f$' % (gan_ratios.mean(), gan_ratios.std())
    ax2.text(0.95, 0.95, legend, transform=ax2.transAxes, horizontalalignment='right',
             verticalalignment='top')
        
    ax3 = fig.add_subplot(2, 1, 2)
    # when plotting the two histograms together, throw out the outliers the GAN makes sometimes.
    # 4 standard deviations should be enough to get rid of the crazy ones.
    trunc_gan_ratios = gan_ratios[abs(gan_ratios - np.mean(gan_ratios)) < 4 * np.std(gan_ratios)]
    hist_range = min(min(real_ratios), min(trunc_gan_ratios)), max(max(real_ratios), max(trunc_gan_ratios))
    ax3.hist([real_ratios, trunc_gan_ratios], bins='auto', range=hist_range, color=['red', 'green'], alpha=0.8)
    ax3.legend(['Real', 'GAN'], loc='upper right')
    ax3.set_title('Comparison')
    
    fig.tight_layout()

    # shift subplots down:
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)

    step_str = '_step{}'.format(step) if step else ''
    filename = 'energy{}_p{}_hist.png'.format(step_str, energy_data.type)
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path)

    #also save to latest
    latestname = 'energy_latest_p{}_hist.png'.format(energy_data.type)
    latest_path = os.path.join(output_dir, latestname)
    plt.savefig(latest_path)

    plt.clf() # Clears matplotlib
    print 'Saved energy data histogram to {}.'.format(full_path)
    
    

def LocalTest():
    data_loader_obj = data_loader.DataLoader([100], test=True)
    batch_iter = data_loader_obj.train_batch_iter(batch_size=100, num_epochs=1)

    # Fake variables for testing
    session = None
    samples_op = None
    output_dir = './exp/eval_local_test'
    y_samples_ph = 'y_ph'
    z_samples_ph = 'z_ph'
    num_randos = 1
    class FakeSession:
        def run(self, dummy_op, feed_dict):
                yield 1.5 * feed_dict['y_ph'][0][1] + 3 + 10 * feed_dict['z_ph'][0]
    session = FakeSession()    

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    p0_data, p1_data = prepare_energy_data(batch_iter)
    plot_energy_data(p0_data, output_dir)
    plot_energy_data(p1_data, output_dir)
    p0_data = add_gan_data(p0_data, session, samples_op, y_samples_ph, z_samples_ph, num_randos)
    p1_data = add_gan_data(p1_data, session, samples_op, y_samples_ph, z_samples_ph, num_randos)
    plot_energy_data(p0_data, output_dir, step=1)
    plot_energy_data(p1_data, output_dir, step=1)    
    create_histogram(p0_data, output_dir, step=1)
    create_histogram(p1_data, output_dir, step=1)
#    energy_regression(p0_data, output_dir)
#    energy_regression(p1_data, output_dir)

    
def MercerTest():
    data_loader_obj = data_loader.DataLoader([10, 90], test=False)
    batch_iter = data_loader_obj.train_batch_iter(batch_size=100, num_epochs=1)

    # Fake variables for testing
    session = None
    samples_op = None
    output_dir = './exp/eval_mercer_test'
    y_samples_ph = 'y_ph'
    z_samples_ph = 'z_ph'
    num_randos = 1
    class FakeSession:
        def run(self, dummy_op, feed_dict):
                yield 1.5 * feed_dict['y_ph'][0][1] + 3 + 10 * feed_dict['z_ph'][0]
    session = FakeSession()    

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)        

    p0_data, p1_data = prepare_energy_data(batch_iter)
    plot_energy_data(p0_data, output_dir)
    plot_energy_data(p1_data, output_dir)
    p0_data = add_gan_data(p0_data, session, samples_op, y_samples_ph, z_samples_ph, num_randos)
    p1_data = add_gan_data(p1_data, session, samples_op, y_samples_ph, z_samples_ph, num_randos)
    plot_energy_data(p0_data, output_dir, step=1)
    plot_energy_data(p1_data, output_dir, step=1)    
    create_histogram(p0_data, output_dir, step=1)
    create_histogram(p1_data, output_dir, step=1)
#    energy_regression(p0_data, output_dir)
#    energy_regression(p1_data, output_dir)
        
    

# Testing code
if __name__ == '__main__':  
    if len(sys.argv) >= 2 and sys.argv[1] == 'mercer':
        'Running mercer test'
        MercerTest()
    else:
        'Running local test'
        LocalTest()
