# functions for evaluating generated image quality against real images. 

# NOTE!
# matplotlib doesn't like virtualenv.
# execute this in bash:

# function frameworkpython {
#   if [[ ! -z "$VIRTUAL_ENV" ]]; then PYTHONHOME=$VIRTUAL_ENV /usr/bin/python "$@";
#   else /usr/bin/python "$@";
#   fi;
# }
# then use 'frameworkpython eval.py' instead of just 'python eval.py'

import collections
import numpy as np
import os
import sys
import cPickle as pkl

import data_loader
from plots import PlotSuite


def load_real_data(batch_iter, batch_limit):
    '''
    joins batch iterations into a complete set
    '''
    num_batches = 0
    all_ecals = []
    all_targets = []
    for ecals, targets in batch_iter:
        if num_batches >= batch_limit:
            break
        all_ecals.append(ecals)
        all_targets.append(targets)
        num_batches += 1
    return np.concatenate(all_ecals), np.concatenate(all_targets)

def load_from_pickle(fname, limit):
    '''
    load generated data from pickle file9
    '''
    with open(fname,'r') as f:
        samples=pkl.load(f)
    ecals,targets = samples
    return ecals[:limit],targets[:limit]

def prepare_energy_data_no_iter(ecals,targets,dummy=0):
    '''
    Returns two EnergyData objects, one for each particle type.
    '''
    output_tuple = (EnergyData(type=0, input=[], output=[], dummy=[]),
                    EnergyData(type=1, input=[], output=[], dummy=[]))
        
    assert ecals.shape[0] == targets.shape[0]
    for i in range(ecals.shape[0]):
        input_energy = targets[i][1]
        ecals_slice = ecals[i, :, :, :]
        output_energy = np.sum(ecals_slice, axis=None)
        particle = int(targets[i][0]) # either zero or one
        output_tuple[particle].input.append(input_energy)
        output_tuple[particle].output.append(output_energy)
        output_tuple[particle].dummy.append(dummy)
    return output_tuple

def merge_real_gen(real,gen):
    '''
    merge real and generated EnergyData
    assumes that gen type matches real type
    '''
    merged = EnergyData(type=real.type, 
                        input=list(real.input) + list(gen.input),
                        output=list(real.output) + list(gen.output),
                        dummy=list(real.dummy) + list(gen.dummy))
    return merged


# EnergyData: namedtuple representing input energy -> sum(output energy). Should include
# both real and generated data.
# input: numpy array of floats representing input energy values
# output: numpy array of floats representing output energy values
# dummy: numpy array of zeros and ones, one for each data point. A one indicates real data, a
#        zero indicates generated data.
# type: The particle type (int, zero or 1)
EnergyData = collections.namedtuple('EnergyData', ['type', 'input', 'output', 'dummy'])


def prepare_energy_data(data_batch_iter,dummy=0):
    """Returns two EnergyData objects, one for each particle type."""
    output_tuple = (EnergyData(type=0, input=[], output=[], dummy=[]),
                    EnergyData(type=1, input=[], output=[], dummy=[]))
        
    for ecals, targets in data_batch_iter:
        assert ecals.shape[0] == targets.shape[0]
        for i in range(ecals.shape[0]):
            input_energy = targets[i][1]
            ecals_slice = ecals[i, :, :, :]
            output_energy = np.sum(ecals_slice, axis=None)
            particle = int(targets[i][0]) # either zero or one
            output_tuple[particle].input.append(input_energy)
            output_tuple[particle].output.append(output_energy)
            output_tuple[particle].dummy.append(dummy)
    return output_tuple


# TODO: linearly interpolate between the maximum energy values found in the real data instead.
def add_gen_data(energy_data, session, samples_op, y_samples_ph, z_samples_ph, num_randos):
    """Returns a copy of energy_data with generated data added. 

    The generated data uses the same input energies and particles that the real data has.
    """
    print('[EVAL] Generating {} data points for particle type {} for plotting'.format(
            len(energy_data.input), energy_data.type))
    new_energy_data = EnergyData(type=energy_data.type, input=list(energy_data.input),
                                 output=list(energy_data.output), dummy=list(energy_data.dummy))
    for i in range(len(energy_data.input)):
        input_energy = energy_data.input[i]
        gen_output_energy = np.sum(session.run(
            samples_op, {z_samples_ph: [np.random.uniform(-1, 1, size=num_randos)],
                         y_samples_ph: [[energy_data.type, input_energy]]}), axis=None)
        new_energy_data.input.append(input_energy)
        new_energy_data.output.append(gen_output_energy)
        new_energy_data.dummy.append(1)
    return new_energy_data

    
def get_exp_name(data_filepath):
    '''
    add experiment id to output
    '''
    filename = os.path.split(data_filepath)[-1]
    idx = filename.find('.')
    if idx != -1:
        return filename[:idx]
    return filename


def split_by_type_and_energy(ecals,targets,cuts):
    '''
    split up the data into groups across two dimensions:
    first split by particle type,
    then bin into low, medium, high (or more) bins of input energy
    '''
    datasets = [[],[]]
    cuts = [-np.inf]+cuts+[np.max(targets)]
    for p_type in [0,1]:
        for i in range(1,len(cuts)):
            requirements = [targets[:,0]==p_type,targets[:,1]>cuts[i-1],targets[:,1]<=cuts[i]]
            ids = np.where(np.all(requirements, axis=0))[0]
            datasets[p_type].append((ecals[ids],targets[ids]))
    return datasets


def find_cuts(targets,n_sets):
    '''
    returns list (size n_sets - 1) of cut points to split data into
    low, med, high input energy (or more groups)
    '''
    cuts = []
    for i in range(1,n_sets):
        cut = np.percentile(targets[:,1],100.*i/n_sets)
        cuts.append(cut)
    return cuts


def main(gen_fpath, real_data_source, output_dir, real_sample_id, limit):
    '''
    Produce charts dashboard of charts
    args:
        gen_fpath: path to generated images
        real_data_source: one of the following
            1. 'dl_test': use dataloader to load test data
            2. 'dl_full': use dataloader to load full dataset
            3. path to pickle file
        output_dir: where you want to save your charts
        real_sample_id: id of a single cube to compare real vs. gen
    '''
    N_SETS = 3 #Number of input energy groups (low, med, high)
    exp_name = get_exp_name(gen_fpath)

    print "Loading data..."
    #load real data    
    if real_data_source=='dl_test':
        batch_size=10
        dl = data_loader.DataLoader([100],test=True)
        batch_iter = dl.batch_iter(partition_index=0,batch_size=batch_size,num_epochs=1)
        real_data = load_real_data(batch_iter, limit / batch_size)
    elif real_data_source=='dl_full':
        batch_size = 100
        dl = data_loader.DataLoader([80,15,5],test=False)
        batch_iter = dl.batch_iter(partition_index=2,batch_size=batch_size,num_epochs=1)
        real_data = load_real_data(batch_iter, limit / batch_size)
    else:
        real_data = load_from_pickle(real_data_source, limit)
        real_sample_id = 5
    
    #load generated data from pickle file
    gen_data = load_from_pickle(gen_fpath, limit)
    
    gen_ecals,gen_targets = gen_data
    real_ecals, real_targets = real_data

    #format data for charts
    print "Formatting data..."
    p0_real, p1_real = prepare_energy_data_no_iter(real_ecals,real_targets,dummy=0)
    p0_gen, p1_gen = prepare_energy_data_no_iter(gen_ecals,gen_targets,dummy=1)
    p0_data = merge_real_gen(p0_real,p0_gen)
    p1_data = merge_real_gen(p1_real,p1_gen)

    #plot averaged charts
    print "Producing charts for all particle types..."
    plots = PlotSuite(real_data, gen_data, exp_name, os.path.join(output_dir,'avg'))
    plots.plot_energy_scatterplot(p0_data, p1_data)
    plots.plot_energy_histogram(p0_data, p1_data)
    plots.plot_chart_suite(sample_id=None)
    
    print "Splitting data into type/energy sets..."
    cuts = find_cuts(real_targets,N_SETS)
    real_sets = split_by_type_and_energy(real_ecals,real_targets,cuts)
    gen_sets = split_by_type_and_energy(gen_ecals,gen_targets,cuts)
    
    print "Producing charts for particle type/energy sets..."
    for p_type in [0,1]:
        for energy_level in range(N_SETS):
            real = real_sets[p_type][energy_level]
            gen = gen_sets[p_type][energy_level]
            set_name = 'p'+str(p_type)+'e'+str(energy_level)
            print set_name
            plots = PlotSuite(real, gen, exp_name+'-'+set_name, os.path.join(output_dir,set_name))
            plots.plot_chart_suite(sample_id=0)
            plots.plot_chart_suite(sample_id=None)

# Testing code
if __name__ == '__main__':
    if len(sys.argv) == 4:
        gen_fpath = sys.argv[1]
        real_data_source = sys.argv[2]
        output_dir = sys.argv[3]
        real_sample_id = 5
        limit = 10000
    else:
        gen_fpath = '../exp/israel/rnn_gen.pkl'
        exp_name = 'rnn'
        output_dir = '../exp/rnn_test'
        real_data_source = '../exp/israel/rnn_true.pkl'
        real_sample_id = 5
        limit = 10000
        print "-"*50
        print "Usage:\n\tpython eval.py generated_fpath real_data_source output_dir"
        print "Ex:\n\tpython eval.py ../exp/alex/spiral_gan_test_final_samples.pkl dl_test ../exp/test"
        print "Args:"
        print "\tgen_fpath: path to pickled generator file"
        print "\toutput_dir: where to put the graphs"
        print "\treal_data_source: dl_test, dl_full, or path/to/pickle/file"
        print "\t\tdl_test: use dataloader to load test version of real data"
        print "\t\tdl_full: use dataloader to load full version of real data"
        print "\treal_sample_id: index of the calorimeter/experiment to view on certain charts"
        print "\nAssuming defaults:"
        print "\tgenerated_fpath =",gen_fpath
        print "\toutput_dir =",output_dir
        print "\treal_data_source =",real_data_source
        print "\treal_sample_id =",real_sample_id
        print "\tlimit =",limit
        print "-"*50
    main(gen_fpath, real_data_source, output_dir, real_sample_id, limit)
