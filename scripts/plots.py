'''
plots.py

author: Charlie Guthrie

A collection of the plots used in eval.py
'''

import matplotlib
matplotlib.use('Agg') #required in order for plt.plot() to run on a remote server.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import os
from data_loader import unroll_ecals


GEN_COLOR = 'green'
REAL_COLOR = 'red'

def _get_pct_zero(ecal):
    '''
    args:
        ecal: (numpy array) can be a cube, slice, or pixel
    returns:
        sum of the pixel intensities
    '''
    flat_ecal = ecal.flatten()
    zeros = flat_ecal[np.where(flat_ecal == 0.0)]
    return float(len(zeros))/len(flat_ecal)   


def _center_of_mass(ecal_slice):
    '''
    calculate the center of mass of a 2D numpy array
    args:
        ecal_slice: 2D numpy array
    returns:
        (x,y) position of center of mass
    '''
    if np.sum(ecal_slice)>0:
        x = np.arange(ecal_slice.shape[1])
        y = np.arange(ecal_slice.shape[0])
        center_x = np.dot(x,ecal_slice.sum(axis=0))/float(ecal_slice.sum())
        center_y = np.dot(y,ecal_slice.sum(axis=1))/float(ecal_slice.sum())
        return (center_x,center_y)
    else:
        return (0,0)


def _root_mean_squared_error(ecal_slice):
    '''
    a measure of dispersion/spread of energy around the center
    args:
        ecal_slice: numpy array - slice of ECAL
    returns:
        root mean squared error (rmse), which is approximately
        the average weighted distance of each pixel from the center
    '''
    center = _center_of_mass(ecal_slice)
    mse = 0
    dist = np.zeros(ecal_slice.shape)
    for i in range(ecal_slice.shape[0]):
        for j in range(ecal_slice.shape[1]):
            x=np.array([i,j])
            dist[i,j] = np.linalg.norm(x-center)
    total_energy = np.sum(ecal_slice)
    if total_energy>0:
        sse = np.sum(np.multiply(ecal_slice,dist**2))
        mse = sse/total_energy
        rmse = np.sqrt(mse)
    else:
        rmse = np.nan
    return rmse


def _get_max_layer(ecals):
    '''
    Takes a batch of cubes, and for each cube:
        Gets the depth of the layer in the cube 
        with the highest total energy
    args:
        ecals (np.array): either 3D batch of 2D shades, 
        or 4D batch of 3D cubes) 
    returns:
        depth (int) of the layer with the highest total energy
    '''
    if len(ecals.shape)==4:
        ecals = unroll_ecals(ecals)    
    assert ecals.shape!=(20,20,25), "Wrong shape"
    sums = ecals.sum(axis=1)
    return sums.argmax(axis=1)


class PlotSuite:
    def __init__(self,real_data, gen_data, exp_name, output_dir):
        self.real_ecals = real_data[0]
        self.gen_ecals = gen_data[0]
        self.exp_name = exp_name
        self.output_dir = output_dir
        self.real_targets = real_data[1]
        self.gen_targets = gen_data[1]
        #Add new directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def find_equivalent_gen_sample(self,real_sample_id):
        '''
        Given the id of a real sample, return the id of
        a generated sample with the same particle type,
        and the closest input energy.
        args:
            real_sample_id: id of real sample
        returns:
            gen_sample_id: id of matching generated sample

        This was as confusing to write as it is to read, 
        but I tested it on a couple values and it works. 
        
        '''
        real_targets = self.real_targets
        gen_targets = self.gen_targets

        if real_sample_id is None:
            real_target = np.mean(real_targets,axis=0)
            gen_target = np.mean(gen_targets,axis=0)
            gen_sample_id = None
        else:
            particle_type,energy = real_targets[real_sample_id]
            same_type_ids = np.where(gen_targets[:,0]==particle_type)[0]
            #concatenate the original index onto the new array so we can recover the original index
            matching_gens = np.concatenate([gen_targets[same_type_ids],same_type_ids[:,np.newaxis]],axis=1)
            matching_id = np.argmin(np.abs(matching_gens[:,1] - energy))
            gen_sample_id = int(matching_gens[matching_id,2])
            assert gen_targets[gen_sample_id,0]==real_targets[real_sample_id,0]
            real_target = real_targets[real_sample_id]
            gen_target = gen_targets[gen_sample_id]
        #print "Sample id, type and input energy:"
        #print "real",real_sample_id, real_target
        #print "gen",gen_sample_id, gen_target
        return gen_sample_id,real_target,gen_target


    def select_sample(self,real_sample_id=None,gen_sample_id=None):
        '''
        select single real and generated ecal
        or provide average across all samples
        '''
        if (real_sample_id is not None) and (gen_sample_id is not None):
            real_ecal = self.real_ecals[real_sample_id]
            gen_ecal = self.gen_ecals[gen_sample_id]
            sample_title = ": Real Sample #{}".format(real_sample_id)
        else:
            real_ecal = np.mean(self.real_ecals,axis=0)
            gen_ecal = np.mean(self.gen_ecals,axis=0)
            sample_title = ': Avg Across Samples'
            real_sample_id = 'avg'

        assert real_ecal.shape==gen_ecal.shape
        assert real_ecal.shape==(20, 20, 25)
        return real_ecal,gen_ecal,sample_title,real_sample_id



    def plot_energy_scatterplot(self,p0_energy_data,p1_energy_data,step=None):
        """Creates a scatterplot of input to output energy for both particle types saves it to disk.

        Only includes generated data if it's present. step should be the current training step number.
        NOTE: not threadsafe
        """
        # Subfunction to create a scatterplot for one of the two kinds of particles.
        exp_name = self.exp_name
        output_dir = self.output_dir

        print('Plotting energy ratio scatterplot...')
        def create_particle_scatterplot(energy_data, ax):
            input_energy_vals = np.array(energy_data.input)
            output_energy_vals = np.array(energy_data.output)
            dummy_vals = np.array(energy_data.dummy)
        
            real_inputs = input_energy_vals[(dummy_vals == 0)]
            real_outputs = output_energy_vals[(dummy_vals == 0)]
            real_label = 'Real: $\mu={0:.2f}$'.format(real_outputs.mean())
            gen_inputs = input_energy_vals[(dummy_vals == 1)]
            gen_outputs = output_energy_vals[(dummy_vals == 1)]
            
            #plot real points
            ax.plot(real_inputs, real_outputs, marker='o', lw=0, color=REAL_COLOR,
                    label=real_label, alpha=0.4, markersize=3,
                    markeredgewidth=0.0)

            #plot generated points, if they exist
            if len(gen_inputs) > 0:
                gen_label = 'Gen: $\mu={0:.2f}$'.format(gen_outputs.mean())
                ax.plot(gen_inputs, gen_outputs, marker='o', lw=0, color=GEN_COLOR,
                 label=gen_label, alpha=0.4, markersize=3, markeredgewidth=0.0)
            ax.set_title('Particle Type {}'.format(energy_data.type))
            ax.legend(loc='upper left')

        fig, (ax0, ax1) = plt.subplots(2, sharex=True, sharey=True)
        create_particle_scatterplot(p0_energy_data, ax0)
        create_particle_scatterplot(p1_energy_data, ax1)

        plt.xlabel('Input Energy')
        fig.text(0.04, 0.5, 'Total Output Energy', va='center', rotation='vertical')
        # NOTE: tight_layout() doesn't play nice with the shared y-axis for some reason

        step_str = ', Step {}'.format(step) if step else ''
        st = fig.suptitle('Input Energy vs. Total Output Energy{}'.format(step_str),
                          fontsize="x-large")
        # shift subplots down:
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)

        if output_dir:
            step_str = '-{}'.format(step) if step is not None else ''
            filename = exp_name + '-energy-scatter{}.png'.format(step_str)
            full_path = os.path.join(output_dir, filename)
            plt.savefig(full_path)
            plt.clf() # Clears matplotlib
            print('Saved energy data plot to {}.'.format(full_path))
        else:
            plt.show()


    def plot_energy_histogram(self,p0_energy_data,p1_energy_data,step=None):
        """Creates a histogram of the ratio of the output to input energy and saves it to disk.

        Creates plots for both real and generated data, and for both particle types.

        NOTE: not threadsafe
        """
        exp_name = self.exp_name
        output_dir = self.output_dir

        print('Plotting energy ratio histogram...')
        # Subfunction to create a histogram for one of the two kinds of particles.
        def create_particle_histogram(energy_data, ax):
            input_energy_vals = np.array(energy_data.input)
            output_energy_vals = np.array(energy_data.output)
            dummy_vals = np.array(energy_data.dummy)
        
            real_inputs = input_energy_vals[(dummy_vals == 0)]
            real_outputs = output_energy_vals[(dummy_vals == 0)]
            real_ratios = np.array([1.0 * output / input
                                    for input, output in zip(real_inputs, real_outputs)])
            real_hist_label = 'Real: $\mu={0:.2f}$, $\sigma={1:.2f}$'.format(real_ratios.mean(),
                                                                             real_ratios.std())
            gen_inputs = input_energy_vals[(dummy_vals == 1)]
            gen_outputs = output_energy_vals[(dummy_vals == 1)]

            #include generated data if it's there
            if len(gen_inputs>0):
                gen_ratios = np.array([1.0 * output / input
                                       for input, output in zip(gen_inputs, gen_outputs)])
                # When plotting the two histograms together, throw out huge outliers to ensure the graph is
                # interpretable. 4 standard deviations should be plenty.
                gen_ratio_deviations = abs(gen_ratios - np.mean(gen_ratios)) / np.std(gen_ratios)
                trunc_gen_ratios = gen_ratios[gen_ratio_deviations < 4]
                # Set the boundaries of the histogram based on the max and min of the data
                hist_range = (min(min(real_ratios), min(trunc_gen_ratios)),
                              max(max(real_ratios), max(trunc_gen_ratios)))
                # Use the untruncated generated data for the mean and standard deviation stats, since the
                # truncated data just serves to make the picture look readable.
                gen_hist_label = 'Gen: $\mu={0:.2f}$, $\sigma={1:.2f}$'.format(gen_ratios.mean(),
                                                                               gen_ratios.std())
                ax.hist(trunc_gen_ratios, bins=20, range=hist_range, color=GEN_COLOR,
                        label=gen_hist_label, histtype='step', 
                        weights=np.zeros_like(trunc_gen_ratios) + 1. / trunc_gen_ratios.size)
            else:
                hist_range = (min(real_ratios),max(real_ratios))

            ax.hist(real_ratios, bins=20, range=hist_range, color=REAL_COLOR, label=real_hist_label,
                    histtype='step', weights=np.zeros_like(real_ratios) + 1. / real_ratios.size)
            ax.legend(loc='best', fancybox=True, framealpha=0.7)
            ax.set_title('Particle Type {}'.format(energy_data.type))
        fig, (ax0, ax1) = plt.subplots(2, sharex=True)
        create_particle_histogram(p0_energy_data, ax0)
        create_particle_histogram(p1_energy_data, ax1)

        fig.tight_layout()    
        
        step_str = ', Step {}'.format(step) if step else ''
        st = fig.suptitle('Particle Output-to-Input Energy Ratio Histogram{}'.format(step_str),
                          fontsize="x-large")
        # shift subplots down:
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)
        
        if output_dir:
            step_str = '-{}'.format(step) if step is not None else ''
            filename = exp_name + '-energy-hist{}.png'.format(step_str)
            full_path = os.path.join(output_dir, filename)
            plt.savefig(full_path)
            plt.clf() # Clears matplotlib
            print('Saved energy data histogram to {}.'.format(full_path))
        else:
            plt.show()


    def plot_pixel_intensity_kernel(self):
        '''
        Display a kernel plot of distributions of individual
        pixel intensities
        TODO may need to take a random sample to reduce processing time
        '''
        exp_name = self.exp_name
        output_dir = self.output_dir
        real_ecals = self.real_ecals
        gen_ecals = self.gen_ecals

        print('Plotting kde of pixel intensity...')
        KDE_BAND = 0.5
        xmin = min(np.min(real_ecals),np.min(gen_ecals))
        xmax = max(np.max(real_ecals),np.max(gen_ecals))
        x = np.linspace(xmin,xmax, 1000)
        
        def process_for_chart(ecals):
            flat_ecals = ecals.flatten()
            flat_ecals = flat_ecals[np.where(flat_ecals > 0.0)]
            kernel = KernelDensity(kernel='gaussian', bandwidth=KDE_BAND)
            kde = kernel.fit(flat_ecals[:, np.newaxis])
            log_dens = kde.score_samples(x[:, np.newaxis])
            return np.exp(log_dens)
        
        real_y = process_for_chart(real_ecals)
        gen_y = process_for_chart(gen_ecals)

        real_zero = _get_pct_zero(real_ecals)
        gen_zero = _get_pct_zero(gen_ecals)
        fig, ax = plt.subplots()
        ax.plot(x, real_y, '-', color=REAL_COLOR, label="Nonzero Real ({:.1f}% zero)".format(real_zero*100))
        ax.plot(x, gen_y, linestyle='dashed', color=GEN_COLOR, label="Nonzero Gen ({:.1f}% zero)".format(gen_zero*100))
        ax.legend(loc='upper right')
        ax.set_yscale("log", nonposy='clip')
        ax.set_ylim([1e-6,1])
        ax.set_xlabel('Energy of Cell')
        ax.set_ylabel('Frequency P(Energy)')

        plt.title("Histogram of Nonzero Cells")

        if output_dir:
            filename = exp_name + '-pixel-intensity-hist.png'
            full_path = os.path.join(output_dir, filename)
            plt.savefig(full_path)
            plt.clf()
        else:
            plt.show()


    def plot_pixel_intensity_hist(self, point_limit=10000):
        '''
        Display a histogram of distributions of individual
        pixel intensities
        TODO may need to take a random sample to reduce processing time
        '''
        exp_name = self.exp_name
        output_dir = self.output_dir
        real_ecals = self.real_ecals
        gen_ecals = self.gen_ecals

        print('Plotting  pixel intensity histogram...')
        real_zero = _get_pct_zero(real_ecals)
        gen_zero = _get_pct_zero(gen_ecals)
        real_label = "Real ({:.1f}% zero)".format(real_zero*100)
        gen_label = "Gen ({:.1f}% zero)".format(gen_zero*100)

        real = real_ecals.flatten()
        gen = gen_ecals.flatten()

        #reduce to just {point_limit} points
        if real.shape>point_limit:
            real = np.random.choice(real,point_limit)
        if gen.shape>point_limit:
            gen = np.random.choice(gen,point_limit)

        hist_range = (min(min(real), min(gen)),
                      max(max(real), max(gen)))

        fig, ax = plt.subplots()    
        ax.hist(real, bins=20, range=hist_range, color=REAL_COLOR, 
                label=real_label, histtype='step', 
                weights=np.zeros_like(real) + 1. / real.size)
        ax.hist(gen, bins=20, range=hist_range, color=GEN_COLOR,
                label=gen_label, histtype='step', 
                weights=np.zeros_like(gen) + 1. / gen.size)
        ax.legend(loc='best')
        ax.set_yscale("log", nonposy='clip')
        #ax.set_ylim([1e-6,1])
        ax.set_xlabel('Energy of Cell')
        ax.set_ylabel('Frequency P(Energy)')
        plt.title("Histogram of Nonzero Cells")

        if output_dir:
            filename = exp_name + '-pixel-intensity-hist.png'
            full_path = os.path.join(output_dir, filename)
            plt.savefig(full_path)
            plt.clf()
            print('Saved pixel intensity histogram to {}.'.format(full_path))
        else:
            plt.show()


    def plot_dispersion(self, sample_id=None):
        '''
        Generates a plot of energy dispersion (root mean
        squared error) vs. depth, for a single real and 
        single generated cube
        args:
            real_ecals: (4D numpy array) batch of real calorimeter readings
            gen_ecals: (4D numpy array) batch of generated calorimeter readings
            exp_name: name of experiment
            output_dir: where to save the figure
            sample_id: id of calorimeter sample to examine.  
                If None, takes average across samples.
        returns:
            None
        '''
        exp_name = self.exp_name
        output_dir = self.output_dir
        real_ecals = self.real_ecals
        gen_ecals = self.gen_ecals

        print('Plotting energy dispersion...')
        gen_sample_id,real_target,gen_target = self.find_equivalent_gen_sample(sample_id)
        real_ecal,gen_ecal,sample_title,sample_id =  self.select_sample(sample_id,gen_sample_id)
        z_range = range(real_ecal.shape[2])
        real_dispersion=[]
        gen_dispersion=[]
        for z in z_range:
            real_rmse=_root_mean_squared_error(real_ecal[:,:,z])
            gen_rmse=_root_mean_squared_error(gen_ecal[:,:,z])
            real_dispersion.append(real_rmse)
            gen_dispersion.append(gen_rmse)
            
        plt.plot(z_range,real_dispersion, color=REAL_COLOR, label='Real')
        plt.plot(z_range,gen_dispersion, color=GEN_COLOR, linestyle='dashed', label='Generated')
        plt.xlabel('Depth into ECAL')
        plt.ylabel('Energy spread (RMSE)')
        plt.title('Energy Spread by Layer'+sample_title)
        plt.legend()
        
        if output_dir:
            filename = exp_name + '-energy-spread-{}.png'.format(sample_id)
            full_path = os.path.join(output_dir, filename)
            plt.savefig(full_path)
            plt.clf()
            print('[EVAL] Saved energy spread plot to {}.'.format(full_path))
        else:
            plt.show()


    def plot_energy_by_layer(self, sample_id=None):
        '''
        Generates a line plot of energy intensity of each layer
        In both real and generated calorimeter
        args:
            real_ecals: (4D numpy array) batch of real calorimeter readings
            gen_ecals: (4D numpy array) batch of generated calorimeter readings
            exp_name: name of experiment
            output_dir: where to save the figure
            sample_id: id of calorimeter sample to examine.  If None, takes average across samples.
        returns:
            None
        '''
        exp_name = self.exp_name
        output_dir = self.output_dir
        real_ecals = self.real_ecals
        gen_ecals = self.gen_ecals

        gen_sample_id,real_target,gen_target = self.find_equivalent_gen_sample(sample_id)
        real_ecal,gen_ecal,sample_title,sample_id =  self.select_sample(sample_id,gen_sample_id)

        z_range = range(real_ecal.shape[2])
        real_layer_energy = np.sum(real_ecal,axis=(0,1))
        gen_layer_energy = np.sum(gen_ecal,axis=(0,1))
            
        plt.plot(z_range,real_layer_energy, color=REAL_COLOR, label='Real')
        plt.plot(z_range,gen_layer_energy, color=GEN_COLOR, linestyle='dashed', label='Generated')
        plt.xlabel('Depth into ECAL')
        plt.ylabel('Total Energy of Layer')
        plt.title('Total Energy by Layer'+sample_title)
        plt.legend()
        
        if output_dir:
            filename = exp_name +'-energy-by-layer-sample-{}.png'.format(sample_id)
            full_path = os.path.join(output_dir, filename)
            plt.savefig(full_path)
            plt.clf()
            print('Saved energy spread plot to {}.'.format(full_path))
        else:
            plt.show()


    def print_energy_per_cube(self):
        '''
        Not used.  Prints average total energy per cube. 
        '''
        real_ecals = self.real_ecals
        gen_ecals = self.gen_ecals
        
        print "Total Energy Per Cube"
        real_energy_per_cube = np.sum(real_ecals)/real_ecals.shape[0]
        print "Real: {:.1f}".format(real_energy_per_cube)
        gen_energy_per_cube = np.sum(gen_ecals)/gen_ecals.shape[0]
        print "Generated: {:.1f}".format(gen_energy_per_cube)


    def plot_depth_histogram(self):
        '''
        Plot the distribution across max-layer depths, 
        where max-energy depth is the index of the layer with 
        the highest total energy

        args:
            real_ecals: (4D numpy array) batch of real calorimeter readings
            gen_ecals: (4D numpy array) batch of generated calorimeter readings
            TODO save option
        returns:
            none
        '''
        exp_name = self.exp_name
        output_dir = self.output_dir
        real_ecals = self.real_ecals
        gen_ecals = self.gen_ecals

        print('Plotting max energy depth histogram...')
        real_depths = _get_max_layer(real_ecals)
        gen_depths = _get_max_layer(gen_ecals)
        
        # Subfunction to create a histogram for one of the two kinds of particles.
        def create_particle_histogram(real_depths, gen_depths, ax):
            # Set the boundaries of the histogram based on the max and min of the data
            hist_range = (0,24)
            real_hist_label = 'Real: $\mu={0:.2f}$, $\sigma={1:.2f}$'.format(real_depths.mean(),
                                                                             real_depths.std())
            ax.hist(real_depths, bins=25, range=hist_range, color=REAL_COLOR, label=real_hist_label,
                    histtype='step', weights=np.zeros_like(real_depths) + 1. / real_depths.size)
            # Use the untruncated generated data for the mean and standard deviation stats, since the
            # truncated data just serves to make the picture look readable.
            gen_hist_label = 'Gen: $\mu={0:.2f}$, $\sigma={1:.2f}$'.format(gen_depths.mean(),
                                                                           gen_depths.std())
            ax.hist(gen_depths, bins=25, range=hist_range, color=GEN_COLOR, linestyle='dashed',
                    label=gen_hist_label, histtype='step', 
                    weights=np.zeros_like(gen_depths) + 1. / gen_depths.size)
            # TODO The 'best' legend location doesn't work that well...
            ax.legend(loc='best', fancybox=True, framealpha=0.7)
            ax.set_xlabel('Depth of maximum-energy layer')
            ax.set_ylabel('Frequency P(Depth)')
            ax.set_title('Distribution of Max-Layer Depths')
        fig, ax = plt.subplots(1)
        create_particle_histogram(real_depths,gen_depths, ax)

        fig.tight_layout()    

        if output_dir:
            filename = exp_name + '-depth-hist.png'
            full_path = os.path.join(output_dir, filename)
            plt.savefig(full_path)
            plt.clf()
            print('Saved max depth histogram to {}.'.format(full_path))
        else:
            plt.show()


    def plot_unrolled_sample(self, sample_id=None): 
        '''
        plot side-by-side sample of real and generated ecal
        args:
            real_ecals: (4D numpy array) batch of real calorimeter readings
            gen_ecals: (4D numpy array) batch of generated calorimeter readings
            sample_id: index of ecal to sample
            exp_name: experiment name
            output_dir: where to put the chart
        '''
        exp_name = self.exp_name
        output_dir = self.output_dir
        print('Plotting unrolled samples...')
        
        gen_sample_id,real_target,gen_target = self.find_equivalent_gen_sample(sample_id)
        real_ecal,gen_ecal,sample_title,sample_id =  self.select_sample(sample_id,gen_sample_id)
        vmax = max(np.max(real_ecal),np.max(gen_ecal))

        def plot_windowshade(ax,ecal,title):
            shades = unroll_ecals(ecal[np.newaxis,:])
            shade = shades[0].T[:,:100]
            im=ax.imshow(shade, interpolation=None, cmap='spectral', \
                vmin=0, vmax=vmax)
            ax.set_title(title)
            return im

        fig, (ax0, ax1) = plt.subplots(2, figsize = (10,5.5))

        real_title = "Real particle type:{} energy:{} id:{}".format(real_target[0],round(real_target[1]),sample_id)
        gen_title = "Gen particle type:{} energy:{} id:{}".format(gen_target[0],round(gen_target[1]),gen_sample_id)
        im1=plot_windowshade(ax0,real_ecal,real_title)
        im2=plot_windowshade(ax1,gen_ecal,gen_title)
        cax = fig.add_axes([0.9, 0.15, 0.01, 0.7])
        fig.colorbar(im2,cax)

        if output_dir:
            filename = exp_name + '-windowshades-sample-{}.png'.format(sample_id)
            full_path = os.path.join(output_dir, filename)
            plt.savefig(full_path)
            plt.clf()
            print('[EVAL] Saved sample windowshade heatmaps to {}.'.format(full_path))
        else:
            plt.show()


    def plot_chart_suite(self,sample_id=None):
        print("\nPlotting chart suite")
        self.plot_pixel_intensity_hist()
        self.plot_depth_histogram()
        self.plot_dispersion(sample_id)
        self.plot_energy_by_layer(sample_id)
        self.plot_unrolled_sample(sample_id)