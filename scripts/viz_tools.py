
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 12:59:59 2016

@author: malkini
"""

import pandas as pd
import numpy as np
import glob

import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D



### converting hdf5 dctionaries to pd.DataFrames
def np2df(array, drop_zeros=True):
    '''
    convert 3D (x,y,z) array into a long-format pd.DataFrame 
    drop_zeros arg will only populate the dataframe with non-zero elements
    
    input: 3D numpy array
    returns: pd.DataFrame()
    '''
    data = []
    X,Y,Z  =  array.shape
    print array.shape
    print X,Y,Z
    for i in range(X):
        chunk = array[i,:,:]
        append = np.vstack([np.ones(Y)*(i+1), np.arange(1,Y+1)]).T
        chunk = np.hstack([append,chunk])
        data.append(chunk)
    data = np.concatenate(data, axis=0)
    data = pd.DataFrame(data, columns=['X','Y']+range(1,Z+1))
    data = data.set_index(['X','Y'])
    data = data.stack()
    data.index.names = ['X','Y','Z']
    if drop_zeros:
        data = data.ix[data!=0.0]
    data = data.reset_index()
    data.rename(columns={0:'energy'},inplace=True)
    return data
    
def data2df(data, meta=None, drop_zeros=True):
    '''
    convert data_loader-dict to long-format pd.DataFrame() 
    drop_zeros arg will only populate the dataframe with non-zero elements
    option to append meta-data 
    exp argument is an indicator for the wide-long reshape index
    
    inputs: 
        data: 3D numpy array
        meta: 2D numpy array (0th index must match -th index of data) 
    returns: pd.DataFrame() with columns [energy,x,y,z,t1,...,t5,exp]
    '''
    if meta:
        assert data.shape[0]==meta.shape[0]
    #loop through obs and append to master df
    master = []
    for obs in range(data.shape[0]):
        print data[obs].shape
        df = np2df(data[obs],drop_zeros)
        if meta:
            for index,value in enumerate(meta[obs].ravel()):
                df['t'+str(index)] = value
        df['id'] = obs
        master.append(df)
    # concat
    master = pd.concat(master, axis=0)
    
    return master
    
#### numpy helpers
    
def zeros2nan(array):
    '''
    return np.array with zeros replaced with nan's
    '''
    replica = array.copy()
    replica[replica==0.0] = np.nan
    return replica
    
    
def binarize(array, threshold=0.0):
    '''
    return binary np.array (if x>threshold: x=1 , else: x=x)
    '''
    replica = array.copy()
    replica[replica>threshold] = 1
    return replica



### vizualizations/plotting

def _parse_axis(axis):
    '''
    utility function for parse axis-arg
    allows plotting function to accept cartesian (X,Y,Z) or index (0,1,2) format 
    '''
    axis2sym = {0:'X',1:'Y',2:'Z'}
    sym2axis = {v:k for k,v in axis2sym.iteritems()}
    if type(axis)==str:
        axis = sym2axis[axis.upper()]
    return axis, axis2sym

#TODO: load the saved momentum and types into this.  They get saved in train.py save_samples()
def samples2iter(samples):
    master = []
    s = np.reshape(samples,(-1,2,5,20,20))
    for i in range(2):
        for j in range(5):
            master.append([(i,j),s[:,i,j,:,:]])
    return master

def conditional_2d_heatmap(samples, sample_id=None, scaling_root=1, binary=False, save=None):
    '''
    plots grid of 2D heatmaps of generate 2D images, one of each conditional type
    maintains consistent colormap across slices
    input:
        samples: dict containing 'gen_samples': array of images, and 'y': array of target values
        sample_id: 1-25.  pick which sample set to view. If None, returns an average across all samples
        scaling_root: rescales data=data**1/scaling_root for better visualization  [data is very skewed]
        binary: plot binarized version of data [if x>1: x=1]
        save: path to save plot
        n_momentums: different 
    '''
    im = samples['gen_samples']
    y = samples['y'][0] #just pick first batch of y becauase they are all the same
    n_particles = y.shape[0]
    n_momentums = y.shape[1]
    
    #either take mean across samples or a single sample
    if sample_id is None:
        im = im.mean(axis=0)
        sample_title = "mean across samples"
    else:
        im = im[sample_id]
        sample_title = "sample #{}".format(sample_id)
    
    #either binarize data, scale it to root, or neither
    if binary: # binary option
        im = binarize(im)
        bin_title = "; binarized"
    else:      # for better visualization (skewed distribution)
        im = np.power(im,float(1)/scaling_root)
        bin_title = ""
    
    fig = plt.figure(figsize=(11,22))
    fig.suptitle('Energy heatmaps: '+ sample_title + bin_title, fontsize=14, fontweight='bold')

    #plot each chart
    for i in range(n_particles):
        for j in range(n_momentums):
            num=n_particles*j+i+1
            ax = fig.add_subplot(n_momentums,n_particles, num)
            
            particle_type = y[i,j,0]
            momentum = y[i,j,1]

            sns.heatmap(im[i,j],
                        vmin = im.min(),
                        vmax = im.max(),
                        xticklabels=False,
                        yticklabels=False)
            plt.title('Particle Type: {}, Momentum: {}'.format(
                                                particle_type,
                                                momentum))
    if save:
        plt.savefig('{}{:03d}.png'.format(save))
    plt.show()
    plt.close()

def slice_axis_heatmap(array, axis, scaling_root=1, binary=False, save=None):
    '''
    iteratively plots 2D heatmap along the specified dimension
    maintains consistent colormap across slices
    input:
        array: 3D np.array
        axis: axis to walk through
        scaling_root: rescales data=data**1/scaling_root for better visualization  [data is very skewed]
        binary: plot binarized version of data [if x>1: x=1]
        save: path to save plot
    '''
    # prase axis arg
    axis, axis2sym = _parse_axis(axis)
    # swap axes so that we are always looping through the 'first' axis
    array = array if axis==0 else array.swapaxes(0,axis)
    if binary: # binary option
        array = binarize(array)
    else:      # for better visualization (skewed distribution)
        array = np.power(array,float(1)/scaling_root)
    # loop through slices and plot
    for i in range(array.shape[0]):
        sns.heatmap(array[i,:,:],
                    vmin = array.min(),
                    vmax = array.max())
        plt.title('Slice along {}-axis @: {} {}'.format(axis2sym[axis],i,'[binary]' if binary else ''))
        plt.ylabel('{}'.format('Y-axis' if axis==0 else 'X-axis'))
        plt.xlabel('{}'.format('Y-axis' if axis==2 else 'Z-axis'))
        if save:
            plt.savefig('{}{:03d}.png'.format(save,i))
        plt.show()
        plt.close()

def hist_by_dim(array, axis, save=None):
    '''
    plots histogram along specified axis, averaging along the remaning axes
    input:
        array: 3D np.array
        axis: axis to plot
        save: path to save plot
    '''
    # prase axis arg
    axis, axis2sym = _parse_axis(axis)
    # swap axes so that we are always averaging the complemtary axes
    array = array if axis==2 else array.swapaxes(axis,2)
    averaged = array.mean(0).mean(0)  # axes-swap allows for naive double 0th-axis mean
    # plot signature
    pd.Series(averaged).plot(kind='bar')
    plt.title('{}-axis signature'.format(axis2sym[axis]))
    plt.show()
    if save:
       plt.savefig('{}.png'.format(save))
    plt.close()
    
    
def collapse_axis_heatmap(array, axis, scaling_root=1, binary=False, save=None):
    '''
    plots 2D heatmap, collapsing/averaging the specified axis
    input:
        array: 3D np.array
        axis: axis to collapse
        scaling_root: rescales data=data**1/scaling_root for better visualization  [data is very skewed]
        binary: plot binarized version of data [if x>1: x=1]
        save: path to save plot
    '''
    # prase axis arg
    axis, axis2sym = _parse_axis(axis)
    # max for consistent color-map
    data = array.mean(axis=axis)
    if binary:
        data = binarize(data)
        cmax = 1
    else:
        data = np.power(data,float(1)/float(scaling_root))
        cmax = (array.max()/float(24))**(1/float(scaling_root))
    # plot 2D heatmap
    sns.heatmap(data,
                vmin = 0.0,
                vmax = cmax)
    plt.title('Collapse along {}-axis {}'.format(axis2sym[axis], '[binary]' if binary else ''))
    plt.ylabel('{}'.format('Y-axis' if axis==0 else 'X-axis'))
    plt.xlabel('{}'.format('Y-axis' if axis==2 else 'Z-axis'))
    if save:
        plt.savefig('{}.png'.format(save))
    plt.show()
    plt.close()
    
def scatter3d(frame, scaling_root=1, top=None, save=None):
    '''
    3D scatterplot
    input:
        scaling_root: rescales data=data**1/scaling_root for better visualization  [data is very skewed]
        top: keep top% highest energy values
        save: path to save plot
    '''
    frame['energy'] = np.power(frame['energy'],1/float(scaling_root))
    # option to keep top% of energy values
    if top:
        size = int((frame.shape[0]*top/float(100)))
        frame = frame.sort_values(by=['energy'], ascending=False)[:size]
    cmax = frame['energy'].max()
    cmin = frame['energy'].min()
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mymap = plt.get_cmap("Reds")
    ax.scatter(frame['X'],frame['Y'],frame['Z'],c=frame['energy'].values,cmap=mymap,vmin=cmin,vmax=cmax)
    if save:
        plt.savefig('{}.png'.format(save))
    plt.show()
    plt.close()
    

    

    
        
    
    