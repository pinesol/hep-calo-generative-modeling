#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:15:03 2016

@author: malkini
"""
import data_loader
import numpy as np
import h5py
import os

class Ticker():
    def __init__(self, N, N_buckets, output_dir):
        self.N = N
        self.N_buckets = N_buckets
        self.output_dir = output_dir
        self.init_buckets()
        self.init_container()
        self.init_buffer()

    
    def init_buckets(self):
        _,buckets = np.histogram(np.arange(501), self.N_buckets)
        buckets[-1] += 1
        self.buckets = buckets
    
    def init_container(self):
        self.container = {}
        self.midpoints = {}
        for b in self.buckets[:-1]:
            for p in [0,1]:
                self.container[(p,b)] = []
                self.midpoints[(p,b)] = b + ((self.buckets[1]-self.buckets[0])/2.0)
                
    def init_buffer(self, counter=None):
        self.buffer_counter = counter if counter else 0
        self.buffer_obs = 0
        self.buffer_ecal = []
        self.buffer_target = []
    
    def add_to_bucket(self, particle, momentum_val, data):
        momentum_bucket = (np.digitize([momentum_val], self.buckets)-1)
        self.container[(particle,self.buckets[momentum_bucket[0]])].append(data)
    
    def check_loads(self, check=True):
        for k,v in self.container.iteritems():
            if check:
                if len(v) >= self.N:
                    self.pop_from_bucket(k)
            else:
                if len(v):
                    self.pop_from_bucket(k)
                
    def pop_from_bucket(self,bucket_key):
        data = np.array(self.container[bucket_key]).mean(0)
        print 'adding to buffer {}'.format(bucket_key)
        self.add_to_buffer(bucket_key, data)
        self.container[bucket_key] = []

    def add_to_buffer(self, bucket_key, data):
        self.buffer_ecal.append(data)
        self.buffer_target.append([bucket_key[0], self.midpoints[bucket_key], 0, 0, 0])
        self.buffer_obs += 1
        if self.buffer_obs == 256:
            'print dumping buffer {}'.format(self.buffer_counter)
            self.dump_buffer()
    
    def dump_buffer(self):
        with h5py.File(os.path.join(self.output_dir,'averaged_{}.h5'.format(self.buffer_counter)), 'w') as hf:
            hf.create_dataset('ECAL',      data=np.array(self.buffer_ecal))
            hf.create_dataset('target', data=np.reshape(np.array(self.buffer_target),[-1,1,5]))
        self.init_buffer(counter=self.buffer_counter+1)

av = Ticker(10,25,'/scratch/cdg356/udon/data/average/25bins/')

data_loader_obj = data_loader.DataLoader([100, 0, 0], False)
train_batch_iter = data_loader_obj.batch_iter(partition_index=0,batch_size=256, num_epochs=1)

for ecals, targets in train_batch_iter:
    for idx in range(ecals.shape[0]):
        av.add_to_bucket(targets[idx, 0], targets[idx, 1], ecals[idx])
        av.check_loads(check=True)
av.check_loads(check=False)








