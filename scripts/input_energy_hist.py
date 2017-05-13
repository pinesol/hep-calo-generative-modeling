import matplotlib
# Force matplotlib to not use any Xwindows backend. Allows this to run on mercer
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import data_loader

dl = data_loader.DataLoader([80,20])

input_energies = []
batch_num = 1
for _, target in dl.train_batch_iter(batch_size=1000, num_epochs=1):
    print batch_num
    batch_num += 1
    for energy in target[:, 1]:
        input_energies.append(energy)
        
plt.hist(input_energies, 50, normed=1, facecolor='green', alpha=0.75)
plt.savefig('input_energies_hist.png')

print 'mean:', np.mean(input_energies)
print 'stddev:', np.std(input_energies)
print 'min:', min(input_energies)
print 'max:', max(input_energies)
