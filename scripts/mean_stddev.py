import numpy as np
import data_loader

dl = data_loader.DataLoader([80, 20])

mean_vals = 0.0
count = 0
# 1 sample per batch and one epoch
for ecals, _ in dl.batch_iter(0, 1, 1):
    ecals = data_loader.log_ecals(ecals)
    ecals = data_loader.truncate_ecals(ecals, (10,10)) # TODO TODO 

    mean_vals = np.sum(ecals)
    count += ecals.size
mean = mean_vals / count
print('mean pixel energy for 80% training data: {}'.format(mean))

stddev = 0.0
for ecals, _ in dl.batch_iter(0, 1, 1):
    ecals = data_loader.log_ecals(ecals)
    ecals = data_loader.truncate_ecals(ecals, (10,10)) # TODO TODO 
    stddev += np.sum(np.square(ecals - mean))
stddev = np.sqrt(stddev / count)
print('standard deviation pixel energy for 80% training data: {}'.format(stddev))
