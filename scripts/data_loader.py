import h5py
import numpy as np
import os.path
import random
import re
import math

# Global constants, to be used by clients of this module.
# The assumed dimensions of a single data point as a 3d tuple
DATA_DIM = (20, 20, 25)
DATA_POINT_SIZE = 20*20*25

# Constants that are local to this file are prefixed with an underscore.
# Directory where the data files are located.
_SCRATCH_DIR = '/scratch/cdg356/udon/data/'
_FILENAME_REGEX = 'GammaPi0_shuffled_[0-9]+.h5'
# Arbitrarily chosen file to use get the data for the test file.
_TEST_BASE_FILENAME = 'GammaPi0_shuffled_26.h5'
# Location of the test file, a tiny subset of the data intended to test code.
_TEST_DATA_FILENAME = 'test_data.h5'
_LOCAL_TEST_DATA_DIR = '..'
_SPLIT_TEST_DATA_DIR = 'test_data_split'
# Location of the averaged data
_AVG_SCRATCH_DIR = '/scratch/cdg356/udon/data/average/25bins/'
_AVG_FILENAME_REGEX = 'averaged_[0-9]+.h5'


class DataLoader(object):
    """Class for loading real or testing data, and spliting it into batches.

    This class assumes that real data is in _SCRATCH_DIR, defined above. It will
    put testing data that it generates from the real data into the directory
    above the current one.

    Example usage:

    dl = DataLoader(partition=[80, 20], test=False)
    for ecal, target in dl.train_batch_iter(batch_size, num_epochs):
      ...do stuff with ecal and target...
    """
    def __init__(self, partition, test=False, local_test_data_dir=_LOCAL_TEST_DATA_DIR):
        """Creates a new DataLoader object.

        Arguments:
          partition: A list of integers representing percentages that describe how data
            should be split for training and validation. The contents of the list must sum
            to 100. The first element will be considered the training data set, and the
            remaining ones will be considered different validation training sets.
          test: A boolean. If true, the test data set will be used.
          local_test_data_dir: Base directory where the test data can be found.
        """
        assert sum(partition) == 100, 'The sum of the partition list must be 100: {}'.format(partition)
        self._partition = partition
        self._test = test
        # Split the files up according to the self._partition list.
        self._partitioned_filenames = []
        filenames = data_filenames(shuffle=False, test=self._test,
                                   local_test_data_dir=local_test_data_dir)
        part_start = 0
        for i, part_size in enumerate(self._partition):
            part_end = part_start + int(len(filenames) * 0.01 * part_size)
            assert part_end - part_start > 0, 'The number of files in partition {} is zero.'.format(i)
            self._partitioned_filenames.append(filenames[part_start:part_end])


    def train_batch_iter(self, batch_size, num_epochs):
        """Calls batch_iter on partition zero, assumed to be the training partition."""
        return self.batch_iter(0, batch_size, num_epochs)
           
            
    def batch_iter(self, partition_index, batch_size, num_epochs):
        """
        Loads the data files from the given partition, splits each one up into batches of the
        specified size, and returns each batch in shuffled order. It does this as many times
        as specified by 'num_epochs'.
      
        Example:
          for ecals, targets in batch_iter(partition_index=1, batch_size=64, num_epochs=20):
          # Give 'ecals' to the discriminator and 'target' to the CGAN.
        
        Args:
          partition_index: The data partition to load. Must be between zero and the number of
            partitions - 1.
          batch_size: The size of each training mini-batch. Must be between 1 and 10000.
          num_epochs: The number of times to loop through the files.

        Returns:
          An iterator for the next mini-batch. Each iterator consists of a tuple with two elements.
          The first element is the 'ECAL' tensor, the second element is the 'target' tensor.
          The ECAL tensor is a 4-dimensional numpy array of shape [batch_size, DATA_DIM[0],
          DATA_DIM[1], DATA_DIM[2]].
          The target tensor is a 2-dimensional numpy array with shape [batch_size, 2]. The second
          dimension has two numbers, the particle type, and the initial energy. It doesn't include
          the momentum values from the input data, because they are redundent: one element of the
          three momentum values is always equal to the initial energy, and the other two are always
          zero.
        Raises:
          AssertionError if the partition_index is out of bounds of the object's list.
        """
        err_msg = 'partition index {} out of range 0-{}'.format(
            partition_index, len(self._partitioned_filenames))
        assert partition_index < len(self._partitioned_filenames), err_msg
        filenames = self._partitioned_filenames[partition_index]
        for epoch in xrange(num_epochs):
            print('[DATA] Partition {}: Epoch {} of {}'.format(partition_index, epoch+1, num_epochs))
            # Shuffle the filenames each epoch so we don't always go through in the same order.
            np.random.shuffle(filenames)
            for filename in filenames:
                # Load the file, split its content into batches.
                h5_dict = load(filename)
                data_size = h5_dict['ECAL'].shape[0]
                num_batches = int(data_size / batch_size) + (0 if data_size % batch_size == 0 else 1)
                #print('[DATA] Loading {} experiments in {} batches from {}'.format(data_size, num_batches, filename))
                # Shuffle the order of the batches, and extract each batch one by one.
                for batch_num in np.random.permutation(np.arange(num_batches)):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    if start_index < end_index:
                        # 20-24 are always zero in the x and y axis, remove them.
                        ecal = h5_dict['ECAL'][start_index:end_index, :DATA_DIM[0], :DATA_DIM[1], :]
                        # Only returns the first two elements of the target array.
                        # The middle dimension is unused, so squeeze() removes it.
                        # If end_index-start_index = 1, then squeeze will be too aggressive and take
                        # it down to shape [2]. np.reshape makes sure the shape is [end_index-start_index,2].
                        target = np.reshape(h5_dict['target'][start_index:end_index, :, :2].squeeze(),
                                            (-1, 2))
                        yield ecal, target
                      
    
# Gets the data file paths in shuffled order
def data_filenames(shuffle=False, test=False, local_test_data_dir=_LOCAL_TEST_DATA_DIR):
    if test:
        filenames = _split_test_data(local_test_data_dir)
    else:
        # Find all files in the scratch dir that match this pattern.
        filenames = []
        for fname in os.listdir(_SCRATCH_DIR):
            fpath = os.path.join(_SCRATCH_DIR, fname)
            if os.path.isfile(fpath) and re.match(_FILENAME_REGEX, fname):
                filenames.append(fpath)
    if shuffle:
        np.random.shuffle(filenames)
    return filenames

                    
# Loads the test h5 data as a dictionary, writing it to disk if it doesn't already exist.
# If from_scratch is True, it reads the data from _SCRATCH_DIR on HPC. Otherwise, it tries to load
# it from the local disk.
def load_test_data(from_scratch=False, local_test_data_dir=_LOCAL_TEST_DATA_DIR):
    test_data_file = os.path.join(local_test_data_dir, _TEST_DATA_FILENAME)
    print test_data_file
    if not from_scratch and os.path.isfile(test_data_file):
        test_h5_dict = load(test_data_file)
    else:
        # Load data file and truncate it from scratch
        fpath = os.path.join(_SCRATCH_DIR, _TEST_BASE_FILENAME)
        print('[DATA] Loading fresh test data from {}'.format(fpath))
        TEST_DATA_SIZE = 100
        test_h5_dict = load(fpath, slice_size=TEST_DATA_SIZE)
        # write this to disk for future use
        with h5py.File(test_data_file, 'w') as test_h5_file:
            print('[DATA] Creating test data file at {}'.format(test_data_file))
            for name, dataset in test_h5_dict.iteritems():
                test_h5_file.create_dataset(name, data=dataset)
    return test_h5_dict


# Returns the full h5 file as a dictionary. 'filepath' is the full path to a h5 file.
# If slice_size is a positive integer, it will only load that many items per dataset.
def load(filepath, slice_size=None):
    with h5py.File(filepath, 'r') as h5:
        h5_dict = {}
        for key in h5.keys():
            # When slice_size is None, the full value of h5[key] is loaded.
            # This line causes the value of h5[key] to be materialized. Without it, closing
            # the h5 file would make this object unreadable.
            h5_dict[key] = h5[key][:slice_size]
    return h5_dict


def unroll_ecals(ecals):
    """
    Unrolls an ecal cube in a spiral, so that the center is at the top spiralling outward.
    
    args:
        ecals: np.array of shape (batch_size, width, height, depth), where width == height.
    returns:
        np.array of shape (batch_size, width*width, depth).
    """
    batch_size, width, height, depth = ecals.shape
    assert width == height, 'Width and height of ECAL are not equal: {}'.format(ecals.shape)
    unrolled = np.zeros((batch_size, width*width, depth))

    half = (width-1) / 2
    x = y = 0
    dx = 0
    dy = -1
    for j in xrange(width*width):
        # record rod
        unrolled[:, j, :] = ecals[:, x+half, y+half, :]
        # if a corner of the spiral is reached, turn
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        # move x and y one unit    
        x, y = x + dx, y + dy
    return np.array(unrolled)


# The exact inverse of unroll_ecals
def roll_ecals(unrolled_ecals):
    batch_size, longwidth, depth = unrolled_ecals.shape
    width = int(math.sqrt(longwidth))
    assert width*width == longwidth, '{} is not square'.format(longwidth)

    rolled_ecals = np.zeros((batch_size, width, width, depth))

    half = (width-1) / 2
    x = y = 0
    dx = 0
    dy = -1
    for j in xrange(width*width):
        # record rod
        rolled_ecals[:, x+half, y+half, :] = unrolled_ecals[:, j, :]
        # if a corner of the spiral is reached, turn
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        # move x and y one unit
        x, y = x + dx, y + dy
    return rolled_ecals


# truncates ecals by height and width from the edges inward, leaving the batch
# and depth dimensions unchanged.
# ecals are assumed to be a 4d tensor: (batch, DATA_DIM[0], DATA_DIM[1], DATA_DIM[2])
# Example usage:
# truncated_ecals = data_loader.truncate_ecals(my_ecals, (10,10))
def truncate_ecals(ecals, new_shape):
    assert len(ecals.shape) == 4, ecals.shape
    assert len(new_shape) == 2, len(new_shape)
    assert new_shape[0] <= DATA_DIM[0] and new_shape[1] <= DATA_DIM[1], new_shape
    assert ecals.shape[1] == DATA_DIM[0] and ecals.shape[2] == DATA_DIM[1] and ecals.shape[3] == DATA_DIM[2], ecals.shape
    top = left = (DATA_DIM[0] - new_shape[0]) // 2
    bottom = right = int(math.ceil((DATA_DIM[1] - new_shape[1]) / 2.0))
    return ecals[:, top:-bottom, left:-right, :]


# Pad the ecals with zeros until you get back to its original shape
def untruncate_ecals(ecals):
    assert len(ecals.shape) == 4, ecals.shape
    assert ecals.shape[1] <= DATA_DIM[0] and ecals.shape[2] <= DATA_DIM[1] and ecals.shape[3] <= DATA_DIM[2], ecals.shape
    # don't pad the 0th and 3rd dims. Pad the 1st and 2nd dims with zeros to get back to the original shape.
    top = left = (DATA_DIM[0] - ecals.shape[1]) // 2
    bottom = right = int(math.ceil((DATA_DIM[1] - ecals.shape[2]) / 2.0))
    pad_dims = ((0, 0), (top, bottom), (left, right), (0, 0))
    return np.pad(ecals, pad_width=pad_dims, mode='constant', constant_values=0.0)


def log_ecals(ecals):
    return np.log(1 + ecals)


def unlog_ecals(ecals):
    return np.exp(ecals) - 1


def normalize_ecals(ecals, mean, stddev):
    return (ecals - mean) / stddev
    

def denormalize_ecals(ecals, mean, stddev):
    return (ecals * stddev) + mean


# If the _SPLIT_TEST_DATA_DIR directory doesn't exist, this function will create
# it. Then it will take the test data file, split it into 10 even parts and save
# those files in the new directory.
# Regardless of whether the data was created on this function call, the 10
# xfilenames will be returned.
# private function.
def _split_test_data(test_data_dir):
    NUM_FILES = 10
    filenames = [os.path.join(test_data_dir,
                              '{}/test_data_{}.h5'.format(_SPLIT_TEST_DATA_DIR, i+1))
                 for i in range(NUM_FILES)]
    # create the files if they don't exist
    split_test_data_dir = os.path.join(test_data_dir, _SPLIT_TEST_DATA_DIR)
    if not os.path.isdir(split_test_data_dir):
        # make the directory
        os.makedirs(split_test_data_dir)
        # load the test data and split it into 10 files
        test_h5_dict = load_test_data(from_scratch=False)
        DATA_PER_FILE = 10
        for i in range(NUM_FILES):
            filename = filenames[i]
            print('[DATA] Creating test data file at {}'.format(filename))
            start = i * DATA_PER_FILE
            end = start + DATA_PER_FILE
            with h5py.File(filename, 'w') as test_h5_file:
                for name, dataset in test_h5_dict.iteritems():
                    test_h5_file.create_dataset(name, data=dataset[start:end])
    return filenames





