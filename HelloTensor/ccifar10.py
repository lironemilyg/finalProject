import os
import sys
from six.moves import urllib
import tarfile
import pickle
import numpy as np

class Data(object):
    def __init__(self):
        self.DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.dest_directory = '\\cifar10_data'
        self.train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        self.test_files = ['test_batch']
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.train_imgs = []
        self.train_labels = []
        self.test_imgs = []
        self.test_imgs = []
        self.cur_ind = 0
        self.maybe_download_and_extract()
        self.load(self.dest_directory)


        self.num_samples = self.train_imgs.shape[0]
        self.sampler = np.random.permutation(self.num_samples).astype('int32')

    def maybe_download_and_extract(self):
        """Download and extract the tarball from Alex's website."""

        if not os.path.exists(self.dest_directory):
          os.makedirs(self.dest_directory)
        filename = self.DATA_URL.split('/')[-1]
        filepath = os.path.join(self.dest_directory, filename)
        if not os.path.exists(filepath):
          def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
          filepath, _ = urllib.request.urlretrieve(self.DATA_URL, filepath, _progress)
          print()
          statinfo = os.stat(filepath)
          print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        extracted_dir_path = os.path.join(self.dest_directory, 'cifar-10-batches-py')
        if not os.path.exists(extracted_dir_path):
          tarfile.open(filepath, 'r:gz').extractall(self.dest_directory)

    def load(self,path):
        path = os.path.join(path,'cifar-10-batches-py')
        for file in self.train_files:
            tpath = os.path.join(path,file)
            with open(tpath, 'rb') as fo:
                dicto = pickle.load(fo, encoding='bytes')
            im_tr = np.reshape(dicto[b'data'], (-1, 3, 32, 32))
            im_tr = np.transpose(im_tr, (0, 2, 3, 1))
            self.train_imgs.append(im_tr)
            self.train_labels.append(dicto[b'labels'])
        self.train_imgs = np.concatenate(self.train_imgs, 0)
        self.train_labels = np.concatenate(self.train_labels, 0)

    def next_batch(self, batch_size):
        if self.cur_ind + batch_size <= self.num_samples:
            next_sample = self.sampler[self.cur_ind:self.cur_ind + batch_size]
            self.cur_ind += batch_size
        else:
            next_sample = self.sampler[self.cur_ind:]
            self.cur_ind = self.cur_ind + batch_size - self.num_samples
            self.sampler = np.random.permutation(self.num_samples).astype('int32')
            next_sample = np.append(next_sample, self.sampler[0:self.cur_ind])
        next_b = self.train_imgs[next_sample]
        next_l = self.train_labels[next_sample]
        return next_b/255.,next_l


