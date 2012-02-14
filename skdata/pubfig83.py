# -*- coding: utf-8 -*-
"""PubFig83 Dataset

http://www.eecs.harvard.edu/~zak/pubfig83

If you make use of this data, please cite the following paper:

"Scaling-up Biologically-Inspired Computer Vision: A Case-Study on Facebook."
Nicolas Pinto, Zak Stone, Todd Zickler, David D. Cox
IEEE CVPR, Workshop on Biologically Consistent Vision (2011).
http://pinto.scripts.mit.edu/uploads/Research/pinto-stone-zickler-cox-cvpr2011.pdf

Please consult the publication for further information.
"""

# Copyright (C) 2011
# Authors: Zak Stone <zak@eecs.harvard.edu>
#          Dan Yamins <yamins@mit.edu>
#          James Bergstra <bergstra@rowland.harvard.edu>
#          Nicolas Pinto <pinto@rowland.harvard.edu>
#          Giovani Chiachia <chiachia@rowland.harvard.edu>

# License: Simplified BSD

# XXX: splits (csv-based) for verification and identification tasks (CVPR'11)


import os
from os import path
import shutil
from glob import glob
import hashlib

import larray
from data_home import get_data_home
from utils import download, extract, int_labels
import utils
import utils.image
from utils.image import ImgLoader

from sklearn import cross_validation
import numpy as np

class PubFig83(object):
    """PubFig83 Face Dataset

    Attributes
    ----------
    meta: list of dict
        Metadata associated with the dataset. For each image with index i,
        meta[i] is a dict with keys:
            name: str
                Name of the individual's face in the image.
            filename: str
                Full path to the image.
            gender: str
                'male or 'female'
            id: int
                Identifier of the image.
            sha1: str
                SHA-1 hash of the image.

    Notes
    -----
    If joblib is available, then `meta` will be cached for faster
    processing. To install joblib use 'pip install -U joblib' or
    'easy_install -U joblib'.
    """

    URL = 'http://www.eecs.harvard.edu/~zak/pubfig83/pubfig83_first_draft.tgz'
    SHA1 = '1fd55188bf7d9c5cc9d68baee57aa09c41bd2246'

    _GENDERS = ['male', 'male', 'female', 'female', 'male', 'female', 'male',
                'male', 'female', 'male', 'female', 'female', 'female',
                'female', 'female', 'male', 'male', 'male', 'male', 'male',
                'male', 'male', 'male', 'female', 'female', 'male', 'male',
                'female', 'female', 'male', 'male', 'female', 'female', 'male',
                'male', 'male', 'male', 'female', 'female', 'female', 'female',
                'female', 'male', 'male', 'female', 'female', 'female',
                'female', 'female', 'female', 'male', 'male', 'female',
                'female', 'female', 'male', 'female', 'female', 'male', 'male',
                'female', 'male', 'female', 'female', 'male', 'female',
                'female', 'male', 'male', 'female', 'female', 'male', 'female',
                'female', 'male', 'male', 'male', 'male', 'female', 'female',
                'male', 'male', 'male']

    def __init__(self, meta=None, seed=42, ntrain=90, ntest=10, num_splits=10):

        self.seed = seed
        self.ntrain = ntrain
        self.ntest = ntest
        self.num_splits = num_splits

        if meta is not None:
            self._meta = meta

        self.name = self.__class__.__name__

        try:
            from joblib import Memory
            mem = Memory(cachedir=self.home('cache'))
            self._get_meta = mem.cache(self._get_meta)
        except ImportError:
            pass

    def home(self, *suffix_paths):
        return path.join(get_data_home(), self.name, *suffix_paths)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

    def fetch(self, download_if_missing=True):
        """Download and extract the dataset."""

        home = self.home()

        if not download_if_missing:
            raise IOError("'%s' exists!" % home)

        # download archive
        url = self.URL
        sha1 = self.SHA1
        basename = path.basename(url)
        archive_filename = path.join(home, basename)
        if not path.exists(archive_filename):
            if not download_if_missing:
                return
            if not path.exists(home):
                os.makedirs(home)
            download(url, archive_filename, sha1=sha1)

        # extract it
        if not path.exists(self.home('pubfig83')):
            extract(archive_filename, home, sha1=sha1, verbose=True)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: meta
    # ------------------------------------------------------------------------

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch(download_if_missing=True)
            self._meta = self._get_meta()
        return self._meta

    def _get_meta(self):
        names2 = sorted(os.listdir(self.home('pubfig83')))
        genders = self._GENDERS
        assert len(names2) == len(genders)
        meta = []
        ind = 0
        for gender, name in zip(genders, names2):
            img_filenames = sorted(glob(self.home('pubfig83', name, '*.jpg')))
            for img_filename in img_filenames:
                img_data = open(img_filename, 'rb').read()
                sha1 = hashlib.sha1(img_data).hexdigest()
                meta.append(dict(gender=gender, name=name, id=ind,
                                 filename=img_filename, sha1=sha1))
                ind += 1

        return meta

    @property
    def names(self):
        if not hasattr(self, '_names'):
            self._names = np.array([self.meta[ind]['name'] for ind in xrange(len(self.meta))])
        return self._names

    @property
    def classification_splits(self):
        """
        generates splits and attaches them in the "splits" attribute

        """
        if not hasattr(self, '_classification_splits'):
            seed = self.seed
            ntrain = self.ntrain
            ntest = self.ntest
            num_splits = self.num_splits
            self._classification_splits = self._generate_classification_splits(seed, ntrain,
                                                                        ntest, num_splits)
        return self._classification_splits

    def _generate_classification_splits(self, seed, ntrain, ntest, num_splits):
        meta = self.meta
        ntrain = self.ntrain
        ntest = self.ntest
        rng = np.random.RandomState(seed)
        classification_splits = {}
        
        splits = {}
        for split_id in range(num_splits):
            splits[split_id] = {}
            splits[split_id]['train'] = []
            splits[split_id]['test'] = []
        
        labels = np.unique(self.names)
        for label in labels:
            samples_to_consider = (self.names == label)
            samples_to_consider = np.where(samples_to_consider)[0]
            
            L = len(samples_to_consider)
            assert L >= ntrain + ntest, 'category %s too small' % label
            
            ss = cross_validation.ShuffleSplit(L, 
                                               n_iterations=num_splits, 
                                               test_fraction=(ntest/float(L))-1e-5, #ceil(.) in lib
                                               train_fraction=(ntrain/float(L))+1e-5,
                                               random_state=rng)
            
            for split_id, [train_index, test_index] in enumerate(ss):
                splits[split_id]['train'] += samples_to_consider[train_index].tolist()
                splits[split_id]['test'] += samples_to_consider[test_index].tolist()
                
        return splits

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        if path.isdir(self.home()):
            shutil.rmtree(self.home())

    # ------------------------------------------------------------------------
    # -- Helpers
    # ------------------------------------------------------------------------

    def image_path(self, m):
        return self.home('pubfig83', m['name'], m['filename'])
        #return self.home('pubfig83', m['name'], m['jpgfile'])

    # ------------------------------------------------------------------------
    # -- Standard Tasks
    # ------------------------------------------------------------------------

    def raw_classification_task(self, split=None, split_role=None):
        """
        :param split: an integer from 0 to 9 inclusive.
        :param split_role: either 'train' or 'test'
        
        :returns: either all samples (when split_k=None) or the specific 
                  train/test split
        """

        if split is not None:
            assert split in range(10), ValueError(split)
            assert split_role in ('train', 'test'), ValueError(split_role)                    
            inds = self.classification_splits[split][split_role]
        else:
            inds = range(len(self.meta))            
        names = self.names[inds]
        paths = [self.meta[ind]['filename'] for ind in inds]
        labels = int_labels(names)
        return paths, labels, inds

    def raw_gender_task(self):
        genders = [m['gender'] for m in self.meta]
        paths = [self.image_path(m) for m in self.meta]
        return paths, utils.int_labels(genders)

    def img_classification_task(self, dtype='uint8', split=None, split_role=None):
        img_paths, labels, inds = self.raw_classification_task(split=split, split_role=split_role)
        imgs = larray.lmap(ImgLoader(ndim=3, dtype=dtype, mode='RGB'),
                           img_paths)
        return imgs, labels


# ------------------------------------------------------------------------
# -- Drivers for skdata/bin executables
# ------------------------------------------------------------------------

def main_fetch():
    """compatibility with bin/datasets-fetch"""
    PubFig83.fetch(download_if_missing=True)


def main_show():
    """compatibility with bin/datasets-show"""
    from utils.glviewer import glumpy_viewer
    import larray
    pf = PubFig83()
    names = [m['name'] for m in pf.meta]
    paths = [pf.image_path(m) for m in pf.meta]
    glumpy_viewer(
            img_array=larray.lmap(utils.image.ImgLoader(), paths),
            arrays_to_print=[names])
