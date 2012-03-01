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

    def __init__(self, meta=None):
        if meta is not None:
            self._meta = meta
        self.name = self.__class__.__name__

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
            self._classification_splits = self._generate_classification_splits()
        return self._classification_splits

    def _generate_classification_splits(self):
        meta = self.meta
        rng = np.random.RandomState(0)
        classification_splits = {}
        
        splits = {}
        labels = np.unique(self.names)
        for label in labels:
            samples_to_consider = (self.names == label)
            samples_to_consider = np.where(samples_to_consider)[0]
            assert len(samples_to_consider) >= 100
            p = rng.permutation(len(samples_to_consider))
            if 'Test' not in splits:
                splits['Test'] = []
            splits['Test'].extend(samples_to_consider[p[:10]])
            remainder = samples_to_consider[p[10:]]
            for _ind in range(5):
                p = rng.permutation(len(remainder))
                if 'Train%d' % _ind not in splits:
                    splits['Train%d' % _ind] = []
                splits['Train%d' % _ind].extend(remainder[p[:80]].copy())
                if 'Validate%d' % _ind not in splits:
                    splits['Validate%d' % _ind] = []
                splits['Validate%d' % _ind].extend(remainder[p[80:90]].copy())

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

    def raw_classification_task(self, split=None):
        """
        :param split: an integer from 0 to 9 inclusive.
        :param split_role: either 'train' or 'test'
        
        :returns: either all samples (when split_k=None) or the specific 
                  train/test split
        """

        if split is not None:                 
            inds = self.classification_splits[split]
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

    def img_classification_task(self, dtype='uint8', split=None):
        img_paths, labels, inds = self.raw_classification_task(split=split)
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
