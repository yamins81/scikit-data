import os
import gzip
import struct

import Image
import numpy as np
import lockfile

import tabular as tb

import skdata.larray as larray
from skdata.data_home import get_data_home
from skdata.utils.download_and_extract import download, extract


class NORB(object):

    FILES = [('norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat.gz',
              'ba6e05ca90b151997d3fca7082e7a6bd3a470fd4'),
             ('norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat.gz',
              '4a2a765a8a351e0c005aef54cf320ca53b04d50c'),
             ('norb-5x01235x9x18x6x2x108x108-testing-01-info.mat.gz',
              '68c63325de7ac11e630de670dc2a9743d9826478'),
             ('norb-5x01235x9x18x6x2x108x108-testing-02-cat.mat.gz',
              '13d11d472f26492e2ee68cf1cfccc23e143a97cb'),
             ('norb-5x01235x9x18x6x2x108x108-testing-02-dat.mat.gz',
              '79e037950a5423fb126685705b4dfb874ca471f3'),
             ('norb-5x01235x9x18x6x2x108x108-testing-02-info.mat.gz',
              '4431515f880ff20473bed2f6f1ffd00075a7ff40'),
             ('norb-5x46789x9x18x6x2x108x108-training-01-cat.mat.gz',
              'b9ed38a6af5d9287872ab7064c416fd35dcea79b'),
             ('norb-5x46789x9x18x6x2x108x108-training-01-dat.mat.gz',
              '38789c74538b0a3f86a4f854f0f0e3482451fd09'),
             ('norb-5x46789x9x18x6x2x108x108-training-01-info.mat.gz',
              '2e2e2a3eef2e0b70dd889ea84f3ce028869eb8de'),
             ('norb-5x46789x9x18x6x2x108x108-training-02-cat.mat.gz',
              '192e87e661f29890e0536bc6445b8fa43fbad832'),
             ('norb-5x46789x9x18x6x2x108x108-training-02-dat.mat.gz',
              'a559ad82aad672ea281db8c437ea89ca6a76a9fa'),
             ('norb-5x46789x9x18x6x2x108x108-training-02-info.mat.gz',
              '8b6cc488a6707e853d065efd338e0d6541f005eb'),
             ('norb-5x46789x9x18x6x2x108x108-training-03-cat.mat.gz',
              'ca6f995f9f5b39ae614f8a355a70d53bf1ce4dff'),
             ('norb-5x46789x9x18x6x2x108x108-training-03-dat.mat.gz',
              '4318343182078b87af6e93eb7a1a0d63b55a7af4'),
             ('norb-5x46789x9x18x6x2x108x108-training-03-info.mat.gz',
              '09460917f05e90337b110527a3840defa84650b2'),
             ('norb-5x46789x9x18x6x2x108x108-training-04-cat.mat.gz',
              '2353025bfcf02ec6fd91171fcf1f964c2ff1f00b'),
             ('norb-5x46789x9x18x6x2x108x108-training-04-dat.mat.gz',
              '32e193bb7234de588b67592ce04c57cb5b36412a'),
             ('norb-5x46789x9x18x6x2x108x108-training-04-info.mat.gz',
              '16e63a9cec76ce1b96ec7ab28ae01a3d35f6eba9'),
             ('norb-5x46789x9x18x6x2x108x108-training-05-cat.mat.gz',
              '931f63df18ec27e18a3859e35953a6f2a6478b11'),
             ('norb-5x46789x9x18x6x2x108x108-training-05-dat.mat.gz',
              '0669b2942215f05c33d7be07df352d96b5a4bbe3'),
             ('norb-5x46789x9x18x6x2x108x108-training-05-info.mat.gz',
              '75b0ef12a7987e88dfc27ce4abde3b6a6f6ba64a'),
             ('norb-5x46789x9x18x6x2x108x108-training-06-cat.mat.gz',
              '48b91f77aa94dfbd04ccf57b753e4cd42d0631cf'),
             ('norb-5x46789x9x18x6x2x108x108-training-06-dat.mat.gz',
              '779bf6ce2e69cc8cabaf0a1716e6528313c250b0'),
             ('norb-5x46789x9x18x6x2x108x108-training-06-info.mat.gz',
              '89ac793064083fec52182402a66dbe71ffaacebc'),
             ('norb-5x46789x9x18x6x2x108x108-training-07-cat.mat.gz',
              'f29cd673bd9840abf0fecd68a4d328b698f50cfa'),
             ('norb-5x46789x9x18x6x2x108x108-training-07-dat.mat.gz',
              'ef6410b90cae2d11a4609787806ae44cb2bbe719'),
             ('norb-5x46789x9x18x6x2x108x108-training-07-info.mat.gz',
              '265248cad68aabc1dc85eb0b68d390aac080d720'),
             ('norb-5x46789x9x18x6x2x108x108-training-08-cat.mat.gz',
              'd3bc1d9111404d683b8ac6b7adb7e4fa972b765f'),
             ('norb-5x46789x9x18x6x2x108x108-training-08-dat.mat.gz',
              '73302e20486b64824e29a907ec7a353c14c12215'),
             ('norb-5x46789x9x18x6x2x108x108-training-08-info.mat.gz',
              'f3678eca658b6c2484ed9d47ca13210c321a88c3'),
             ('norb-5x46789x9x18x6x2x108x108-training-09-cat.mat.gz',
              '801d3b16f7813a9e383507a933b1d8a292f3722b'),
             ('norb-5x46789x9x18x6x2x108x108-training-09-dat.mat.gz',
              '26ad80a8b99283804ab69207f81ff902343037b3'),
             ('norb-5x46789x9x18x6x2x108x108-training-09-info.mat.gz',
              'a4e404c0fa154a6fdb64b82bd6b3203e824f30e9'),
             ('norb-5x46789x9x18x6x2x108x108-training-10-cat.mat.gz',
              '1496bce011b0435da3f54e50560b66abfda9824e'),
             ('norb-5x46789x9x18x6x2x108x108-training-10-dat.mat.gz',
              'f9d53d3cb99188b085ce304f826c95f1e3dc8e14'),
             ('norb-5x46789x9x18x6x2x108x108-training-10-info.mat.gz',
              '03cb78af70fc588b606b0d4b78d9df4396b096b8')]

    name = 'NORB'

    def __init__(self):
        pass

    def home(self, *suffix_paths):
        return os.path.join(get_data_home(), self.name, *suffix_paths)

    def fetch(self):
        """Download and extract the dataset."""
        home = self.home()
        if not os.path.exists(home):
            os.makedirs(home)
        lock = lockfile.FileLock(home)
        with lock:
            for base, sha1 in self.FILES:
                filename = os.path.join(home, base)
                if not os.path.exists(filename):
                    url = 'http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/' + \
                           base
                    print ('downloading %s' % url)
                    download(url, filename, sha1=sha1, verbose=True)

    @property
    def meta(self):
        """metadata object is a rec array with fields (see meta.dtype)"""
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = self._get_meta()
        return self._meta

    def _get_meta(self):
        catfiles = [x[0] for x in self.FILES if '-cat' in x[0]]
        infofiles = [x[0] for x in self.FILES if '-info' in x[0]]

        categories = []
        series = []
        traintest = []
        infos = []
        fns = []
        for (catfile, infofile) in zip(catfiles, infofiles):
            assert infofile == catfile.replace('-cat', '-info')
            trt = catfile.split('-')[2]
            s = catfile.split('-')[3]
            catfile = self.home(catfile)
            a = gzip.GzipFile(catfile, 'rb').read()[20:]
            cats = struct.unpack('i' * (len(a) / 4), a)
            categories.extend(cats)
            series.extend([s] * len(cats))
            traintest.extend([trt] * len(cats))
            infofile = self.home(infofile)
            a = gzip.GzipFile(infofile, 'rb').read()[20:]
            info = struct.unpack('i' * (len(a) / 4), a)
            info = np.array(info).reshape((29160, 10))
            infos.append(info)
            fns.extend(range(len(cats)))
        infos = np.row_stack(infos)
        objs = [str(c) + '_' + str(o) for c, o in zip(categories, infos[:, 0])]
        cols = [categories, objs, series, traintest, fns] + infos.T.tolist()

        names = ['category', 'obj', 'series', 'traintest', 'fn'] + \
                ['instance', 'elevation', 'azimuth', 'lighting', 'horizontal',
                 'vertical', 'lumination', 'contrast', 'scale', 'rotation']

        return tb.tabarray(columns=cols, names=names)

    @property
    def dat_fhs(self):
        """file handles to underlying image files"""

        if not hasattr(self, '_dat_fhs'):
            datfiles = [x[0] for x in self.FILES if '-dat' in x[0]]
            self._dat_fhs = dict([(df,
                             gzip.GzipFile(self.home(df))) for df in datfiles])
        return self._dat_fhs

    def get_images(self, preproc):
        """returns a pair of larrays, for the left and right images
        respectively"""

        shape = preproc['shape']
        dtype = preproc['dtype']
        normalize = preproc['normalize']
        transpose = preproc.get('transpose', False)
        meta = self.meta
        imagesL = larray.lmap(ImgLoader(self, (108, 108),
                            shape, dtype, 'l', transpose, normalize), meta)
        imagesR = larray.lmap(ImgLoader(self, (108, 108),
                            shape, dtype, 'r', transpose, normalize), meta)
        return imagesL, imagesR


class ImgLoader(object):
    def __init__(self, cls, inshape, shape, dtype, lr, transpose, normalize):
        self.cls = cls
        self.dat_fhs = self.cls.dat_fhs
        self.FILES = self.cls.FILES
        self._inshape = tuple(inshape)
        self.transpose = transpose
        if self.transpose:
            self._shape = tuple(np.array(shape)[list(self.transpose)])
        else:
            self._shape = shape
        self._dtype = dtype
        self._ndim = None if (shape is None) else len(shape)
        assert lr in ['l', 'r']
        self.lr = lr
        self.normalize = normalize

    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self._shape is not None:
            return self._shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self._dtype
        raise AttributeError(attr)

    def __call__(self, m):
        series = m['series']
        trt = m['traintest']
        fn = m['fn']
        imnbytes = (108 ** 2)
        start = 24 + 2 * imnbytes * fn
        end = 24 + 2 * imnbytes * (fn + 1)
        datfile = [x[0] for x in self.FILES if
                                trt + '-' + series + '-' + 'dat' in x[0]][0]
        datfh = self.dat_fhs[datfile]
        datfh.seek(start)
        s = datfh.read(end - start)
        if self.lr == 'l':
            s = s[:imnbytes]
        elif self.lr == 'r':
            s = s[imnbytes:]
        intarr = np.array(struct.unpack('B' * (len(s) / 1), s),
                                           dtype='uint8').reshape((108, 108))
        im = Image.fromarray(intarr, mode='L')
        assert im.size == self._inshape
        if max(im.size) != self._shape[0]:
            m = self._shape[0] / float(max(im.size))
            new_shape = (int(round(im.size[0] * m)),
                         int(round(im.size[1] * m)))
            im = im.resize(new_shape, Image.ANTIALIAS)
        rval = np.asarray(im, self._dtype)
        if self.normalize:
            rval -= rval.mean()
            rval /= max(rval.std(), 1e-3)
        else:
            if 'float' in str(self._dtype):
                rval /= 255.0
        if self.transpose:
            rval = rval.transpose(*tuple(self.transpose))
        assert rval.shape == self._shape, (rval.shape, self._shape)
        return rval
