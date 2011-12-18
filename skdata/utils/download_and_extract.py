"""Helpers to download and extract archives"""

# Authors: Nicolas Pinto <pinto@rowland.harvard.edu>
#          Nicolas Poilvert <poilvert@rowland.harvard.edu>
#          Daniel Yamins <yamins@mit.edu>
# License: BSD 3 clause

from urllib2 import urlopen
from os import path
import hashlib
import urlparse

import boto

import archive


def verify_sha1(filename, sha1):
    data = open(filename, 'rb').read()
    if sha1 != hashlib.sha1(data).hexdigest():
        raise IOError("File '%s': invalid SHA-1 hash! You may want to delete"
                      "this corrupted file..." % filename)


def download(url, output_filename, sha1=None, verbose=True):
    """Downloads file at `url` and write it in `output_dirname`"""

    page = urlopen(url)
    page_info = page.info()

    output_file = open(output_filename, 'wb+')

    # size of the download unit
    block_size = 2 ** 15
    dl_size = 0

    if verbose:
        print "Downloading '%s' to '%s'" % (url, output_filename)
    # display  progress only if we know the length
    if 'content-length' in page_info and verbose:
        # file size in Kilobytes
        file_size = int(page_info['content-length']) / 1024.
        while True:
            buffer = page.read(block_size)
            if not buffer:
                break
            dl_size += block_size / 1024
            output_file.write(buffer)
            percent = min(100, 100. * dl_size / file_size)
            status = r"Progress: %20d kilobytes [%4.1f%%]" \
                    % (dl_size, percent)
            status = status + chr(8) * (len(status) + 1)
            print status,
        print ''
    else:
        output_file.write(page.read())

    output_file.close()

    if sha1 is not None:
        verify_sha1(output_filename, sha1)


def download_boto(url, credentials, output_filename, sha1=None):
    """Downloads file from S3 via boto at `url` and write it in `output_dirname`"""

    conn = boto.connect_s3(*credentials)
    url = urlparse.urlparse(url)
    bucketname = url.netloc.split('.')[0]
    file = url.path.strip('/')
    bucket = conn.get_bucket(bucketname)
    key = bucket.get_key(file)
    key.get_contents_to_filename(output_filename)

    if sha1 is not None:
        verify_sha1(output_filename, sha1)


def extract(archive_filename, output_dirname, sha1=None, verbose=True):
    """Extracts `archive_filename` in `output_dirname`.

    Supported archives:
    -------------------
    * Zip formats and equivalents: .zip, .egg, .jar
    * Tar and compressed tar formats: .tar, .tar.gz, .tgz, .tar.bz2, .tz2
    """
    if verbose:
        print "Extracting '%s' to '%s'" % (archive_filename, output_dirname)
    if sha1 is not None:
        if verbose:
            print " SHA-1 verification..."
        verify_sha1(archive_filename, sha1)
    archive.extract(archive_filename, output_dirname, verbose=verbose)


def download_and_extract(url, output_dirname, sha1=None, verbose=True):
    """Downloads and extracts archive in `url` into `output_dirname`.

    Note that `output_dirname` has to exist and won't be created by this
    function.
    """
    archive_basename = path.basename(url)
    archive_filename = path.join(output_dirname, archive_basename)
    download(url, archive_filename, sha1=sha1, verbose=verbose)
    extract(archive_filename, output_dirname, sha1=sha1, verbose=verbose)
