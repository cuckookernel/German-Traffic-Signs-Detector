# -*- coding: utf-8 -*-
"""
Created on Sat May 12 09:23:22 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""
import os
from os import path
from pprint import pformat
import random
import shutil
import zipfile

import skimage
import skimage.io
import pandas as pd

#pylint: disable=C0326

def do_download(  data_url, dest_dir,
                  skip_download=False,
                  skip_decompress=False,
                  remove_zip_after=False  ) :
    """Download a zipfile from DATA_URL decompress it and put results
    in dest_dir
    """
    zip_file_path = dest_dir + '/tmp_file.zip'

  #%%
    if not skip_download :
        import urllib.request
        response = urllib.request.urlopen(data_url)

        chunk_size = 1024 * 64
        read_bytes  = 0
        #%%
        with open( zip_file_path, 'wb') as f_out:
            for chunk in read_in_chunks( response, chunk_size ) :
                read_bytes += len( chunk )
                print( "%d bytes read" % read_bytes )
                f_out.write( chunk )
    else :
        print( "skipping download" )

    if not skip_decompress :
        print( "Decompressing tmp zip file: " + zip_file_path  )
        zip_ref = zipfile.ZipFile(zip_file_path, 'r')
        #%%
        zip_ref.extractall( dest_dir )
        zip_ref.close()
        print( "Done decompressing.\nListing destination dir: "  + dest_dir )
        print( pformat( os.listdir( dest_dir ) ) )
    else :
        print( "skipping decompress" )

    if remove_zip_after :
        os.remove( zip_file_path )


    print('making train test dirs and distributing images in them')
    make_train_test_dirs( base_dir = dest_dir,
                          orig_data_subdir = 'FullIJCNN2013',
                          max_id=42,
                          train_prop=0.8,
                          seed=1337)


def test() :
    """Quick test"""
    #%%
    do_download(  data_url = None,
                  dest_dir = 'images/' ,
                  skip_download=True,
                  skip_decompress=True,
                  remove_zip_after=False  )
    #%%

def read_in_chunks(infile, chunk_size=1024*64):
    """Read from a binary file (which could be an http) response in chunks

    Freely borrowed from:
    https://stackoverflow.com/questions/4566498/python-file-iterator-over-a-binary-file-with-newer-idiom
    """
    while True:
        chunk = infile.read(chunk_size)
        if chunk:
            yield chunk
        else:
            # The chunk was empty, which means we're at the end
            # of the file
            return

def make_train_test_dirs(  base_dir = 'images/',
                           orig_data_subdir = 'FullIJCNN2013',
                           max_id=42,
                           train_prop=0.8,
                           seed=1337) :

    """This function does the following:
    1. make train and test subdirectories under base_dir, if they aren't
    already there
    2. walk through origd_data_subdir and copy distribute images in it
    to either train or test subdirs ccording to the probability train_prop
    3. collect stats about the images, essentially shape and class_id and
    return those as data_set
    """

    train_dir = base_dir + 'train'
    test_dir = base_dir + 'test'

    if not path.exists( train_dir ) :
        os.mkdir( train_dir )

    if not path.exists( test_dir ) :
        os.mkdir( test_dir )

    img_stats = []

    random.seed( seed )
    cnt = 0
    for i in range(max_id + 1) :
        orig_dir = '%s/%s/%02d' % (base_dir, orig_data_subdir, i)

        for fname in os.listdir( orig_dir ) :
            cnt += 1
            orig_path = orig_dir + '/' + fname
            img_st, dest_path = copy_one_image( i,  orig_path,
                                                train_dir, test_dir,
                                                train_prop )
            img_stats.append( img_st )

            print( "%d %s => %s" % (cnt, orig_path, dest_path) )


    return pd.DataFrame( img_stats )


def copy_one_image( class_id, orig_path, train_dir, test_dir, train_prop  ) :
    """Copy one image with to either  train_dir or test_dir, decided
    at random based on train_prop"""

    img = skimage.io.imread(orig_path)

    img_st = { 'class_id' : class_id,
               'orig_path' : orig_path,
               'shape0' : img.shape[0],
               'shape1' : img.shape[1],
               'shape2' : img.shape[2]  }

    dest_dir = train_dir if random.random() < train_prop else test_dir
    fname = os.path.basename( orig_path )

    dest_path = "%s/%s_%02d.ppm" % (dest_dir, fname[:-4], class_id )

    shutil.copy( orig_path, dest_path )
    return img_st, dest_path
