# -*- coding: utf-8 -*-
"""
Created on Sat May 12 09:35:32 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""

import os
import logging
import skimage.io
import skimage.transform
import numpy as np
from sklearn.externals import joblib

#pylint: disable=C0326
log = logging.getLogger( __name__ ).log #pylint: disable=C0103

def make_4d_arrays( images_dir, target_size=(32,32), rescale_mode='max',
                    order=3, verbose=0, **kwargs ) :
    """This function does the following:

    1. Reads images from directory
    2. resizes each to target_size
    3. optionally rescales them
    4. extracts the ground truth label from the filename for each of them
       (it's assumed that images are labeled as '00123_23.ppm'
        with 23 being the class_id)
    5. collects results in np.array of shape
        (num_images, target_size[0], target_size[1], 3 )
        assuming num_channels is 3.
    6. returns this array as well as the corresponding (parallel) np.array
       of ground_truths of shape (num_images)

    The order argument is passed directly to skimage.transform.resize as well
    as **kwargs, see help for that function.

    if rescaled_mode equals 'max' then, after resizing the image is
    rescaled as: img_resized /= img_resized.max() so that
    the resulting pixel values should go all the way to 1.0.
    For any other value of rescaled_model

    Any value for rescale_mode different from max
    """

    arr_3d_list = []
    gt_list = []

    resize = skimage.transform.resize

    for i, img_fn in enumerate( os.listdir( images_dir ) ) :
        if img_fn.startswith( '.') :
            continue

        if i % 20 == 0 and verbose > 0 :
            print( "resizing img %i" % i )

        img = skimage.io.imread( images_dir + '/' + img_fn )

        # Extract the ground truth label as an int
        fn_pieces = img_fn.split('.')[0].split('_')
        assert len(fn_pieces) == 2, \
        "img_fn=%s does not have the format <something>_<ground_truth>.<ext>"

        gt_list.append( int( fn_pieces[1] ) )

        img_resized = resize( img, target_size,
                              mode='reflect',
                              order=order, **kwargs )


        if rescale_mode == 'max' :
            imax = img_resized.max()

        elif rescale_mode == 'max_q' :
            wi, he = target_size
            imax = img_resized[ he//4: 3*he//4, wi//4 : 3*wi//4, : ].max()

        elif rescale_mode == '' :
            imax = 1.0

        assert imax <= 1.0,\
               "imax = %.3f : Using wrong version of skimage?" % imax

        img_resized /= imax
        img_resized = np.clip( img_resized, 0.0, 1.0 )

        arr_3d_list.append( img_resized )

    return  np.array( arr_3d_list ), np.array( gt_list )

def save_sklearn_model( model_obj, model_name, logl=100 ) :
    """save a sklearn model to models/..."""
    model_path = get_save_dir( model_name ) + "%s.pkl" % model_name
    log(logl, "Saving model to: %s" , model_path)

    joblib.dump( model_obj, model_path )


def load_sklearn_model( model_name, logl=100 ) :
    """loads a pickled sklearn model"""

    model_path =get_save_dir( model_name ) + '%s.pkl' % model_name
    log(logl, model_path )

    model = joblib.load( model_path )
    return model


def save_tf_model( sess, model_name, logl=100 ) :
    """save a tensorflow model to models/..."""
    import tensorflow as tf
    saver = tf.train.Saver()
    model_path = get_save_dir( model_name )

    log(logl, "Saving tensorflow model t: %s", model_path)
    saver.save(sess, model_path )

def get_save_dir( model_name, create=True) :
    """get a path under models"""
    model_dir = "models/" + model_name
    if not os.path.exists( model_dir ) :
        os.mkdir( model_dir )

    saved_dir = model_dir + "/saved/"
    if not os.path.exists( saved_dir ) :
        if create :
            os.mkdir( saved_dir )
        else :
            assert False, "Path doesn't exist: " + saved_dir

    return saved_dir
