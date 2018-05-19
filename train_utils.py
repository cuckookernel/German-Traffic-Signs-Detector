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

class TrainerTester :
    """Abstract class for doing training, testing and saving a model """
    def __init__(self, model_traits ) :
        """the constructor"""
        self.model_traits = model_traits

    def train( self, data, logl=0, save_model=True ) :
        """train a model"""
        raise NotImplementedError("Concrete subclasses need to implement this!")

    def test( self, data, return_inferred=False, logl=0 ) :
        """Test model"""
        raise NotImplementedError("Concrete subclasses need to implement this!")


def make_4d_arrays( images_dir, target_size=(32,32), rescale_mode='max',
                    order=3, verbose=0, **kwargs ) :
    """This function does the following:

    1. Reads images from directory
       (IMPORTANT: images are assumed to be one level deep,
                   i.e the directory structure is flat.
        Files whoe names start with '.' are ignored)
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

    for i, img_fn in enumerate( os.listdir( images_dir ) ) :

        if img_fn.startswith( '.') :
            continue

        if i % 20 == 0 and verbose > 0 :
            print( "resizing img %i" % i )

        image_path = images_dir + '/' + img_fn
        img_resized, class_ = resize_one( image_path, rescale_mode,
                                          target_size, order, **kwargs )

        gt_list.append( class_ )
        arr_3d_list.append( img_resized )


    return  np.array( arr_3d_list ), np.array( gt_list )


def resize_one( image_path, rescale_mode, target_size, order, **kwargs ) :
    """Load an image from a path, resize it and return it as
    a np.array along with its class_ extracted from filename """

    resize = skimage.transform.resize

    img = skimage.io.imread( image_path )

    # Extract the ground truth label as an int
    fn_pieces = os.path.basename( image_path ).split('.')[0].split('_')

    assert len(fn_pieces) == 2, \
    "img_fn=%s does not have the format <something>_<ground_truth>.<ext>"

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

    class_ = int( fn_pieces[1] )

    return img_resized, class_

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


def train_tf( model_traits, data, build_graph, logl=0,
              save_model=True, log_every=5 ) :
    """Generic function for traing a tf-based model"""

    log( logl, "train_lenet : importing tensorflow" )
    import tensorflow as tf

    #%%
    batch_size = model_traits["batch_size"]
    epochs     = model_traits["epochs"]

    build_graph( num_classes=42, mode='train')
    #%%
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        log( logl, "Training...\n")

        valid_accu_log = []
        train_accu_log = []

        for i in range(epochs):
            # X_train, y_train = shuffle(X_train, y_train)

            valid_accu, train_accu = run_one_epoch( sess, batch_size, data )
            valid_accu_log.append( valid_accu )
            train_accu_log.append( train_accu )

            if i % log_every== 0 :
                log( logl, "EPOCH %d: train accuracy = %.4f validation Accuracy = %.4f",
                     i+1,  train_accu, valid_accu)

        if save_model :
            save_tf_model( sess, model_traits["model_name"], logl=logl )

    return { "accuracy" : train_accu_log[-1] }


def test_tf( model_traits, data, build_graph, return_inferred=False, logl=0 ):
    """Generic function for testing a tf-based model"""

    log(logl, "test_lenet : importing tensorflow" )
    import tensorflow as tf

    batch_size = model_traits["batch_size"]

    build_graph( num_classes=42, mode='eval' )

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())

        log( logl, "Testing...\n")
        saver.restore(sess, get_save_dir(model_traits["model_name"],
                                         create=False) )

        result = tf_evaluate( data["test_4d"], data["test_gt"], batch_size,
                              return_inferred=return_inferred )

    return result

def run_one_epoch( sess, batch_size, data) :
    """ run one epoch of training """

    #%%
    images_in = tf_node("images_in:0")
    target    = tf_node("target:0")
    train_op  = tf_op("train_op")
    #%%
    train_x = normalize_imgs( data["train_4d"] )
    train_y = data["train_gt"]

    do_valid = "test_4d" in data
    #print( "do_valid", do_valid, list(data.keys()) )

    for offset in range(0, len(train_x), batch_size):
        end = offset + batch_size
        batch_x, batch_y = train_x[offset:end], train_y[offset:end]

        sess.run( train_op, feed_dict={ images_in: batch_x,
                                        target   : batch_y})

    if do_valid :
        valid_accu = tf_evaluate( data["test_4d"], data["test_gt"], batch_size)
    else :
        valid_accu = float("nan")

    train_accu = tf_evaluate( train_x, train_y, batch_size )

    return valid_accu, train_accu

def tf_node(name) :
    """Get a node (tensor) in the graph by name.
    The name is usually <name>:<idx>"""
    import tensorflow as tf
    return tf.get_default_graph().get_tensor_by_name( name )

def tf_op(name) :
    """Get an op (operation) in the graph by name.
    The name is usually <name>:<idx>"""

    import tensorflow as tf
    return tf.get_default_graph().get_operation_by_name( name )

def normalize_imgs( imgs_4d ) :
    """Substract mean and divide by std for each image"""
    mean = imgs_4d.mean( axis=(1,2,3), keepdims=True )
    std  = imgs_4d.std( axis=(1,2,3), keepdims=True )

    return (imgs_4d - mean)/ std

def tf_evaluate(x_data, y_data, batch_size, return_inferred=False):
    """Evaluate accuracy in tf.session"""
    import tensorflow as tf
    #%%
    images_in = tf_node( "images_in:0" )
    target    = tf_node( "target:0" )
    accu      = tf_node( "accu:0" )
    #%%
    num_examples = len( x_data )

    x_data = normalize_imgs( x_data )
    total_accuracy = 0
    inferred = []
    sess = tf.get_default_session()

    for offset in range(0, num_examples, batch_size):

        batch_x = x_data[offset:offset + batch_size]
        batch_y = y_data[offset:offset + batch_size]

        accuracy, target_v = sess.run( [accu, target],
                                       feed_dict={ images_in : batch_x,
                                                   target    : batch_y } )
        total_accuracy += (accuracy * len(batch_x))
        inferred += list( target_v )

    return ( total_accuracy / num_examples if not return_inferred
             else inferred )


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

def compute_accuracy( tf, cross_entropy, one_hot_y, logits, params ) : #pylint: disable=C0103
    """Accuracy"""

    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = params["learning_rate"])
    optimizer.minimize( loss_operation, name="train_op" )

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accu" )
