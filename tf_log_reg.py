# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:19:22 2018

@author: Mateo
"""
#pylint: disable=C0326

import train_utils as tu


def train_log_reg( model_traits, data, logl=0, save_model=True ):
    """Train a lenet type NN for image classification"""
    return tu.train_tf( model_traits, data,
                        build_graph=build_log_reg,
                        logl=logl,
                        save_model=save_model )

def test_log_reg(  model_traits, data, return_inferred=False, logl=0 ):
    """Test a logistic regression shallow NN for image classification"""
    return tu.test_tf( model_traits, data,
                       build_graph=build_log_reg,
                       return_inferred=return_inferred,
                       logl=logl )

def build_log_reg( model_traits, num_classes, mode ) :
    """Builds LeNet Neural network"""
    #%%
    import tensorflow as tf

    #%%
    params = model_traits.copy()
    params.update({ "mean" :0.0,
                    "sigma" : 0.1,
                    "num_classes" : num_classes,
                    "mode" : mode } )

    tf.reset_default_graph()
    img_width, img_height = model_traits["target_size"]
    total_dim = img_width * img_height

    gray_scale_flat = tf_log_reg_layer0( tf, img_width, img_height )
    logits = tf_log_reg_layer1(tf,  gray_scale_flat, total_dim, params)
    tf_log_reg_final( tf, logits, params )


def tf_log_reg_layer0( tsf, img_width, img_height ) :
    """Input layer plus conversion to gray-scale and flattening
    of matrices into vectors"""

    from tensorflow.contrib.layers import flatten
    images_in = tsf.placeholder( dtype=tsf.float32,
                                 shape=(None, img_width, img_height, 3),
                                 name="images_in" )

    gray_scale = tsf.reduce_mean( images_in, axis = 3, keepdims=True )

    gray_scale_flat = flatten( gray_scale )
    #%%

    return gray_scale_flat

def tf_log_reg_layer1( tsf, gray_scale_flat,  total_dim,params ) :
    """compute logits as logits = x . W  + b with W and b being the
    learnable parameters"""

    num_classes = params["num_classes"]

    mat_w = tsf.Variable(tsf.truncated_normal(shape = (total_dim, num_classes),
                                              mean = params["mean"],
                                              stddev = params["sigma"]))
    vec_b = tsf.Variable(tsf.zeros(num_classes))

    logits = tsf.matmul( gray_scale_flat, mat_w ) + vec_b

    return logits

def tf_log_reg_final( tsf, logits, params ) :
    """Compute cross_entropy (to be optimized) and accuracy, for reporting """
    #%%
    target = tsf.placeholder( tsf.int32, (None),  name="target" )

    one_hot_y = tsf.one_hot(target, params["num_classes"])
    cross_ent = tsf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                            labels=tsf.stop_gradient(one_hot_y))

    tu.compute_accuracy( tsf, cross_ent, one_hot_y, logits, params )
    #%%
