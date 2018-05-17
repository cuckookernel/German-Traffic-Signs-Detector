# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:19:22 2018

@author: Mateo
"""
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
    #params = model_traits.copy()
    params =  {"mean" :0.0,
                    "sigma" : 0.1,
                    "num_classes" : num_classes,
                    "mode" : mode }

    tf.reset_default_graph()
    img_width, img_height = model_traits["target_size"]
    total_dim = img_width * img_height

    gray_scale_flat = tf_log_reg_layer0( img_width, img_height )
    logits = tf_log_reg_layer1(tf,  gray_scale_flat, total_dim, params)
    tf_log_reg_final( logits )


def tf_log_reg_layer0( tf, img_width, img_height ) :
    #%%
    from tensorflow.contrib.layers import flatten
    images_in = tf.placeholder( dtype=tf.float32,
                                shape=(None, img_width, img_height, 3),
                                name="images_in" )

    gray_scale = tf.reduce_mean( images_in, axis = 3, keepdims=True )

    gray_scale_flat = flatten( gray_scale )
    #%%

    return gray_scale_flat

def tf_log_reg_layer1( tf, gray_scale_flat,  total_dim,params ) :

    num_classes=params["num_classes"]

    W = tf.Variable(tf.truncated_normal(shape = (total_dim, num_classes),
                                        mean = params["mean"],
                                        stddev = params["sigma"]))
    b = tf.Variable(tf.zeros(num_classes))

    logits = tf.matmul( gray_scale_flat, W ) + b

    return logits

def tf_log_reg_final( tf, logits, params ) :
    #%%
    target = tf.placeholder( tf.int32, (None),  name="target" )

    one_hot_y = tf.one_hot(target, params["num_classes"])
    cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                           labels=tf.stop_gradient(one_hot_y))

    tu.compute_accuracy( tf, cross_ent, one_hot_y, logits, params )
    #%%

def test_build_log_reg() :
    #%%
    from  model_traits import MODEL_TRAITS

    #%%
