# -*- coding: utf-8 -*-
"""
Created on Mon May 14 17:35:17 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""

import tensorflow as tf

def main() :
    #%%
    build_lenet_graph( model_traits, num_classes=42, mode='train' )
    #%%
    x = tf.placeholder( dtype=tf.float32, shape=(None,8) )
    y = tf.placeholder( dtype=tf.float32, shape=(None,5) )
    #%%
    w = tf.Variable( tf.truncated_normal(shape  = [8,5],
                                         mean   = 0.0,
                                         stddev = 0.01) )
    #%%
    prod =  tf.matmul(  x , w )
    #%%
    xmat = tf.reshape( x, shape=[-1,8,1])
    ones = tf.ones( shape=[1,5] )
    #%%
    xmat -w
    #%%

    xmat * ones

    #%%
    tf.tensordot( xmat, ones, axes=(2,0))
    #%%
    diff = xmat * ones - w
    diff
    #%%
    tf.square( diff )
    #%%
    diff = w - x
    #%%