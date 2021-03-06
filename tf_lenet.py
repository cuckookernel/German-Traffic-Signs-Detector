# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:02:53 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""
#pylint: disable=C0326

import logging
import train_utils as tu

logging.basicConfig( level = 2 )

LOGGER = logging.getLogger( __name__ )
log = LOGGER.log #pylint: disable=C0103


class LenetTrainerTester( tu.TrainerTester ) :
    """Class for training and testing a Lenet model constructed with tensorflow"""
    def train( self, data, logl=0, save_model=True ):
        """Train a lenet type NN for image classification"""
        return tu.train_tf( self.model_traits, data,
                            build_graph=self.build_lenet_graph,
                            logl=logl,
                            save_model=save_model )


    def test(  self, data, return_inferred=False, logl=0 ):
        """Test a LeNet type NN for image classification"""
        return tu.test_tf( self.model_traits, data,
                           build_graph=self.build_lenet_graph,
                           return_inferred=return_inferred,
                           logl=logl )


    def build_lenet_graph( self, num_classes, mode ) :
        """Builds LeNet Neural network"""
        #pylint: disable=R0914

        import tensorflow as tf

        m_traits = self.model_traits

        params = m_traits.copy()
        params.update( {"mean" :0.0,
                        "sigma" : 0.1,
                        "num_classes" : num_classes,
                        "mode" : mode })

        tf.reset_default_graph()
        img_width, img_height = m_traits["target_size"]

        images_in = tf.placeholder( dtype=tf.float32,
                                    shape=(None, img_width, img_height, 3),
                                    name="images_in" )

        if m_traits["net_version"] == "orig" :
            non_linear_fun = lambda tensor : 1.7159 * tf.tanh( tensor )
            fc3_fun = fc3_orig
        elif  m_traits["net_version"] == "v1" :
            non_linear_fun = tf.nn.relu
            fc3_fun = fc3_v1
        else:
            raise NotImplementedError("invalid version: %s" % m_traits["net_version"])

        pool_1 = layer_c1( tf, images_in, non_linear_fun, params )

        pool_2 = layer_c2( tf, pool_1, non_linear_fun, params )

        fc2 = fully_connected( tf, pool_2, non_linear_fun, params )

        target = tf.placeholder( tf.int32, (None), name="target" )

        one_hot_y = tf.stop_gradient( tf.one_hot(target, num_classes) )

        logits, cross_entropy = fc3_fun( tf, fc2, one_hot_y, params)

        tu.compute_accuracy( tf, cross_entropy, one_hot_y, logits, params )


def layer_c1( tf, images_in, non_linear_fun, params ) : #pylint: disable=C0103
    """Convolutional layer C1 + pooling"""
    if params["drop_colors"] :
        images_in = tf.reduce_mean( images_in, axis = 3, keepdims=True )

        conv1_w = tf.Variable(tf.truncated_normal(shape  = [5,5,1,6],
                                                  mean   = params["mean"],
                                                  stddev = params["sigma"]))
        conv1_b = tf.Variable(tf.zeros(6))

    else :
        conv1_w = tf.Variable(tf.truncated_normal(shape  = [5,5,3,6],
                                                  mean   = params["mean"],
                                                  stddev = params["sigma"]))
        conv1_b = tf.Variable(tf.zeros(6))

    conv1 = tf.nn.conv2d(images_in, conv1_w, strides = [1,1,1,1],
                         padding = 'VALID') + conv1_b
    # Activation.
    conv1 = non_linear_fun(conv1)

    pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1],
                            padding = 'VALID')
    return pool_1

def layer_c2( tf, pool_1, non_linear_fun, params ) : #pylint: disable=C0103
    """Convolutional layer C1 + pooling"""
    # TLayer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape  = [5,5,6,16],
                                              mean   = params["mean"],
                                              stddev = params["sigma"]))
    conv2_b = tf.Variable(tf.zeros(16))

    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1],
                         padding = 'VALID') + conv2_b
    # Activation.
    conv2 = non_linear_fun(conv2)

    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1],
                            padding = 'VALID')

    return pool_2
    # Flatten. Input = 5x5x16. Output = 400.

def fully_connected( tf, pool_2, non_linear_fun, params ) : #pylint: disable=C0103
    """Fully connected layers FC1, FC2"""
    from tensorflow.contrib.layers import flatten

    fc1 = flatten( pool_2 )
    #  Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400, 120),
                                            mean = params["mean"],
                                            stddev = params["sigma"]))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul( fc1, fc1_w ) + fc1_b

    # Activation.
    fc1 = non_linear_fun( fc1 )

    fc1_do = tf.layers.dropout( inputs=fc1,
                                rate= params["dropout_rate"],
                                training= params["mode"] == 'train' )

    #  Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape  = (120,84),
                                            mean   = params["mean"],
                                            stddev = params["sigma"] ))
    fc2_b = tf.Variable(tf.zeros(84))

    fc2 = tf.matmul( fc1_do, fc2_w ) + fc2_b

    # Activation.
    fc2 = non_linear_fun( fc2 )

    return fc2



def fc3_orig( tf, fc2, one_hot_y, params ) : #pylint: disable=C0103
    """Final layer with Gaussian connections following original paper"""

    ncls = params["num_classes"]
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84, ncls ),
                                            mean = params["mean"] ,
                                            stddev = params["sigma"]),
                        name="fc3_w")

    ## Gaussian connections!
    fc2_mat = tf.reshape( fc2, [-1, 84, 1] )
    #print( "fc2=", fc2)
    #print( "fc2_mat=", fc2_mat)
    #print( "fc3_w=", fc3_w)
    #ones = tf.ones( [nc])
    logits = - tf.reduce_sum( tf.square(fc2_mat - fc3_w ), axis=1)
    #print( "logits=", logits)
    #tf.nn.softmax()
    cross_ent_fun = tf.nn.softmax_cross_entropy_with_logits_v2
    cross_ent = cross_ent_fun(logits=logits, labels=one_hot_y )

    return logits, cross_ent

def fc3_v1( tf, fc2, one_hot_y, params ) : #pylint: disable=C0103
    """Final layer"""

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84, params["num_classes"]),
                                            mean = params["mean"] ,
                                            stddev = params["sigma"]),
                        name="fc3_w")

    fc3_b = tf.Variable(tf.zeros(params["num_classes"]))
    logits = tf.add( tf.matmul(fc2, fc3_w) , fc3_b, name="logits")
    cross_ent_fun = tf.nn.softmax_cross_entropy_with_logits_v2
    cross_ent = cross_ent_fun(logits=logits, labels=one_hot_y )

    return logits, cross_ent

def testing() :
    """Quick interctive tests"""
    #%%
    from model_traits import MODEL_TRAITS
    #%%

    images_dir = "images/train"
    target_size = (32,32)

    #%%
    train_4d, train_gt = tu.make_4d_arrays( images_dir=images_dir,
                                            target_size=target_size)
        #%%
    test_4d, test_gt = tu.make_4d_arrays( images_dir="images/test",
                                          target_size=target_size)
    #%%
    #model_traits = MODEL_TRAITS["model2"]
    data = { "train_4d" : train_4d,  "train_gt" : train_gt,
             "test_4d" : test_4d, "test_gt" : test_gt}

    lenet_tt = LenetTrainerTester( MODEL_TRAITS["model3"] )

    train_results = lenet_tt.train( data, logl = 1 )
    #%%
    model_traits = MODEL_TRAITS["model2"]
    build_lenet_graph( model_traits, num_classes=42, mode='train' )
    #%%
    # accu = train_logistic( train_4d, train_gt, verbose=1 )
    #%%
    return   train_results
