# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:02:53 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""
#pylint: disable=C0326

import logging
import train_utils as tu
#TODO: lo de no backpropagar...
logging.basicConfig( level = 2 )

LOGGER = logging.getLogger( __name__ )
log = LOGGER.log #pylint: disable=C0103

def train_lenet( model_traits, data, logl=0, save_model=True ):
    """Train a lenet type NN for image classification"""

    log( logl, "train_lenet : importing tensorflow" )
    import tensorflow as tf

    #%%
    batch_size = model_traits["batch_size"]
    epochs     = model_traits["epochs"]

    build_lenet_graph( model_traits["target_size"],
                       drop_out_rate=model_traits["drop_out_rate"],
                       num_classes=42,
                       learning_rate=model_traits["learning_rate"] )
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

            log( logl, "EPOCH %d: train accuracy = %.4f validation Accuracy = %.4f",
                 i+1,  train_accu, valid_accu)

        if save_model :
            tu.save_tf_model( sess, model_traits["model_name"], logl=logl )

    return { "accuracy" : train_accu_log[-1] }


def test_lenet( model_traits, data, return_inferred=False, logl=0 ):
    """Train a lenet type NN for image classification"""

    log(logl, "test_lenet : importing tensorflow" )
    import tensorflow as tf


    batch_size = model_traits["batch_size"]

    build_lenet_graph( model_traits["target_size"],
                       drop_out_rate=model_traits["drop_out_rate"],
                       num_classes=42,
                       learning_rate=model_traits["learning_rate"],
                       mode=tf.estimator.ModeKeys.EVAL )

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())

        log( logl, "Testing...\n")
        saver.restore(sess, tu.get_save_dir(model_traits["model_name"]) )

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
    print( "do_valid", do_valid, list(data.keys()) )

    for offset in range(0, len(train_x), batch_size):
        end = offset + batch_size
        batch_x, batch_y = train_x[offset:end], train_y[offset:end]

        sess.run( train_op, feed_dict={ images_in: batch_x,
                                        target   : batch_y})

    if do_valid :
        valid_accu = tf_evaluate( data["test_4d"], data["test_y"], batch_size )
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


def build_lenet_graph( target_size, drop_out_rate,
                       num_classes = 42,
                       learning_rate=0.001,
                       mode='train' ) :
    """Builds LeNet Neural network"""

    import tensorflow as tf
    from tensorflow.contrib.layers import flatten

    mean = 0
    sigma = 0.1
    num_classes = 42 # used in last layer

    tf.reset_default_graph()
    img_width, img_height = target_size

    images_in = tf.placeholder( dtype=tf.float32,
                                shape=(None, img_width, img_height, 3),
                                name="images_in" )
    target = tf.placeholder( tf.int32, (None),
                             name="target" )

    one_hot_y = tf.one_hot(target, num_classes)

    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,3,6],
                                              mean = mean, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))

    conv1 = tf.nn.conv2d(images_in, conv1_w, strides = [1,1,1,1],
                         padding = 'VALID') + conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)

    pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1],
                            padding = 'VALID')

    # TLayer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16],
                                              mean=mean, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))

    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1],
                         padding = 'VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)

    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1],
                            padding = 'VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten( pool_2 )


    #  Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120),
                                            mean = mean, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    fc1_do = tf.layers.dropout( inputs=fc1,
                                rate= drop_out_rate,
                                training= mode == tf.estimator.ModeKeys.TRAIN )

    #  Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84),
                                            mean = mean, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1_do,fc2_w) + fc2_b
    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84, num_classes),
                                            mean = mean , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(num_classes))
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    optimizer.minimize( loss_operation, name="train_op" )

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accu" )

    return {"target" : target }


def tf_evaluate(x_data0, y_data, batch_size, return_inferred=False):
    """Evaluate accuracy in tf.session"""
    import tensorflow as tf
    #%%
    images_in = tf_node( "images_in:0" )
    target    = tf_node( "target:0" )
    accu      = tf_node( "accu:0" )
    #%%
    num_examples = len( x_data0 )

    x_data = normalize_imgs( x_data0 )
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

def testing() :
    """Quick interctive tests"""
    #%%
    from model_traits import MODEL_TRAITS

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

    train_results = train_lenet( MODEL_TRAITS["model2"], data, logl = 1 )
    #%%
    # accu = train_logistic( train_4d, train_gt, verbose=1 )
    #%%
    return   train_results

def normalize_imgs( imgs_4d ) :
    """Substract mean and divide by std for each image"""
    mean = imgs_4d.mean( axis=(1,2,3), keepdims=True )
    std  = imgs_4d.std( axis=(1,2,3), keepdims=True )

    return (imgs_4d - mean)/ std
