# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:19:22 2018

@author: Mateo
"""
#pylint: disable=C0326

import train_utils as tu


class LRTrainerTester(tu.TrainerTester) :
    """Train and test a logistic regression model with dropout
    using tensorflow"""

    def __init__(self, model_traits ) :
        # We import tf here and bind it to an objects attribute to avoid
        # paying for the high price of this import for all other models....
        import tensorflow as tf

        super().__init__( model_traits )
        self.params = self.model_traits.copy()
        self.tf = tf #pylint: disable=C0103

    def train( self, data, logl=0, save_model=True ):
        """Train a lenet type NN for image classification"""
        return tu.train_tf( self.model_traits, data,
                            build_graph=self.build_log_reg,
                            logl=logl,
                            save_model=save_model )

    def test(  self, data, return_inferred=False, logl=0 ):
        """Test a logistic regression shallow NN for image classification"""
        return tu.test_tf( self.model_traits, data,
                           build_graph=self.build_log_reg,
                           return_inferred=return_inferred,
                           logl=logl )

    def build_log_reg( self, num_classes, mode ) :
        """Builds LeNet Neural network"""

        self.params.update({ "mean" :0.0,
                             "sigma" : 0.1,
                             "num_classes" : num_classes,
                             "mode" : mode } )

        self.tf.reset_default_graph()
        img_width, img_height = self.model_traits["target_size"]
        total_dim = img_width * img_height

        gray_scale_flat = self.tf_log_reg_layer0( img_width, img_height )
        logits = self.tf_log_reg_layer1(gray_scale_flat, total_dim)
        self.tf_log_reg_final( logits  )


    def tf_log_reg_layer0( self, img_width, img_height ) :
        """Input layer plus conversion to gray-scale and flattening
        of matrices into vectors"""

        from tensorflow.contrib.layers import flatten
        images_in = self.tf.placeholder( dtype=self.tf.float32,
                                         shape=(None, img_width, img_height, 3),
                                         name="images_in" )

        gray_scale = self.tf.reduce_mean( images_in, axis = 3, keepdims=True )

        gray_scale_flat = flatten( gray_scale )
    #%%

        return gray_scale_flat

    def tf_log_reg_layer1( self, gray_scale_flat, total_dim ) :
        """compute logits as logits = x . W  + b with W and b being the
        learnable parameters"""

        num_classes = self.params["num_classes"]

        var_init = self.tf.truncated_normal( shape = (total_dim, num_classes),
                                             mean = self.params["mean"],
                                             stddev = self.params["sigma"])
        mat_w = self.tf.Variable(var_init)
        vec_b = self.tf.Variable(self.tf.zeros(num_classes))

        logits = self.tf.matmul( gray_scale_flat, mat_w ) + vec_b

        return logits

    def tf_log_reg_final( self, logits ) :
        """Compute cross_entropy (to be optimized) and accuracy, for reporting """
        #%%
        target = self.tf.placeholder( self.tf.int32, (None),  name="target" )

        one_hot_y = self.tf.one_hot(target, self.params["num_classes"])

        cross_ent_fun = self.tf.nn.softmax_cross_entropy_with_logits_v2
        cross_ent = cross_ent_fun(logits=logits,
                                  labels=self.tf.stop_gradient(one_hot_y))

        tu.compute_accuracy( self.tf, cross_ent, one_hot_y, logits, self.params )
        #%%
