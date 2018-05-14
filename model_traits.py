# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:38:14 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""

#pylint: disable=C0326
import skl_logistic
import tf_lenet

MODEL_TRAITS = [

    { "model_name" : "model1",
      "train_fun" : skl_logistic.train_logistic,
      "test_fun"  : skl_logistic.test_logistic },

    { "model_name" : "model2",
      "train_fun" : tf_lenet.train_lenet,
      "test_fun"  : tf_lenet.test_lenet,
      "target_size" : (32,32),
      "rescale_mode" : "",
      "batch_size" : 50,
      "learning_rate" : 0.0003,
      "drop_out_rate" : 0.3,
      "epochs" : 3 }
]

MODEL_TRAITS = { traits["model_name"] : traits for traits in MODEL_TRAITS }