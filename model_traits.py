# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:38:14 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""

#pylint: disable=C0326
import skl_logistic
import tf_log_reg
import tf_lenet


MODEL_TRAITS = [

    { "model_name" : "model1",
      "trainer_tester_class" : skl_logistic.LRTrainerTester },

    { "model_name" : "model2",
      "trainer_tester_class" : tf_log_reg.LRTrainerTester,
      "target_size" : (32,32),
      "rescale_mode" : "",
      "batch_size" : 50,
      "learning_rate" : 0.001,
      "dropout_rate" : 0.2,
      "epochs" : 86 },

    { "model_name" : "model3",
      "trainer_tester_class" : tf_lenet.LenetTrainerTester,
      "target_size" : (32,32),
      "rescale_mode" : "",
      "batch_size" : 100,
      "net_version" : "orig",
      "drop_colors" : 1,
      "learning_rate" : 0.0005,
      "dropout_rate" : 0.3,
      "epochs" : 300 },

    { "model_name" : "model3v1",
      "trainer_tester_class" : tf_lenet.LenetTrainerTester,
      "target_size" : (32,32),
      "rescale_mode" : "",
      "batch_size" : 100,
      "net_version" : "v1",
      "drop_colors" : 0,
      "learning_rate" : 0.0003,
      "dropout_rate" : 0.25,
      "epochs" : 61 },

    { "model_name" : "model4",
      "trainer_tester_class" : tf_lenet.LenetTrainerTester,
      "target_size" : (32,32),
      "rescale_mode" : "max_q",
      "batch_size" : 50,
      "net_version" : "orig",
      "drop_colors" : 1,
      "learning_rate" : 0.0001,
      "dropout_rate" : 0.1,
      "epochs" : 300 },


    { "model_name" : "model3b",
      "trainer_tester_class" : tf_lenet.LenetTrainerTester,
      "target_size" : (32,32),
      "rescale_mode" : "",
      "batch_size" : 50,
      "net_version" : "v1",
      "drop_colors" : 1,
      "learning_rate" : 0.0005,
      "dropout_rate" : 0.3,
      "epochs" : 3 }
]

MODEL_TRAITS = { traits["model_name"] : traits for traits in MODEL_TRAITS }
