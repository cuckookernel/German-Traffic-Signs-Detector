# -*- coding: utf-8 -*-
"""
Created on Fri May 11 17:59:13 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""
#pylint: disable=C0326

import logging

import train_utils as tu
from model_traits import MODEL_TRAITS

LOG=logging.getLogger(__name__)
LOG.setLevel( 1 )
log = LOG.log  #pylint: disable=C0103

def run_train( model_name, images_dir, logl=100) :
    """Train a model by name on a set of images"""
    #%%
    target_size = (32,32)
    train_4d, train_gt = tu.make_4d_arrays(images_dir=images_dir,
                                           target_size=target_size)

    log(logl, "train_4d has shape = %s", (train_4d.shape,) )

    model_traits = MODEL_TRAITS[model_name]
    train_fun  = model_traits["train_fun"]
    data = { "train_4d" : train_4d, "train_gt" : train_gt }

    train_res = train_fun( model_traits, data, logl=logl -1 )

    print( train_res["accuracy"] )
    #%%

def run_test( model_name, images_dir, logl=100 ) :
    """Test a model by name on a set of images"""
    target_size = (32,32)
    test_4d, test_gt = tu.make_4d_arrays(images_dir=images_dir,
                                         target_size=target_size)


    log(logl, "test_4d has shape = %s", (test_4d.shape,) )

    data = { "test_4d" : test_4d, "test_gt" : test_gt }

    model_traits = MODEL_TRAITS[model_name]
    test_fun = model_traits["test_fun"]

    #model_obj = tu.load_model(model_name, model_type)
    accu = test_fun( model_traits, data, logl=logl-1 )

    print( accu )
    #%%
