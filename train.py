# -*- coding: utf-8 -*-
"""
Created on Fri May 11 17:59:13 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""
#pylint: disable=C0326

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import train_utils as tu


def run_train( model_name, images_dir, verbose=0 ) :
    """Train a model by name on a set of images"""
    #%%
    target_size = (32,32)
    train_4d, train_gt = tu.make_4d_arrays(images_dir=images_dir,
                                           target_size=target_size)

    if verbose > 0 :
        print( "train_4d has shape = %s" % (train_4d.shape,) )

    model_traits = MODEL_TRAITS[model_name]
    train_fun  = model_traits["train_fun"]
    model_type = model_traits["type"]

    model, accu = train_fun( train_4d, train_gt, verbose=verbose )
    tu.save_model( model, model_name, model_type, verbose = 0 )

    print( accu )
    #%%

def run_test( model_name, images_dir, verbose=0 ) :
    """Test a model by name on a set of images"""
    #%%
    target_size = (32,32)
    test_4d, test_gt = tu.make_4d_arrays(images_dir=images_dir,
                                         target_size=target_size)

    if verbose > 0 :
        print( "test_4d has shape = %s" % (test_4d.shape,) )

    test_fun = MODEL_TRAITS[model_name]["test_fun"]
    model_type = MODEL_TRAITS[model_name]["type"]

    model_obj = tu.load_model(model_name, model_type)
    accu = test_fun( model_obj, test_4d, test_gt, verbose=verbose )

    print( accu )
    #%%


def train_logistic( train_4d, train_gt, verbose=0 ) :
    """Train a model by name on a set of images"""
    train_2d = train_4d.reshape( (train_4d.shape[0], -1 ) )
    train_2d_b =  train_2d - train_2d.mean( axis=1, keepdims = True )

    lreg = LogisticRegression( C=20.0, max_iter= 200 )
    lreg.fit( train_2d_b, train_gt )

    #joblib.dump( lreg, model_fn )

    train_pred = lreg.predict( train_2d_b )
    accu = metrics.accuracy_score( train_gt, train_pred )
    if verbose> 0 :
        print( lreg )

    return lreg, accu

def test_logistic( model_obj, test_4d, test_gt, verbose=0 ) :
    """Train a model by name on a set of images"""
    test_2d = test_4d.reshape( (test_4d.shape[0], -1 ) )
    test_2d_b =  test_2d - test_2d.mean( axis=1, keepdims = True )


    test_pred = model_obj.predict( test_2d_b )
    accu = metrics.accuracy_score( test_gt, test_pred )

    if verbose> 0 :
        print( model_obj )

    return accu


def testing() :
    """Quick interctive tests"""
    #%%
    images_dir = "images/train"
    target_size = (32,32)

    train_4d, train_gt = tu.make_4d_arrays(images_dir=images_dir,
                                           target_size=target_size)
    #%%
    accu = train_logistic( train_4d, train_gt, verbose=1 )
    #%%
    return accu

MODEL_TRAITS = {
    "model1" : {
        "train_fun" : train_logistic,
        "test_fun"  : test_logistic,
        "type" : "sklearn"
    }
}
