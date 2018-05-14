# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:01:01 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""

#pylint: disable=C0326
from itertools import product
import os
import json
import numpy as np

import train_utils as tu
from tf_lenet import train_lenet
from model_traits import MODEL_TRAITS

def run_experiments() :
    """ Hyperparameter tuning for lenet """
    #%%
    target_size=(32,32)
    g_specs = {
        "batch_size" : [ 30 , 60, 100  ],
        "learning_rate" : [ 0.0002, 0.0003, 0.0005 ],
        "drop_out_rate" : [  0.2, 0.25, 0.3 ],

        "rescale_mode" : [ "max_q" , "max", "" ]
    }

    model_traits = MODEL_TRAITS["model2"].copy()
    del model_traits["train_fun"]
    del model_traits["test_fun"]


    cnt = 0
    for batchs, lrate, do_rate, resc_mode in product( g_specs["batch_size"],
                                                      g_specs["learning_rate"],
                                                      g_specs["drop_out_rate"],
                                                      g_specs["rescale_mode"] ) :

        model_traits.update( {"batch_size" : batchs,
                              "learning_rate" : lrate,
                              "rescale_mode" :  resc_mode,
                              "drop_out_rate" : do_rate })

        train_4d, train_gt = tu.make_4d_arrays( images_dir="images/train",
                                                target_size=target_size)

        train_mean = train_4d.mean( axis=(1,2,3), keepdims=True )
        train_std  = train_4d.std( axis=(1,2,3), keepdims=True )
        train_x = (train_4d - train_mean)/ train_std


        test_4d, test_gt = tu.make_4d_arrays( images_dir="images/test",
                                              target_size=target_size)

        test_mean = test_4d.mean( axis=(1,2,3), keepdims=True )
        test_std  = test_4d.std( axis=(1,2,3), keepdims=True )
        test_x = (test_4d - test_mean)/ test_std

        data = {"train_x" :train_x,
                "test_x"  :test_x,
                "train_y" : train_gt,
                "test_y"  : test_gt}

        valid_accu_log, train_accu_log = train_lenet( model_traits, data,
                                                      logl=100 )
        idx_v = int(np.argmax( valid_accu_log))
        idx_t = int(np.argmax( train_accu_log))

        model_traits.update({"valid_accu_log" : valid_accu_log,
                             "train_accu_log" : train_accu_log,
                             "best_valid" : max(valid_accu_log),
                             "best_valid_at" : idx_v,
                             "train_at_best_valid" : train_accu_log[idx_v],
                             "best_train" : max(train_accu_log),
                             "best_train_at":  idx_t
                            })

        #print(cnt, pformat(model_traits) )
        print( "%d : best_train = %.4f, best_valid = %.4f" % \
               (cnt, max(train_accu_log), max(valid_accu_log) ))

        with open( "exp_results_%d.json" % cnt,
                   "wt" , encoding="utf8" ) as f_out :
            print( json.dumps( model_traits ), file=f_out)


        cnt += 1
    #%%

def collect_results( results_dir = "experiments" ) :
    """Collect results from experiments"""
    #%%
    import pandas as pd
    exps_fn = os.listdir( results_dir )
    dics = []
    for fname in exps_fn :
        with open( results_dir + "/" + fname, "rt", encoding="utf8" ) as f_out :
            dics.append( json.load( f_out ) )

    results_df = pd.DataFrame( dics )
    #%%
    return results_df


def test():
    """Quick test"""
    #%%
    exps_df = collect_results( results_dir = "experiments2" )
    #%%
    return exps_df
