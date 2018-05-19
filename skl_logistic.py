# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:39:12 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""

#pylint: disable=C0326
import logging

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import train_utils as tu

log = logging.getLogger( __name__ ).log #pylint: disable=C0103

class LRTrainerTester( tu.TrainerTester ) :
    """Train and test a logistic regression model using scikit-learn"""

    def train( self, data, logl=100, save_model=True ) :
        """Train a model by name on a set of images"""

        train_4d = data["train_4d"]
        train_gt = data["train_gt"]

        train_2d = train_4d.reshape( (train_4d.shape[0], -1 ) )
        train_2d_b = train_2d - train_2d.mean( axis=1, keepdims = True )

        lreg = LogisticRegression( C=20.0, max_iter= 200 )
        lreg.fit( train_2d_b, train_gt )

        #joblib.dump( lreg, model_fn )
        model_name = self.model_traits["model_name"]
        if save_model :
            tu.save_sklearn_model( lreg, model_name, logl=logl-1 )

        train_pred = lreg.predict( train_2d_b )
        accu = metrics.accuracy_score( train_gt, train_pred )
        log( logl, lreg )

        return { "accuracy" : accu }

    def test( self, data, return_inferred=False, logl=100 ) :
        """Train a model by name on a set of images"""
        test_4d = data["test_4d"]
        test_gt = data["test_gt"]

        test_2d = test_4d.reshape( (test_4d.shape[0], -1 ) )
        test_2d_b = test_2d - test_2d.mean( axis=1, keepdims = True )

        model_obj = tu.load_sklearn_model( self.model_traits["model_name"],
                                           logl=logl-1 )

        test_pred = model_obj.predict( test_2d_b )
        accu = metrics.accuracy_score( test_gt, test_pred )

        log(logl, "%s", model_obj )

        return accu if not return_inferred else  list(test_pred)
