# German-Traffic-Signs-Detector

This repository includes the following parts: 

1. **requirements.txt** which can be used with pip to install the required libraries, most of them are pretty standard, except perhaps for `pyglet` which is only used for the GUI functionality required by 

1. **app.py**: is the main python script that contains the application. 
   We have implemented the following subcommands: 
    
    1. **download**: does downloading of a zip file from a hardcoded URL, unzipping and distributing the images between `images/train` and `images/test` subdirectories. 
    1. **infer**: actually opens windows for each image in a directory (this functionality, depends on the pyglet python library for the GUI functionality) 
    2. **train**:  trains a model given a model name and an images directory. The hyperparameters and other information necessary for the model must be specified in the  `MODEL_TRAITS` dictionary defined in the `model_traits.py`
    3. **test**: tests a model given its name and an images directory. As for train, the model parameters have to be defined in `model_traits.py`, plus the model has to be saved under directory `models/<model_name>/saved`. 
 
 2. **download.py** Implements the download subcommand
 
 4. **train_test.py** Actually implements `run_train`, `run_test`, and `run_infer` functions. 
 
 4. **skl_models.py** Contains code for training and testing LogisticRegression through scikit-learn
 
 5. **tf_lenet.py**: Has the code for training and testing tensorflow (tf) version of LeNet, included the network construction.
 
 6. **train_utils.py** Contains some common low-level utilities used through out. 
 
 
 ## Models trained 
 
 1. Model 1:  Basic Logistic regression Classifier with L2 regularization built on top of Scikit-Learn.
 2. Model 2:  Logistic regression Classifier built on top  of tensorflow with drop-out regularization 
 3. Model 3:  LeNet implementation built on top of tensorflow with drop-out regularization
 
