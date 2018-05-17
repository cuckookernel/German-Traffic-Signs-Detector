# German-Traffic-Signs-Detector

This repository includes the following parts: 

1. **app.py**: is the main python script that contains the application. 
   We have implemented the following subcommands: 
    
    1. **download**: does downloading of a zip file from a hardcoded URL, unzipping and distributing the images between `images/train` and `images/test` subdirectories. 
    1. **infer**: actually opens windows for each image in a directory (this functionality, depends on the pyglet python library for the GUI functionality) 
    2. **train**:  trains a model given a model name and an images directory. The hyperparameters and other information necessary for the model must be specified in the  `MODEL_TRAITS` dictionary defined in the `model_traits.py`
    3. **test**: tests a model given its name and an images directory. As for train, the model parameters have to be defined in `model_traits.py`, plus the model has to be saved under directory `models/<model_name>/saved`. 
    
 2. **train_test.py** Actually implements `run_train`, `run_test`, and `run_infer` functions. 
	
