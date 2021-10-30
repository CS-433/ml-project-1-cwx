# CS-433 Machine Learning Project 1

__Team Member__: Zewei XU, Haoxuan WANG, Ganyuan CAO


## Project Description
We implemented a machine learning model on CERN particle accelerator data to recreate the process of discovering the Higgs particle. In our model, we implemented several regression algorithms. We evaluated our model with the AiCrowd platform, and with local validation data. Our model achieves relatively high categorical accuracy on AiCrowd and our local cross-validation. 

## File Description
* `main.py`: Use the functions in `implementation.py` to compute loss and last weight vector.

* `implementation.py`: Implement
  *  Linear Regression with Gradient Descent (GD)
  *  Linear Regression with Stochastic Gradient Descent (SGD) using Mini-Batch-Size-1
  *  Least Squares Regression with Normal Equation
  *  Ridge Regression with Normal Equation
  *  Logistic Regression 
  *  Regularized Logistic Regression. 

* `proj1_helpers.py`: Some helper functions for project 1
  * Functions for normalization: standardize(), lognormal(), min_max_std();
  * Functions for data cleaning: fix_empty(), delete_outlier();
  * Function for generate polynomial features: build_poly();
  * Function for embedding: pca();
  * Function for validation: cross_validation().
    
* `util.py`: Implements modules to crate deep neural network
  *  Class Linear
  *  Class ReLU
  *  Class Tanh
  *  Class Dropout
  *  Class MSE
  *  Class CEL
  *  Class SGD
  *  Class Sequential

* `Result.ipynb`: Used to show the results of cross validation of various models

* `DNN_Result.ipynb`: Used to show the results of DNN with different optimization function

* `run.py` :  Creates csv predictions submitted to the competition system

## Requirements
* Python 3.8.5 
* Jupyter Notebook 6.1.4
* Numpy 1.20.3
