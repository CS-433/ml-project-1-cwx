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

* `util.py`: Implements modules to crate deep neural network

* `Result.ipynb`: Used to show the results of cross validation of various models

* `run.py` :  Creates csv predictions submitted to the competition system

## Requirements
* Python 3.8.5 
* Jupyter Notebook 6.1.4
* Numpy 1.20.3
