# dislib-block-size-estimation
This repository contains the modules implementing a Machine Learning-based solution for optimizing the execution of dislib algorithms.

In particular, a stacked classification model is leveraged to predict the most suitable value of the block-size parameter for the execution of dislib algorithms.

Two modules are provided:
- ***GridSearch***: generate the training dataset for the Machine Learning model, starting from a log of executions of several algorithms on different datasets.
- ***StackedClassifier***: implements the Machine Learning model that is able to predict the most suitable value of the block-size parameter, given an algorithm to be executed and the dataset information.

An additional module, namely ***Main*** is also provided for showing how to use the aforementioned modules.
