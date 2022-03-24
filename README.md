# dislib-block-size-estimation
This repository contains the modules implementing a Machine Learning-based solution for optimizing the execution of dislib algorithms.

In particular, a stacked classification model is leveraged to predict the most suitable value of the block-size parameter for the execution of dislib algorithms.

Two modules are provided:
- ***GridSearch***: generate the training dataset for the Machine Learning model, starting from a log of executions of several algorithms on different datasets.
- ***StackedClassifier***: implements the Machine Learning model that is able to predict the most suitable value of the block-size parameter, given an algorithm to be executed and the dataset information.

An additional module, namely ***Main***, is also provided for showing how to use the aforementioned modules.

The ***config.json*** file contains all the configuration parameters, listed in the following:
- *algorithms*: a json containing the information about all the executed algorithms. Each algorithm is instantiated via reflection mechanism and requires the module and class name, along with the type of the task (supervised or not).
- *var_config*: contains the general configuration.
- *number_of_row_partitions*: a list containing all the partitions along rows to be tested during the grid search.
- *number_of_column_partitions*: a list containing all the partitions along columns to be tested during the grid search.
- *model_path_name*: the path where the trained Machine Learning model is stored.
- *training_dataset_path_name*: the path where the training set, used for training the Machine Learning model, is stored.
- *execution_log_path_name*: the path where the log of executions, used for creating the training dataset, is stored.
- *execution_info_path_name*: the path where the file containing the information about the performed executions is stored.
- *classification_metrics_path_name*: the path where the file containing the classificaiton metrics is stored.
- *test_failed*: constant to be used for the elapsed time in the case of a filed test.
- "exec_config": a json containing the configuration of the different executions to be performed. For each algorithm (the same key as in the "algorithm" section) a list must be provided, that contains a json configuration related to the execution of that algorithm on a specified dataset. In particular, the configuration contains the path to the dataset to be used and the algorithm parameters, which are set dinamically.
