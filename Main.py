from StackedClassifier import StackedClassifier
from GridSearch import GridSearch
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split

CONFIG_PATH = "config.json"

# STEP 1. grid search
grid_search = GridSearch(CONFIG_PATH)
## generation of execution log
grid_search.generate_log()
## generation of the training dataset
grid_search.generate_training_dataset()

# STEP 2. dataset preparation for classification model training
dataset_training_path = grid_search.configuration["var_config"]["training_dataset_path_name"]
## load the training dataset in csv format
dataset_training = pd.read_csv(dataset_training_path, header=None).values
## extract training features
X = dataset_training[:, :-2]
## extract the 2-dimensional target variable, i.e. the number of rows and column partitions
Y = dataset_training[:, -2:]
## random split, set "random_state" for reproducibility
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# STEP 3. classification model training
stacked_cls = StackedClassifier(CONFIG_PATH)
## fit the model
stacked_cls.fit(X_train, Y_train)
## test the model for evaluating classification performance
stacked_cls.test(X_test, Y_test)
## save the model for future reuse
stacked_cls.save_model()

# STEP 4. load the model and make predictions
new_stacked_cls = StackedClassifier(CONFIG_PATH)
## load the model into a new instance
loaded_stacked_cls = new_stacked_cls.load_model()
## prepare a batch of test instances
## in this example the batch contains just one test instance
dataset_rows = 888
dataset_columns = 11
alg_name = "KMeans"
alg_params = {"random_state": 21, "n_clusters": 2}
dataset_path = "datasets/titanic.csv"
dataset = pd.read_csv(dataset_path).values
## N.B. Input batch must be 2-dimensional.
test_batch = np.array([[dataset_rows, dataset_columns, alg_name]])
## predict the blocksize for each test instance in the batch.
## N.B. Output prediction return an array-like with the block-size for each test_instance in the input batch
predicted_block_sizes = loaded_stacked_cls.predict(test_batch)
## extract the blocksize from the batch predictions
predicted_block_size = predicted_block_sizes[0]
print("Predicted blocksize: " + str(predicted_block_size))
## compare execution time achieved with the predicted blocksize and with a random blocksize
el_time_predicted_blockSize = grid_search.execute(alg_name, predicted_block_size, dataset, alg_params)
random_block_size = (random.randint(1, dataset_rows), random.randint(1, dataset_columns))
print("Random blocksize: " + str(random_block_size))
el_time_random_blockSize = grid_search.execute(alg_name, random_block_size, dataset, alg_params)
print("\nElapsed time (predicted blocksize): " + str(el_time_predicted_blockSize) + " sec.")
print("Elapsed time (random blocksize): " + str(el_time_random_blockSize) + " sec.")
