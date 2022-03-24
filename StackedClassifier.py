import json
import numpy as np
import pickle
from sklearn.metrics import classification_report as cr
from sklearn.tree import DecisionTreeClassifier as decTree
from sklearn.preprocessing import OneHotEncoder
np.set_printoptions(precision=3)


# Author: Riccardo Cantini <rcantini@dimes.unical.it>
#         Alessio Orsino <aorsino@dimes.unical.it>
#         DIMES Department, University of Calabria, Italy
#

class StackedClassifier:
    """
        class implementing the stacked classification model
    """

    def __init__(self, config_path):
        self.configuration = json.load(open(config_path, "rb"))
        self.fitted = False
        self.Rn = decTree()
        self.Rm = decTree()
        self.algoEncoder = OneHotEncoder()

    def fit(self, x_train, y_train_labels):
        """
            Create the stacked classification model and train it on the given x,y pairs.

            Two chained decision trees are fitted as follows:
            1. fit the Rn decision tree on the train instances x, using the number of block rows as target feature;
            2. fit the Rm decision tree using as input the current train instance x concatenated with the output of Rm,
            using the number of block columns as target feature;

            Parameters
            ----------
            x_train: array-like of shape (n_samples, n_features) containing the feature of each train instance,
                i.e.  number of rows, number of columns, algorithm_name
            y_train_labels: array-like of shape (n_samples, 2) containing the target information to be predicted,
                i.e. the number of rows and columns partitions
        """
        x_train = np.array(x_train)
        algs = x_train[:, -1].reshape(-1, 1)
        x_train = x_train[:, :-1].astype(int)
        self.algoEncoder.fit(algs)
        algs_vect = self.algoEncoder.transform(algs).toarray().astype(int)
        x_train = np.column_stack((x_train, algs_vect))
        y_train_Rn = y_train_labels[:, 0]
        y_train_Rn = y_train_Rn.astype(int)
        self.Rn.fit(x_train, y_train_Rn)
        block_n = np.array([np.ceil(n) for n in self.Rn.predict(x_train)])
        x_train_Rm = np.column_stack((x_train, block_n))
        y_train_Rm = y_train_labels[:, 1]
        y_train_Rm = y_train_Rm.astype(int)
        self.Rm.fit(x_train_Rm, y_train_Rm)
        self.fitted = True

    def __predict(self, x_test):
        """
            Predict class values, i.e. the number of row and column partitions.

            Parameters
            ----------
            x_test: array-like of shape (n_samples, n_features), containing the input samples.

            Returns
            -------
            pred_n: array-like of shape (n_samples,), containing the predicted number of partitions
                along rows for each sample.
            pred_m: array-like of shape (n_samples,), containing the predicted number of partitions
                along columns for each sample.
        """
        if not self.fitted:
            raise Exception("This StackedClassifier instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this estimator.")
        x_test = np.array(x_test)
        algs = x_test[:, -1].reshape(-1, 1)
        x_test = x_test[:, :-1].astype(int)
        algs_vect = self.algoEncoder.transform(algs).toarray().astype(int)
        x_test = np.column_stack((x_test, algs_vect))
        pred_n = self.Rn.predict(x_test)
        pred_n = pred_n.astype(int)
        x_test_Rm = np.column_stack((x_test, pred_n))
        pred_m = self.Rm.predict(x_test_Rm)
        pred_m = pred_m.astype(int)
        return pred_n, pred_m

    def save_model(self):
        """
            Save the model state to disk.

            Parameters
            ----------
            path: the path where to save the model.
        """
        model_path_name = self.configuration["var_config"]["model_path_name"]
        pickle.dump(self, open(model_path_name, "wb"))

    def load_model(self):
        """
            Load the model from disk and return the loaded instance.
            > model = StackedClassifier("config_file.json")
            > ...  # use model
            > loaded_model = model.load_model()
            > ... #  use the loaded model

            Returns
            -------
            model: a StackedClassifier instance loaded from disk.
        """
        model_path_name = self.configuration["var_config"]["model_path_name"]
        return pickle.load(open(model_path_name, "rb"))

    def test(self, x_test, y_test):
        """
            Test the trained model on the test set. The method measures the classification performance
            in terms of accuracy, precision, recall and f-measure and log the results in the path specified
            in the config.json file, in the "classification_metrics_path_name" property.

            Parameters
            ----------
            x_test: array-like of shape (n_samples, n_features) containing the feature of each test instance,
                i.e.  number of rows, number of columns, algorithm_name, dataset_name
            y_test: array-like of shape (n_samples, 2) containing the target information to be predicted,
                i.e. the number of row and column partitions
        """
        if not self.fitted:
            raise Exception("This StackedClassifier instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this estimator.")
        y_test_Rn = y_test[:, 0]
        y_test_Rn = y_test_Rn.astype(int)
        y_test_Rm = y_test[:, 1]
        y_test_Rm = y_test_Rm.astype(int)
        pred_n, pred_m = self.__predict(x_test)
        # convert to blockSizes
        pred_blockSizes = []
        true_blockSizes = []
        for x_t, y_t_n, y_t_m, p_n, p_m in zip(x_test, y_test_Rn, y_test_Rm, pred_n, pred_m):
            pred_blockSizes.append(StackedClassifier.__to_blockSize(p_n, p_m, x_t[0], x_t[1]))
            true_blockSizes.append(StackedClassifier.__to_blockSize(y_t_n, y_t_m, x_t[0], x_t[1]))
        classification_metrics_path_name = self.configuration["var_config"]["classification_metrics_path_name"]
        with open(classification_metrics_path_name, 'w') as out:
            # log predictions (a conversion to blockSize is performed)
            out.write("Predictions:")
            for x_t, true_bs, pred_bs in zip(x_test, true_blockSizes, pred_blockSizes):
                out.write("\ntest instance:" + str(x_t) + " -- real blockSize: " + str(true_bs) +
                          " --> prediction: " + str(pred_bs))
            # compute classification metrics
            class_rep_Rn = cr(y_test_Rn, pred_n)
            class_rep_Rm = cr(y_test_Rm, pred_m)
            out.write("\n\nClassification report (row partitions):\n" + str(class_rep_Rn))
            out.write("\nClassification report (column partitions):\n" + str(class_rep_Rm))
            out.close()

    @staticmethod
    def __to_blockSize(p_n, p_m, rows, columns):
        """
            Compute blocksize from model prediction.

            Parameters
            ----------
            p_n: predicted number of partitions along rows.
            p_m: predicted number of partitions along columns.
            rows: number of rows of the dataset to be partitioned.
            columns: number of columns of the dataset to be partitioned.

            Returns
            -------
            blocksize: a tuple containing the number of rows and columns (i.e., the size) of the block.
        """
        return int(rows / p_n), int(columns / p_m)

    def predict(self, x_test):
        """
            Predict the blocksize value, i.e. the number of row and columns of the block, for
            each instance of the given batch.

            Parameters
            ----------
            x_test: array-like of shape (n_samples, n_features), containing the input samples.
                    The feature of each sample must be n, m, alg_name, where:
                    - n is the number of rows of the dataset to be partitioned
                    - m is the number of columns of the dataset to be partitioned
                    - alg_name is the name of the algorithm specified in the config file

            Returns
            -------
            blocksizes: array-like of shape (n_samples,2), containing the predicted number of rows
            and columns of the block for each sample.
        """
        if not self.fitted:
            raise Exception("This StackedClassifier instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this estimator.")
        x_test = np.array(x_test)
        algs = x_test[:, -1].reshape(-1, 1)
        x_test = x_test[:, :-1].astype(int)
        algs_vect = self.algoEncoder.transform(algs).toarray().astype(int)
        x_test = np.column_stack((x_test, algs_vect))
        pred_n = self.Rn.predict(x_test)
        pred_n = pred_n.astype(int)
        x_test_Rm = np.column_stack((x_test, pred_n))
        pred_m = self.Rm.predict(x_test_Rm)
        pred_m = pred_m.astype(int)
        blockSizes = [StackedClassifier.__to_blockSize(p_n, p_m, x_t[0], x_t[1]) for (p_n, p_m, x_t)
                      in zip(pred_n, pred_m, x_test)]
        return blockSizes
