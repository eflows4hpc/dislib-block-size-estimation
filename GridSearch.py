import numpy as np
import pandas as pd
import importlib
import time
import json
import dislib as ds
np.set_printoptions(precision=3)


class GridSearch:
    """
        class implementing the grid search procedure
    """

    def __init__(self, config_path):
        self.configuration = json.load(open(config_path, "rb"))

    def generate_log(self):
        """
            Generate the execution log used for building the dataset for the classification model.

            In particular the method reads the "exec_config" section of the config.json file, which contains
            a list of executions to be performed for a given algorithm.
            Each execution in the list is characterized by the path to the dataset to be used and a dictionary
            containing the configuration parameters of the algorithm.

            For each pair <algorithm, dataset> different executions are performed varying the value of blocksize along
            rows and columns, in the intervals specified in the "var_config" section, i.e. "number_of_row_partitions"
            and "number_of_column_partitions". If an execution fails, the corresponding elapsed time is set to the
            value specified by the "test_failed" config property (default = -1).

            The generated execution log file is stored in json format in the path specified in the config file by the
            "execution_log_path_name" and it is structured as follows:
            {
                "RandomForestClassifier": [
                  {
                    "dataset_path": "datasets/dataset1",
                    "n": 1138,  # dataset rows
                    "m": 60,  # dataset columns
                    "times": [1.76, 0.56, 0.42, -1],  # exec. times (-1 --> test_failed, see config.json)
                    "partitioning": ["(1, 1)", "(4, 1)", "(32, 1)", "(64, 1)"]  # used paritioning
                  },
                  {
                    "dataset_path": "datasets/dataset2",
                    ...
                  }
                ],
                "KMeans": [
                  {
                    ...
                  }
                ],
                ...
            }
        """
        exec_config = self.configuration["exec_config"]
        exec_log = {}
        for alg_name, exec_to_be_performed in exec_config.items():
            executions = []
            for exec in exec_to_be_performed:
                exec_on_ds = {}
                dataset_path = exec["dataset_path"]
                dataset = pd.read_csv(dataset_path).values
                (n,m) = dataset.shape
                exec_params = exec["params"]
                partitioning = []
                times = []
                for p_rows in self.configuration["var_config"]["number_of_row_partitions"]:
                    for p_columns in self.configuration["var_config"]["number_of_columns_partitions"]:
                        # convert partitioning to blocksize for the execution
                        block_size = (int(n/p_rows), int(m/p_columns))
                        el_time = self.execute(alg_name, block_size, dataset, exec_params)
                        times.append(el_time)
                        partitioning.append((p_rows, p_columns))
                exec_on_ds["dataset_path"] = dataset_path
                exec_on_ds["n"] = n
                exec_on_ds["m"] = m
                exec_on_ds["times"] = times
                exec_on_ds["partitioning"] = [str(part) for part in partitioning]
                executions.append(exec_on_ds)
            exec_log[alg_name] = executions
        execution_log_path_name = self.configuration["var_config"]["execution_log_path_name"]
        with open(execution_log_path_name, 'w') as json_log:
            json_log.write(json.dumps(exec_log, indent=4))

    def generate_training_dataset(self):
        """
            Create the training dataset to be used by the stacked classification model, starting from the log of executions.
            The created dataset is stored on disk in csv format, in the path specified in the config.json file

            Given an algorithm and a dataset, the method extracts the best blocksize found through grid search
            from the log, i.e. the size that led to the minimum execution time.

            In the provided implementation, the log file is expected to contain the information about the execution
            of a set of algorithms on different datasets. In particular, the log file is stored in json format and
            must strictly follow the structure described in the "generate_log" method.
        """
        output_dataset = []
        execution_log_path_name = self.configuration["var_config"]["execution_log_path_name"]
        TEST_FAILED = int(self.configuration["var_config"]["test_failed"])
        exec_log = json.load(open(execution_log_path_name, "rb"))
        for alg_name, executions in exec_log.items():
            for exec_on_ds in executions:
                partitioning = np.array([eval(t) for t in exec_on_ds["partitioning"]])
                times = np.array(exec_on_ds["times"])
                not_failed_mask = times != TEST_FAILED
                partitioning = partitioning[not_failed_mask]
                times = times[not_failed_mask]
                n = exec_on_ds["n"]
                m = exec_on_ds["m"]
                if len(times) > 0:
                    best_partitioning = partitioning[np.argmin(times)]
                    line = [n, m, alg_name, best_partitioning[0], best_partitioning[1]]
                    output_dataset.append(line)
        dataset_path_name = self.configuration["var_config"]["training_dataset_path_name"]
        pd.DataFrame(output_dataset).to_csv(dataset_path_name, index=False, header=None)

    def execute(self, alg_name, block_size, dataset, dict_params):
        """
            Execute a given algorithm on a dataset using the specified parameters.

            Parameters
            ----------
            alg_name: the name of the algorithm to be executed. It must be present in the config file,
                      in the "algorithms" section.
            block_size: the blocksize to be used.
            dataset: the dataset to be used. Datasets are supposed to be in csv format with no column header.
                     The last column is treated as the target variable for supervised executions.
            dict_params: a dictionary containing the configuration parameters of the algorithm, e.g.:
                         "params": {
                                "random_state": 21,
                                "n_clusters": 2
                         }

            Returns
            -------
            el_time: the duration of the performed execution (in seconds). If an execution fails, the returned elapsed
                     time is set to the value specified by the "test_failed" config property (default = -1).
        """
        (n,m) = dataset.shape
        alg_conf = self.configuration["algorithms"][alg_name]
        supervised = eval(alg_conf["supervised"])
        if supervised:
            x = dataset[:, :-1]
            y = dataset[:, -1]
            if block_size[1] > x.shape[1]:
                block_size = (block_size[0], x.shape[1])
        else:
            x = dataset
        module_name = alg_conf["module_name"]
        class_name = alg_conf["class_name"]
        module = importlib.import_module(module_name)
        alg_class = getattr(module, class_name)
        alg_instance = alg_class()
        for (p_name, p_value) in dict_params.items():
            setattr(alg_instance, p_name, p_value)
        exec_info_path_name = self.configuration["var_config"]["execution_info_path_name"]
        exec_info = open(exec_info_path_name, 'a')
        exec_info.write("\nExecuting algorithm " + str(module_name) + "." + str(class_name) +
                        " using blocksize: " + str(block_size) +
                        "\nDataset dimension: " + str(n) + " rows, " + str(m) + " columns" +
                        "\nParams: " + str(dict_params) + "\n")
        TEST_FAILED = self.configuration["var_config"]["test_failed"]
        try:
            ds_x_train = ds.array(x, block_size=block_size)
            if supervised:
                ds_y_train = ds.array(y.reshape(-1, 1), block_size=(block_size[0], 1))
            start = time.time()
            if supervised:
                alg_instance.fit(ds_x_train, ds_y_train)
            else:
                alg_instance.fit(ds_x_train)
            end = time.time()
            el_time = end - start
            exec_info.write("Elapsed time: " + str(el_time) + "\n")
        except Exception as e:
            el_time = TEST_FAILED
            exec_info.write("\nAn error occurred. " + str(e) + " -- Elapsed time set to " + str(TEST_FAILED) + "\n")
        return el_time
