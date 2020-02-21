"""
    Contains a function that executes the steps of learning a model.
"""
import os
import logging
import json

import data_utils as du
import model_utils as mu
import plot_utils as pu
import matplotlib.pyplot as plt

DEFAULT_RATIO_FIGURE_MNIST = (3, 3) # (6, 3)
DEFAULT_RATIO_FIGURE_IMAGENET = (3, 3)

def run_exp(base_params,
            out_folder="19_01_11_exps", db_name="MNIST", figsize=(3, 3)):
    """
        Runs an experiment in the out_folder chosen.
        Does 4 steps:
            1. Loads the data,
            2. Learns the model,
            3. Saves the weights and graphs,
            4. Plots the results.
    """

    params = base_params.copy()

    if db_name == "MNIST":
        train_data, test_data = du.load_preprocess_MNIST(params)
    elif db_name == "ImageNet":
        train_data, test_data = du.load_preprocess_ImageNet(params)

    run_exp_loaded_data(train_data, test_data, params,
                        out_folder=out_folder, db_name=out_folder,
                        figsize=out_folder)

def run_exp_loaded_data(train_data, test_data, base_params,
                        out_folder="19_01_11_exps", db_name="MNIST",
                        figsize=(3, 3)):
    """
        Runs an experiment in the out_folder chosen.
        Does 3 steps:
            1. Learns the model,
            2. Saves the weights and graphs,
            3. Plots the results.
    """
    assert base_params["type_weight"] in ["uniform", "prediction", "stratum"]
    assert base_params["model_type"] in ["LINEAR", "MLP"]

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    logging.basicConfig(filename='{}/learning_process.log'.format(out_folder),
                        format='%(asctime)s - %(message)s', # - %(levelname)s
                        level=logging.INFO, datefmt='%m/%d/%y %I:%M:%S %p',
                        filemode="w")
    logging.info("Starting exp...")

    params = base_params.copy()

    if db_name == "MNIST":
        default_ratio_figure = DEFAULT_RATIO_FIGURE_MNIST
    elif db_name == "ImageNet":
        default_ratio_figure = DEFAULT_RATIO_FIGURE_IMAGENET

    message = "p_z_train : " + " | ".join(["%d : %5.4f" % (i, w)
                                           for i, w in
                                           enumerate(params["p_z_train"])])
    logging.info(message)
    message = "p_z_test : " + " | ".join(["%d : %5.4f" % (i, w)
                                          for i, w in
                                          enumerate(params["p_z_test"])])
    message = "Params: " + " | ".join(["%s : %s" % (k, v)
                                       for k, v in params.items()])
    logging.info(message)

    X_train, Z_train, Y_train = train_data
    X_test, Z_test, Y_test = test_data
    params["X_test"] = X_test
    params["Z_test"] = Z_test
    params["Y_test"] = Y_test

    plt.figure(figsize=default_ratio_figure)
    pu.plot_class_probas(params, with_ticklabels=(db_name == "MNIST"))
    plt.savefig("{}/{}.pdf".format(out_folder, "class_probas"), format="pdf")

    plt.figure(figsize=default_ratio_figure)
    pu.plot_strata_probas(params, with_ticklabels=(db_name == "MNIST"))
    plt.savefig("{}/{}.pdf".format(out_folder, "strata_ratios"), format="pdf")

    logging.info("Start learning.")
    if params["model_type"] == "LINEAR":
        model = mu.LinearModel()
    elif params["model_type"] == "MLP":
        model = mu.MLP()
    model.fit(X_train, Z_train, Y_train, params)

    logging.info("Saving the elements.")
    # Transform the asymmetry weights to serializable objects.
    params["p_z_train"] = list(params["p_z_train"])
    params["p_z_test"] = list(params["p_z_test"])


    # We get rid of the testing numpy arrays.
    keys_to_delete = list(filter(lambda x: x in [x + "_test"
                                                 for x in ["X", "Z", "Y"]],
                                 params.keys()))
    # keys_to_delete = keys_to_delete + ["cost_train", "acc_train",
    # "cost_test", "acc_test", "final_weights", "final_bias"]
    params["p_y_train"] = list(params["p_y_train"])
    params["p_y_test"] = list(params["p_y_test"])

    for x in keys_to_delete:
        params.pop(x)

    # print(params)
    json.dump(params, open("{}/params.json".format(out_folder), "wt"))

    logging.info("Plotting the results.")
    plt.figure(figsize=figsize)
    pu.plot_cost_acc(params, lim_acc=params["dynamics_lim_acc"],
                     lim_cost=params["dynamics_lim_cost"])
    plt.savefig("{}/{}.pdf".format(out_folder, "dynamics"), format="pdf")
