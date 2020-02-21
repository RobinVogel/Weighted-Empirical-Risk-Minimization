"""Utilities for plotting the results of the experiments."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
# Avoid trouble when generating pdf's on a distant server
# matplotlib.use("TkAgg") # Be able to import matplotlib in ipython
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_cost_acc(params, lim_acc=None, lim_cost=None):
    """Plots the cost value and accuracy for test and train."""
    plt.plot(params["step"], params["cost_test"],
             label="SCE test", color="red")
    plt.plot(params["step"], params["cost_train"],
             label="SCE train", color="red", linestyle="--")
    plt.grid()
    plt.legend(loc="lower left")
    plt.ylabel("SCE", color="red")
    if lim_cost:
        plt.ylim(lim_cost)
    plt.twinx()
    plt.plot(params["step"], 1 - np.array(params["acc_test"]),
             label="miss test", color="blue")
    plt.plot(params["step"], 1 - np.array(params["acc_train"]),
             label="miss train", color="blue", linestyle="--")
    plt.ylabel("misses", color="blue")
    if lim_acc:
        plt.ylim(lim_acc)
    plt.legend(loc="upper right")
    plt.tight_layout()

def plot_norm(params):
    """Plots the regularization."""
    plt.plot(params["step"], params["norm_mat"], label="norm", color="red")
    plt.tight_layout()

# --- Quantile plotting functions ---

def quantile(X, q, axis=0):
    """np.quantile only exists on numpy 1.15 and higher."""
    assert axis == 0
    X = np.array(X)
    return np.sort(X, axis=0)[int(X.shape[0]*q), :]

def param_list_to_quant(key, q, p_list):
    """Returns a quantile."""
    if key.startswith("acc") or key.startswith("top"):
        # We plot the error rates 1-acc.
        return quantile(1-np.array([p[key] for p in p_list]), q)
    return quantile(np.array([p[key] for p in p_list]), q)

def plot_quant(params_list, param_name, label, color,
               linestyle="-", alpha=0.05):
    """Plots quantile intervals with the desired value."""
    p_list = params_list
    params = p_list[0]

    # Plot the result
    plt.fill_between(params["step"],
                     param_list_to_quant(param_name, (1-alpha/2), p_list),
                     param_list_to_quant(param_name, alpha/2, p_list),
                     color=color, alpha=0.25)
    plt.plot(params["step"], param_list_to_quant(param_name, 0.5, p_list),
             label=label, color=color, linestyle=linestyle)

def plot_quant_cost_acc(params_list, alpha, lim_acc=None, lim_cost=None,
                        left_label_remove=False, right_label_remove=False):
    """Plots quantile intervals of the cost value and accuracy."""

    p_list = params_list
    params = p_list[0]

    # Plot the result
    plt.fill_between(params["step"],
                     param_list_to_quant("cost_test", (1-alpha/2), p_list),
                     param_list_to_quant("cost_test", alpha/2, p_list),
                     color="red", alpha=0.25)
    plt.plot(params["step"], param_list_to_quant("cost_test", 0.5, p_list),
             label="SCE test", color="red")
    plt.plot(params["step"], param_list_to_quant("cost_train", 0.5, p_list),
             label="SCE train", color="red", linestyle="--")
    plt.grid()
    plt.legend(loc="lower left")

    if not left_label_remove:
        plt.ylabel("SCE", color="red")
    else:
        plt.gca().yaxis.set_ticklabels([])
    if lim_cost:
        plt.ylim(lim_cost)
    plt.twinx()
    plt.fill_between(params["step"],
                     param_list_to_quant("acc_test", (1-alpha/2), p_list),
                     param_list_to_quant("acc_test", alpha/2, p_list),
                     color="blue", alpha=0.25)
    plt.plot(params["step"],
             np.array(param_list_to_quant("acc_test", 0.5, p_list)),
             label="miss test", color="blue")
    plt.plot(params["step"],
             np.array(param_list_to_quant("acc_train", 0.5, p_list)),
             label="miss train", color="blue", linestyle="--")

    if not right_label_remove:
        plt.ylabel("misses", color="blue")
    else:
        plt.gca().yaxis.set_ticklabels([])
    if lim_acc:
        plt.ylim(lim_acc)
    plt.legend(loc="upper right")
    plt.tight_layout()

def plot_quantiles(root_exp_folder,
                   subfolds=("uniform", "prediction", "stratum"),
                   alpha=0.05, figsize=(3, 3),
                   lim_acc=None, lim_cost=None,
                   camera_ready=True):
    """
        Plot the quantile graphs for a standardized experiment.

        If the experiment data is in a folder named exp, it expects to find
        subfolders subfolds in which there are folders with runs and a
        params.json file that contains the result of the runs. It will generate
        a quantile plot for each of the experiments.
    """
    for type_weight in subfolds:
        cur_dir = "{}/tw_{}/".format(root_exp_folder, type_weight)
        params_list = get_param_list_for_quantiles(root_exp_folder,
                                                   type_weight)

        plt.figure(figsize=figsize)
        right_label_remove = camera_ready and type_weight == "uniform"
        left_label_remove = camera_ready and type_weight == "prediction"
        plot_quant_cost_acc(params_list, alpha=alpha,
                            lim_acc=lim_acc, lim_cost=lim_cost,
                            left_label_remove=left_label_remove,
                            right_label_remove=right_label_remove)
        if not camera_ready:
            plt.title(type_weight)
        plt.savefig("{}/{}.pdf".format(cur_dir, "quantiles"), format="pdf")

def get_param_list_for_quantiles(root_exp_folder, type_weight):
    """Accumulates all of the json files for different runs and reweighting."""
    cur_dir = "{}/tw_{}/".format(root_exp_folder, type_weight)
    cur_runs = os.listdir(cur_dir)
    params_list = list()
    for cur_run in cur_runs:
        cur_file = cur_dir + cur_run + "/params.json"
        if os.path.exists(cur_file):
            params_list.append(json.load(open(cur_file, "rt")))
    return params_list

def plot_quantiles_cost_n_miss(root_exp_folder,
                               subfolds=("uniform", "prediction", "stratum"),
                               sf_style={"uniform": {"color": "blue",
                                                     "label": "Uniform"},
                                         "prediction": {"color": "green",
                                                        "label": "Weighted"},
                                         "stratum": {"color": "green",
                                                     "label": "Stratum"},
                                        },
                               alpha=0.05, figsize=(3, 3), lim_acc=None,
                               lim_cost=None, lim_top=None, camera_ready=True):
    """
        Plot the quantile graphs for a standardized experiment.

        If the experiment data is in a folder named exp, it expects to find
        subfolders subfolds in which there are folders with runs and a
        params.json file that contains the result of the runs. It will generate
        a quantile plot for each of the experiments.
    """
    print("Loading the data...")
    list_params_list = list()
    for type_weight in subfolds:
        list_params_list.append(get_param_list_for_quantiles(
            root_exp_folder, type_weight))

    print("Plotting results...")
    for plotted_val in ("acc", "cost", "topk"):
        plt.figure(figsize=figsize)
        for itw, type_weight in enumerate(subfolds):
            params_list = list_params_list[itw]

            params = params_list[0]

            color = sf_style[type_weight]["color"]

            plot_quant(params_list, "{}_test".format(plotted_val), "",
                       color=color)
            plt.plot(params["step"], param_list_to_quant(
                "{}_train".format(plotted_val), 0.5, params_list),
                     color=color, linestyle="--")
        if plotted_val == "acc" and lim_acc:
            plt.ylim(lim_acc)
            plt.ylabel("Miss rate")
        if plotted_val == "cost" and lim_cost:
            plt.ylim(lim_cost)
            plt.ylabel("SCE")
        if plotted_val == "topk" and lim_top:
            plt.ylim(lim_top)
            plt.ylabel("Top-5 error")
        if not camera_ready:
            plt.title(plotted_val)

        train_lgd = Line2D([0, 0], [1, 1], color="black", linestyle="--")
        test_lgd = Line2D([0, 0], [1, 1], color="black", linestyle="-")
        legend1 = plt.gca().legend([train_lgd, test_lgd], ["Train", "Test"],
                                   loc="upper right")
        legend_lines = [Line2D([0, 0], [1, 1],
                               color=sf_style[k]["color"], linestyle="-")
                        for k in subfolds]
        legend_names = [sf_style[k]["label"] for k in subfolds]
        plt.gca().legend(legend_lines, legend_names, loc="lower left")
        plt.gca().add_artist(legend1)

        plt.grid()
        plt.tight_layout()
        # elif plotted_val == "acc":
        #     plt.title("Miss rate")
        # else:
        #     plt.title("SCE")
        plt.savefig("{}/{}_{}.pdf".format(root_exp_folder, "quant",
                                          plotted_val), format="pdf")
    print("Done !")

def plot_class_probas(params, with_ticklabels=True):
    """Plots the probabilities of each class for the train and test set."""
    n_classes = len(params["p_y_train"])
    width = 0.35
    ind = np.arange(n_classes)
    ax = plt.gcf().subplots()
    rects1 = ax.bar(ind, params["p_y_train"], width, color="blue")
    rects2 = ax.bar(ind + width, params["p_y_test"], width, color="green")

    ax.set_ylabel("Probability")
    ax.set_xlabel("Class")
    ax.set_xticks(ind+width/2)

    if with_ticklabels:
        ax.set_xticklabels(ind)
    else:
        ax.set_xticklabels(["" for _ in ind])

    ax.legend((rects1[0], rects2[0]), ("Train", "Test"))
    plt.grid()
    plt.gcf().tight_layout()

def plot_strata_probas(params, with_ticklabels=True):
    """Plot the probabilities of each strata for train and test."""
    n_stratas = len(params["p_z_train"])
    width = 0.35
    ind = np.arange(n_stratas)
    ax = plt.gcf().subplots()
    rects1 = ax.bar(ind, params["p_z_train"], width, color="blue")
    rects2 = ax.bar(ind + width, params["p_z_test"], width, color="green")

    ax.set_ylabel("Probability")
    ax.set_xlabel("Strata")
    ax.set_xticks(ind+width/2)
    if with_ticklabels:
        ax.set_xticklabels(ind)
    else:
        ax.set_xticklabels(["" for _ in ind])
    ax.legend((rects1[0], rects2[0]), ("Train", "Test"))
    plt.gca().yaxis.grid(True)
    # plt.grid()
    plt.gcf().tight_layout()
