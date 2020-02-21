"""
    Contains functions that:
    Explores the distribution of the instances for ImageNet
    for different strata asymmetries gamma.

    Summarizes the results of the runs from a folder.
"""

import sys
import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("pdf")
# Avoid trouble when generating pdf's on a distant server
# matplotlib.use("TkAgg") # Be able to import matplotlib in ipython
import matplotlib.pyplot as plt

import plot_utils as pu

IMAGE_NET_LOC = "imagenet_data/data"

SEED_SHUFFLE_P_Z = 42

def explore_strata_dist():
    """Plots lots of distributions over strata for asymmetric parameters."""
    default_ratio_figure = (12, 3)
    out_folder = "strata_dist_imagenet"

    train_file = "{}/{}.npy".format(IMAGE_NET_LOC, "original_p_z_train")
    p_z_train = np.load(open(train_file, "rb"))
    test_file = "{}/{}.npy".format(IMAGE_NET_LOC, "original_p_z_test")
    p_z_test = np.load(open(test_file, "rb"))

    n_strata = len(p_z_test)

    for strata_asym in np.linspace(0, 1, 101):
        # np.logspace(-10, 0, 101):
        # Inducing asymmetries on the p_z's
        np.random.seed(SEED_SHUFFLE_P_Z)
        prob_z_ratio = np.array([strata_asym**(-k/n_strata)
                                 for k in range(0, n_strata)])
        print(strata_asym, prob_z_ratio)
        np.random.shuffle(prob_z_ratio)
        np.random.seed()

        # Heuristic way to do strata asymmetry
        p_z_train = p_z_test*prob_z_ratio
        p_z_train = p_z_train/p_z_train.sum()

        params = {"p_z_train": p_z_train, "p_z_test": p_z_test}

        plt.figure(figsize=default_ratio_figure)
        pu.plot_strata_probas(params)
        plt.savefig("{}/{}.pdf".format(
            out_folder, "strata_ratios_"+str(strata_asym)), format="pdf")

def summarize_results(in_folder=""):
    """Gives statistics for top-5 error and miss for the runs in in_folder."""
    all_runs = os.listdir(in_folder)
    top_5_errors = list()
    miss_rates = list()
    for cur_run in all_runs:
        param_name = "{}/{}/params.json".format(in_folder, cur_run)
        if os.path.exists(param_name):
            params = json.load(open(param_name, "rt"))
            miss_rates.append(1-params["acc_test"][-1])
            top_5_errors.append(1-params["topk_test"][-1])
    mean_miss = np.mean(miss_rates)
    std_miss = np.std(miss_rates)
    # print(miss_rates)
    print("miss: {:.3f} (+/- {:.3f})".format(mean_miss, 2*std_miss))
    mean_top = np.mean(top_5_errors)
    std_top = 2*np.std(top_5_errors)
    # print(top_5_errors)
    print("top: {:.3f} (+/- {:.3f})".format(mean_top, 2*std_top))


def main():
    # explore_strata_dist()

    folder_basis = sys.argv[1]
    values = ["/tw_uniform", "/tw_stratum", "/tw_prediction", "/tw_all_data"]
    for val in values:
        print(folder_basis + val)
        summarize_results(in_folder=folder_basis + val)

    # dist_train_over_strata()

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(message)s', # - %(levelname)s
                        level=logging.INFO, datefmt='%m/%d/%y %I:%M:%S %p')
    main()
