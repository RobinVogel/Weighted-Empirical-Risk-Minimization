"""
    Computes information about the data, specifically the original strata
    distribution, plots an histogram of the data distribution over strata
    and the list of the strata sorted by its number of occurences in the
    train.
"""
import sys
import logging
import datetime
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

IMAGE_NET_LOC = "data"

def compute_original_strata_proportion():
    """Computes the proportions over all strata in the original databases."""
    train_name = "df_train_st_only"
    val_name = "df_val_st_only"
    # Reading it all
    logging.info("Loading imagenet - Reading val data...")
    df_test = pd.read_csv("{}/{}.csv".format(IMAGE_NET_LOC, val_name))
    logging.info("Loading imagenet - Reading train data...")
    df_train = pd.read_csv("{}/{}.csv".format(IMAGE_NET_LOC, train_name))

    logging.info("Loading imagenet - Processing...")
    all_strata = np.unique(
        np.concatenate([df_train["STRATA"].values,
                        df_test["STRATA"].values]))
    strata_to_no = {strata: i for i, strata in enumerate(all_strata)}

    Z_train = np.array([strata_to_no[sta] for sta in df_train["STRATA"]])
    Z_test = np.array([strata_to_no[sta] for sta in df_test["STRATA"]])

    # - Calculating the original probabilities p_z's
    c_z_train = Counter(Z_train)
    c_z_test = Counter(Z_test)

    n_train = Z_train.shape[0]
    n_test = Z_test.shape[0]
    n_strata = len(all_strata)

    p_z_train = np.array([c_z_train[i] for i in range(n_strata)])/n_train
    p_z_test = np.array([c_z_test[i] for i in range(n_strata)])/n_test

    train_file = "{}/{}.npy".format(IMAGE_NET_LOC, "original_p_z_train")
    np.save(open(train_file, "wb"), p_z_train)
    test_file = "{}/{}.npy".format(IMAGE_NET_LOC, "original_p_z_val")
    np.save(open(test_file, "wb"), p_z_test)

def dist_train_over_strata():
    """Plots an histogram of the train distribution over the strata."""
    print("Reading the df...")
    df = pd.read_csv("{}/df_train_st_only.csv".format(IMAGE_NET_LOC))
    d = Counter(df["STRATA"].values)
    print("Plotting...")
    plt.figure(figsize=(9, 4))
    di = d.items()
    ks = [k for k, v in di]
    vs = [v for k, v in di]

    sort_ind = np.argsort(vs)[::-1]
    ks = [ks[i] for i in sort_ind]
    vs = [vs[i] for i in sort_ind]
    with open("data/strata_sorted.txt", "wt") as f:
        for k in ks:
            f.write("\n".join(k.split("_")) + "\n\n\n")
    plt.bar(ks, vs)
    plt.tight_layout()
    plt.xticks(rotation=45, rotation_mode="anchor", ha="right")
    plt.subplots_adjust(left=0.1, bottom=0.4)
    plt.grid()
    plt.ylabel("# instances (thousands)")
    plt.xlabel("strata")
    plt.savefig("summaries/train_dist_over_strata.pdf", format="pdf")
    print("Done !")

def main():
    compute_original_strata_proportion()
    dist_train_over_strata()
