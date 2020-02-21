"""Data utilities."""
import sys
import logging
from collections import Counter
import numpy as np
import pandas as pd

SEED_SHUFFLE = 42
IMAGE_NET_LOC = "imagenet_data/data"
MNIST_FOLDER = "mnist_data"
# "ResNet50"
PROP_TEST_IMGNET = 0.1
# To fasten the loading I can avoid using the DataFrame module.
PREPROCESSED = False # True 
PREPROCESS_AND_SAVE = False

def get_ind_from_split(Z, pk):
    """Efficient way of finding the indices for Z to follow probs pk."""
    last_rem_card = Z.shape[0]+100000
    rem_card = Z.shape[0]
    stratas = list(range(pk.shape[0]))

    # Init of the indices lists per strata
    elem_per_strata = dict()
    card_per_strata = dict()
    for i, z in enumerate(Z):
        elem_per_strata[z] = elem_per_strata.get(z, list()) + [i]
        card_per_strata[z] = card_per_strata.get(z, 0) + 1
    for z in elem_per_strata:
        elem_per_strata[z] = np.array(elem_per_strata[z])
        np.random.shuffle(elem_per_strata[z])
    cards = np.array(list(card_per_strata.values()))

    i = 0
    selected_ind = list()
    while np.min(cards) > 0:
        # Sample from a categorical distribution
        n_chosen_per_strata = np.random.multinomial(np.min(cards), pk)
        for z, n in zip(stratas, n_chosen_per_strata):
            selected_ind += list(elem_per_strata[z][:n])
            elem_per_strata[z] = elem_per_strata[z][n:]
            card_per_strata[z] -= n
            rem_card -= n

        cards = np.array(list(card_per_strata.values()))
        if last_rem_card - rem_card > 10000:
            logging.info("Generating... ite %s with %s ind left",
                         i, rem_card)
            last_rem_card = rem_card
        i += 1
    return selected_ind

def discard_data_split(elems, Z, probs, seed=SEED_SHUFFLE):
    """
        Distributes the elems following the probs of modalities in Z.
    """
    assert np.all([arr.shape[0] == Z.shape[0] for arr in elems])

    np.random.seed(seed)
    ind = get_ind_from_split(Z, probs)
    np.random.seed()

    return [arr[ind] for arr in elems]

# ----- Dataset loading functions -----

### ---- MNIST loading ----

def load_MNIST():
    """Returns train_img, train_lab, test_img, test_lab."""
    fold = MNIST_FOLDER
    def get_file(x):
        res = np.load(open("{}/{}.npy".format(fold, x), "rb"))
        if x.endswith("_img"):
            res = res.astype(float)/255
        if x.endswith("_lab"):
            res = np.where(res)[1]
        return res
    return (get_file(x)
            for x in ["train_img", "train_lab", "test_img", "test_lab"])

def load_preprocess_MNIST(params):
    """
        Loads and preprocess MNIST.
    """
    # Required parameters:
    assert np.all([x in params for x in ["p_z_train"]])

    train_img, train_lab, test_img, test_lab = load_MNIST()

    # Command to add a constant:
    train_img = np.hstack([train_img, np.ones((train_img.shape[0], 1))])
    test_img = np.hstack([test_img, np.ones((test_img.shape[0], 1))])

    # X, Z, Y
    elems_train = [train_img, train_lab, train_lab]
    elems_test = [test_img, test_lab, test_lab]

    elems_train = discard_data_split(elems_train, train_lab,
                                     params["p_z_train"])
    elems_test = discard_data_split(elems_test, test_lab,
                                    params["p_z_test"])

    Y_train = elems_train[2]
    Y_test = elems_test[2]
    classes = range(0, 10)

    n_train = Y_train.shape[0]
    n_test = Y_test.shape[0]

    params["p_y_train"] = np.array([(Y_train == c).sum()/n_train
                                    for c in classes])
    params["p_y_test"] = np.array([(Y_test == c).sum()/n_test
                                   for c in classes])

    return elems_train, elems_test

### ---- ImageNet loading ----

def load_preprocess_ImageNet(params):
    """
        Loads and preprocesses ImageNet.
    """
    train_name = "df_train_st"
    val_name = "df_val_st"
    # test_name = "df_test_st"
    if not PREPROCESSED:
        # Reading it all
        logging.info("ImageNet - Reading val data... %s/%s.csv",
                     IMAGE_NET_LOC, val_name)
        df_test = pd.read_csv("{}/{}.csv".format(IMAGE_NET_LOC, val_name))
        # The test part of the data was removed, for lack of ground truth.
        # logging.info("Loading ImageNet - Reading test data...")
        # df_test = pd.read_csv("{}/{}.csv".format(IMAGE_NET_LOC, test_name))
        # pd.concat(, ignore_index=True) # [, df_test]
        logging.info("ImageNet - Reading train data... %s/%s.csv",
                     IMAGE_NET_LOC, train_name)
        df_train = pd.read_csv("{}/{}.csv".format(IMAGE_NET_LOC, train_name))

        logging.info("Loading ImageNet - Processing...")
        all_synset = np.unique(
            np.concatenate([df_train["SYNSET"].values,
                            df_test["SYNSET"].values]))
        all_strata = np.unique(
            np.concatenate([df_train["STRATA"].values,
                            df_test["STRATA"].values]))
        synset_to_no = {synset: i for i, synset in enumerate(all_synset)}
        strata_to_no = {strata: i for i, strata in enumerate(all_strata)}

        feat_cols = [col for col in df_train.columns
                     if col.startswith("DIM")]

        # Define train values
        X_train = np.array(df_train[feat_cols].values)
        Y_train = np.array([synset_to_no[syn]
                            for syn in df_train["SYNSET"]])
        Z_train = np.array([strata_to_no[sta]
                            for sta in df_train["STRATA"]])

        # Define test values
        X_test = np.array(df_test[feat_cols].values)
        Y_test = np.array([synset_to_no[syn] for syn in df_test["SYNSET"]])
        Z_test = np.array([strata_to_no[sta] for sta in df_test["STRATA"]])

        if PREPROCESS_AND_SAVE:
            names = ["X_train", "Y_train", "Z_train",
                     "X_test", "Y_test", "Z_test"]
            values = [X_train, Y_train, Z_train, X_test, Y_test, Z_test]
            
            for name, val in zip(names, values):
                out_loc = "{}/{}/{}.npy".format(IMAGE_NET_LOC, "preprocessed",
                                                name)
                np.save(open(out_loc, "wb"), val)
    else:
        out_loc = IMAGE_NET_LOC + "/preprocessed/"

        # Define train values
        logging.info("Loading ImageNet - Reading train data...")
        X_train = np.load(open(out_loc + "X_train" + ".npy", "rb"))
        Y_train = np.load(open(out_loc + "Y_train" + ".npy", "rb"))
        Z_train = np.load(open(out_loc + "Z_train" + ".npy", "rb"))

        # Define test values
        logging.info("Loading ImageNet - Reading test data...")
        X_test = np.load(open(out_loc + "X_test" + ".npy", "rb"))
        Y_test = np.load(open(out_loc + "Y_test" + ".npy", "rb"))
        Z_test = np.load(open(out_loc + "Z_test" + ".npy", "rb"))

    elems_train = (X_train, Z_train, Y_train)
    elems_test = (X_test, Z_test, Y_test)

    all_synset = np.unique(np.concatenate([Y_train, Y_test]))
    all_strata = np.unique(np.concatenate([Z_train, Z_test]))

    # - Calculating the original probabilities p_z's
    c_z_train = Counter(Z_train)
    c_z_test = Counter(Z_test)

    n_train = Y_train.shape[0]
    n_test = Y_test.shape[0]
    n_synsets = len(all_synset)
    n_strata = len(all_strata)

    p_z_train = np.array([c_z_train[i] for i in range(n_strata)])/n_train
    p_z_test = np.array([c_z_test[i] for i in range(n_strata)])/n_test

    if params["discard_data"]:
        # - Inducing asymmetries on the p_z's
        np.random.seed(params["seed_shuffle_p_z"])
        strata_asym = params["strata_asym_coeff"]
        prob_z_ratio = np.array([strata_asym**(-k/n_strata)
                                 for k in range(0, n_strata)])
        np.random.shuffle(prob_z_ratio)
        np.random.seed()

        # Heuristic way to do strata asymmetry.
        p_z_train = p_z_test*prob_z_ratio
        p_z_train = p_z_train/p_z_train.sum()

        logging.info("Loading ImageNet - Generating asymmetric strata sample...")
        elems_train = discard_data_split(elems_train, Z_train, p_z_train,
                                         seed=params["seed_shuffle_p_z"])
        logging.info("Loading ImageNet - Generated symmetric strata sample...")

    params["p_z_train"] = p_z_train
    params["p_z_test"] = p_z_test

    # - Calculating the probabilities and p_y's
    X_train, Z_train, Y_train = elems_train

    c_y_train = Counter(Y_train)
    c_y_test = Counter(Y_test)

    n_train = Y_train.shape[0]

    params["p_y_train"] = np.array([c_y_train[i]
                                    for i in range(n_synsets)])/n_train
    params["p_y_test"] = np.array([c_y_test[i]
                                   for i in range(n_synsets)])/n_test

    params["p_test"] = n_test/(n_test + n_train)

    logging.info("Output statistics: n_train = %s / n_test = %s",
                 n_train, n_test)

    logging.info("Loading ImageNet - Done !")
    return elems_train, elems_test
