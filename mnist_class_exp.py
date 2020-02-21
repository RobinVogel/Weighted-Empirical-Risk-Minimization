"""
    Asymmetric classes with MNIST.
"""
import sys
import os
import logging
import shutil
import argparse

import numpy as np

import learning_utils as lu
import data_utils as du
import plot_utils as pu

SEED_SHUFFLE_P_Z_TRAIN = 42
DEFAULT_ROOT_FOLDER = "tmp"
DEFAULT_MODEL_TYPE = "linear"
DEFAULT_HL_SIZE = 396 # Average between 28x28 and 10

# Defines what the plots should look like for all models
DEFAULT_LIM_ACC_LIN = (0.08, 0.15)
DEFAULT_LIM_COST_LIN = (0.45, 0.65) # Values used to be terrible (0, 2)

DEFAULT_LIM_ACC_MLP = (0.05, 0.10)
DEFAULT_LIM_COST_MLP = (0.4, 0.6) # (0.3, 0.8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-P1", "--only_plot_type_1",
                        help="Only plots with plot type 1.",
                        action="store_true")
    parser.add_argument("-P2", "--only_plot_type_2",
                        help="Only plots with plot type 2.",
                        action="store_true")
    parser.add_argument("-s", "--strata_asym",
                        help="Strata asymmetry coefficient, between 0 and 1.",
                        type=float, default=0.1)
    parser.add_argument("-m", "--model_type",
                        help="Model type.", type=str,
                        default=DEFAULT_MODEL_TYPE)
    parser.add_argument("-r", "--root_exp_folder",
                        help="Root folder for the experiments.", type=str,
                        default=DEFAULT_ROOT_FOLDER)
    parser.add_argument("-f", "--index_first_run",
                        help="Starting index of the runs.", type=int,
                        default=0)
    parser.add_argument("-l", "--index_last_run",
                        help="End index of the runs.", type=int,
                        default=100)
    parser.add_argument("-UBC", "--ub_cost", help="Upper bound cost.",
                        type=float, default=None)
    parser.add_argument("-LBC", "--lb_cost", help="Lower bound cost.",
                        type=float, default=None)
    parser.add_argument("-UBA", "--ub_acc", help="Upper bound accuracy.",
                        type=float, default=None)
    parser.add_argument("-LBA", "--lb_acc", help="Lower bound accuracy.",
                        type=float, default=None)
    args = parser.parse_args()

    p_z_train = np.array([args.strata_asym**(-k/10) for k in range(0, 10)])
    p_z_train = p_z_train/np.sum(p_z_train)
    np.random.seed(SEED_SHUFFLE_P_Z_TRAIN)
    np.random.shuffle(p_z_train)
    np.random.seed()

    base_params = {
        "std_init": 0.01,
        "l2_coeff": 0.01,
        "learning_rate": 0.01, "momentum": 0.9,
        "n_ite": 15000, # 10000,
        "batch_size": 1000,
        "display_step": 100,
        "p_z_train": p_z_train,
        "p_z_test": np.array([0.1]*10),
        "dynamics_lim_cost": None,
        "dynamics_lim_acc": None,
        # defined later
        "model_type": None,
        "type_weight": None
    }

    # If a limit is specified, you have to specify both.
    if args.lb_cost is None or args.ub_cost is None:
        base_params["dynamics_lim_cost"] = None
    else:
        base_params["dynamics_lim_cost"] = (args.lb_cost, args.ub_cost)

    if args.lb_acc is None or args.ub_acc is None:
        base_params["dynamics_lim_acc"] = None
    else:
        base_params["dynamics_lim_acc"] = (args.lb_acc, args.ub_acc)

    assert args.model_type in ("linear", "mlp")
    if args.model_type == "linear":
        base_params["model_type"] = "LINEAR"
    else:
        base_params["model_type"] = "MLP"
        base_params["hl_size"] = DEFAULT_HL_SIZE

    if args.model_type == "linear":
        lim_acc = DEFAULT_LIM_ACC_LIN
        lim_cost = DEFAULT_LIM_COST_LIN
    elif args.model_type == "mlp":
        lim_acc = DEFAULT_LIM_ACC_MLP
        lim_cost = DEFAULT_LIM_COST_MLP
    else:
        raise ValueError("Wrong model type !")

    if args.only_plot_type_1:
        print("Starting plotting of results...")
        pu.plot_quantiles(args.root_exp_folder,
                          subfolds=("uniform", "prediction"),
                          lim_acc=lim_acc, lim_cost=lim_cost)
        print("Done plotting of results !")
        sys.exit()

    if args.only_plot_type_2:
        print("Starting plotting of results...")
        pu.plot_quantiles_cost_n_miss(
            args.root_exp_folder, subfolds=("uniform", "prediction"),
            sf_style={"uniform": {"color": "blue", "label": "Uniform"},
                      "prediction": {"color":"green", "label": "Weighted"}
                     },
            lim_acc=lim_acc, lim_cost=lim_cost)
        print("Done plotting of results !")
        sys.exit()

    if not os.path.exists(args.root_exp_folder):
        os.makedirs(args.root_exp_folder)

    if not os.listdir(args.root_exp_folder):
        choice = ""
        while choice not in ("y", "n"):
            choice = input("Erase folder {} ?(y/n) ".format(
                args.root_exp_folder))
        if choice == "y":
            shutil.rmtree(args.root_exp_folder)

    # Loading the data
    cand_weights = ["uniform", "prediction"]
    out_folder = "{}/tw_{}/run_{:02d}".format(
        args.root_exp_folder, cand_weights[0], args.index_first_run)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        shutil.copy(os.path.basename(__file__),
                    "{}/mnist_class_exp.py".format(out_folder))

    logging.basicConfig(filename='{}/data_loading.log'.format(out_folder),
                        format='%(asctime)s - %(message)s', # - %(levelname)s
                        level=logging.INFO, datefmt='%m/%d/%y %I:%M:%S %p',
                        filemode="w")

    train_data, test_data = du.load_preprocess_MNIST(base_params)

    for i_run in range(args.index_first_run, args.index_last_run):
        print("Run {}".format(i_run))
        for type_weight in cand_weights:
            # The next two lines allow to change the logging file.
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            out_folder = "{}/tw_{}/run_{:02d}".format(
                args.root_exp_folder, type_weight, i_run)
            params = base_params.copy()
            params["type_weight"] = type_weight
            lu.run_exp_loaded_data(train_data, test_data, params,
                                   out_folder=out_folder, db_name="MNIST")

if __name__ == "__main__":
    main()
