"""
    Asymmetric strata with ImageNet.
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

SEED_SHUFFLE_P_Z = 42
DEFAULT_ROOT_FOLDER = "tmp"
DEFAULT_MODEL_TYPE = "linear"
DEFAULT_HL_SIZE = 1524 # Average between 2048 and 1000

# Defines what the plots should look like for all models
DEFAULT_LIM_TOP_MLP = (0, 0.20)
DEFAULT_LIM_ACC_MLP = (0.10, 0.45)
DEFAULT_LIM_COST_MLP = (0.8, 3) # 2.50)

DEFAULT_LIM_TOP_LIN = DEFAULT_LIM_TOP_MLP
DEFAULT_LIM_ACC_LIN = DEFAULT_LIM_ACC_MLP
DEFAULT_LIM_COST_LIN = DEFAULT_LIM_COST_MLP
# DEFAULT_LIM_TOP_LIN = (0, 0.20)
# DEFAULT_LIM_ACC_LIN = (0.1, 0.45)
# DEFAULT_LIM_COST_LIN = (0.75, 2.2)
# DEFAULT_LIM_TOP_LIN = (0, 0.30)
# DEFAULT_LIM_ACC_LIN = (0.20, 0.45)
# DEFAULT_LIM_COST_LIN = (2, 6)

# DIC_TRY_REG = {0 : 0.001, 1 : 0.01, 2 : 0.1, 3 : 1, 4 : 0.005, 5 : 0.0025}
DIC_TRY_REG = {0 : 0.01, 1 : 0.005, 2 : 0.1, 3 : 1, 4 : 0.05}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-P1", "--only_plot_type_1",
                        help="Only plots with plot type 1.",
                        action="store_true")
    parser.add_argument("-P2", "--only_plot_type_2",
                        help="Only plots with plot type 2.",
                        action="store_true")
    parser.add_argument("-O", "--only_uniform",
                        help="Run only the uniform fitting.",
                        action="store_true")
    parser.add_argument("-S", "--only_stratum",
                        help="Run only the stratum fitting.",
                        action="store_true")
    parser.add_argument("-TR", "--try_regularizations",
                        help="Try a set of pre-defined regularizations.",
                        action="store_true")
    parser.add_argument("-s", "--strata_asym",
                        help="Strata asymmetry coefficient, between 0 and 1.",
                        type=float, default=0.05)
    parser.add_argument("-m", "--model_type",
                        help="Model type (linear or mlp).", type=str,
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


    base_params = {
        "std_init": 0.01,
        "l2_coeff": 0.001,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "n_ite": 20000,
        "batch_size": 1000,
        "display_step": 100,
        "seed_shuffle_p_z": SEED_SHUFFLE_P_Z,
        "strata_asym_coeff": args.strata_asym,
        "dynamics_lim_cost": None,
        "dynamics_lim_acc": None,
        "discard_data": True,
        # defined later
        "p_z_train": None,
        "p_z_test": None,
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
        lim_top = DEFAULT_LIM_TOP_LIN
        lim_acc = DEFAULT_LIM_ACC_LIN
        lim_cost = DEFAULT_LIM_COST_LIN
    else:
        lim_top = DEFAULT_LIM_TOP_MLP
        lim_acc = DEFAULT_LIM_ACC_MLP
        lim_cost = DEFAULT_LIM_COST_MLP

    if args.only_plot_type_1:
        print("Starting plotting of results...")
        pu.plot_quantiles(args.root_exp_folder,
                          subfolds=("uniform", "stratum", "prediction"),
                          lim_acc=lim_acc,
                          lim_cost=lim_cost)
        print("Done plotting of results !")
        sys.exit()

    if args.only_plot_type_2:
        print("Starting plotting of results...")

        sf_style = {"uniform": {"color": "blue",
                                "label": "Unif."},
                    "prediction": {"color": "grey",
                                   "label": "Class"},
                    "stratum": {"color": "green",
                                "label": "Strata"},
                    "all_data": {"color": "darkred",
                                 "label": "No bias"},
                   }
        pu.plot_quantiles_cost_n_miss(
            args.root_exp_folder,
            subfolds=("uniform", "stratum", "prediction", "all_data"),
            sf_style=sf_style,
            lim_top=lim_top, lim_acc=lim_acc, lim_cost=lim_cost)
        print("Done plotting of results !")
        sys.exit()

    if not os.path.exists(args.root_exp_folder):
        os.makedirs(args.root_exp_folder)

    if os.listdir(args.root_exp_folder):
        choice = ""
        while choice not in ("y", "n"):
            choice = input("Erase folder {} ?(y/n) ".format(
                args.root_exp_folder))
        if choice == "y":
            shutil.rmtree(args.root_exp_folder)

    if args.only_uniform:
        cand_weights = ["uniform"]
    elif args.only_stratum:
        cand_weights = ["stratum"]
    else:
        cand_weights = ["uniform", "stratum", "prediction", "all_data"]

    # Loading the data
    out_folder = "{}/tw_{}/run_{:02d}".format(
        args.root_exp_folder, cand_weights[0], args.index_first_run)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        shutil.copy(os.path.basename(__file__),
                    "{}/image_net_strata_exp.py".format(out_folder))

    logging.basicConfig(filename='{}/data_loading.log'.format(out_folder),
                        format='%(asctime)s - %(message)s', # - %(levelname)s
                        level=logging.INFO, datefmt='%m/%d/%y %I:%M:%S %p',
                        filemode="w")

    train_data, test_data = du.load_preprocess_ImageNet(base_params)
    if "all_data" in cand_weights:
        params = base_params.copy()
        params["discard_data"] = False
        train_data_all, test_data_all = du.load_preprocess_ImageNet(params)

    for i_run in range(args.index_first_run, args.index_last_run):
        print("Run {}".format(i_run))
        for type_weight in cand_weights:
            # The next two lines allow to change the logging file.
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            out_folder = "{}/tw_{}/run_{:02d}".format(
                args.root_exp_folder, type_weight, i_run)

            params = base_params.copy()
            if args.try_regularizations:
                params["l2_coeff"] = DIC_TRY_REG[i_run]

            if type_weight != "all_data":
                params["type_weight"] = type_weight
                lu.run_exp_loaded_data(train_data, test_data, params,
                                       out_folder=out_folder,
                                       db_name="ImageNet")
            else:
                params["type_weight"] = "uniform"
                lu.run_exp_loaded_data(train_data_all, test_data_all, params,
                                       out_folder=out_folder,
                                       db_name="ImageNet")

if __name__ == "__main__":
    main()
