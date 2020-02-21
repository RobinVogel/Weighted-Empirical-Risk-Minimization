"""
    Creates dataframes with the train and validation instances, with the
    information "SYNSET", "ID", "DIM_i" where i ranges from 0001 to 2048.
"""
import sys
import os
from datetime import datetime
import numpy as np

def fuse_all(in_fold, folder):
    """Fuse the .npy files as expected."""
    assert folder in ("trial", "val_trial", "train", "val", "test")

    basedir = "{}/ILSVRC2012_ResNet50_encodings/{}/".format(in_fold, folder)

    print("Getting the list of all files - " + datetime.now().ctime())
    all_files = sorted(os.listdir(basedir))
    n_files = len(all_files)
    print("Starting merging the files - " + datetime.now().ctime())
    i = 0
    n_feat = None
    first_row = True
    def columns(n_feat):
        return ["ID", "SYNSET"] + [
            "DIM_{:04d}".format(i) for i in range(0, n_feat)]

    if folder in ("val", "test", "test_trial"):
        base_fname = in_fold + "/ILSVRC2012_{}_gt_syn.txt"
        fname = base_fname.format(folder)
        gt_file = open(fname, "rt")

    with open("{}/df_{}.csv".format(in_fold, folder), "wt") as f_out:
        for f in all_files:
            feat = np.load(basedir + "/" + f).ravel()
            if first_row:
                n_feat = len(feat)
            str_val_num = ["{:.2f}".format(v) if v >= 0.01 else "0"
                           for v in feat]
            val_num = ",".join(str_val_num)

            if folder in ("train", "trial"):
                synset, index = f.split("_")
                index = int(index.split(".")[0])
            else:
                synset = gt_file.readline().strip("\n")
                index = i

            if first_row:
                f_out.write(",".join(columns(n_feat)) + "\n")
                first_row = False
            f_out.write(",".join([str(index), synset, val_num]) + "\n")

            if i % 1000 == 0:
                print("Done {} files out of {} - ".format(i, n_files)
                      + datetime.now().ctime())
            i += 1

def correct_gt_files(in_fold, type_data):
    """Corrects the text ground truth file from indices to synset string."""
    assert type_data in ("val", "test", "test_trial")
    tdata = "validation" if type_data == "val" else type_data
    class_fname = "{}/ILSVRC2012_{}_ground_truth.txt".format(in_fold, tdata)
    with open(class_fname, "rt") as f:
        all_classes = list(map(lambda x: x.strip(), f.readlines()))
    mapping_fname = "{}/ILSVRC2012_mapping.txt".format(in_fold)
    with open(mapping_fname, "rt") as f:
        mapping_dict = {l.strip().split(" ") for l in f.readlines()}
    result_fname = "{}/ILSVRC2012_{}_gt_syn.txt".format(in_fold, type_data)
    with open(result_fname, "wt") as f:
        for l in all_classes:
            f.write(mapping_dict[l] + "\n")

if __name__ == "__main__":
    if sys.argv[1] == "val":
        correct_gt_files("data", "val")
    # correct_gt_files("data", "test")
    # correct_gt_files("data", "test_trial")
    print("Working with {}".format(sys.argv[1]))
    fuse_all("data", sys.argv[1])
