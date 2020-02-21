# Weighted Empirical Risk Minimization: Transfer Learning based on Importance Sampling

Contains the code associated to the ESANN 2020 publication: Weighted Empirical
Risk Minimization: Sample Selection Bias Correction based on Importance
Sampling.

The experiemnts requires a lot of disk space (>200Go) and a lot of RAM (>20Go) since
they load all of the encodings of ImageNet directly in memory.

## Requirements

* Python (used version was 3.5.2).

* Standard python libraries:
  ` abc, argparse, array, collections, datetime, json, logging, os, sys, shutil, struct, xml`.

* Non-standard python packages:
  ` keras-mxnet (2.2.2), scipy (1.1.0), matplotlib (2.2.3), numpy (1.14.5), 
  pandas (0.23.4), requests (2.18.4), tensorflow-gpu (1.9.0)`.

* Bash: `wget (1.19.5), gawk (4.1.4), gzip (1.6)`.

## Preliminaries

First step is to download the images of ILSVRC2012
Get the files from http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads

* Training images (Task 1 & 2).  138GB. MD5: 1d675b47d978889d74fa0da5fadfb00e
* Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622
* Development kit (Task 1 & 2).  2.5MB.

And put them in the folder data using symbolic links if required.

## Files that are supposed to be run (interface)

### Illustrations on simple distributions of Section 7.1

**`sim_exp.py`**: Creates the figures for Section 7.1.

* Usage: `python sim_exp.py`

### MNIST experiments of Section 7.3

**`mnist_data/make_MNIST_npy_format.sh`**: Downloads and converts the MNIST dataset.

* Usage: `cd mnist_data; bash make_MNIST_npy_format.sh; cd ..`

**`mnist_class_exp.py`**: Deals with the MNIST experiments.

* Usage 1: Starts 25 runs of the experiments with the model specified by `-m`:
```bash
  python mnist_class_exp.py -m (linear|mlp) -r mnist_exp/ -f 0 -l 25
```
* Usage 2: Plots the experiments that were run there:
```bash
  python mnist_class_exp.py -P2 -m (linear|mlp) -r mnist_exp/ 
```

### ImageNet experiments of Section 4

**`imagenet_data/do_all_preprocessing.sh`**: Prepares the files above, takes a long time to run.

* Usage: `cd imagenet_data; bash do_all_preprocessing.sh; cd ..`

**`imagenet_strata_exp.py`**: Deals with the ImageNet experiments.

* Usage 1: Starts 25 runs of the experiments with the model specified by `-m`:
```bash
  python imagenet_strata_exp.py -m (linear|mlp) -r imagenet_exp/ -f 0 -l 25
```
* Usage 2: Plots the experiments that were run there:
```bash
  python imagenet_strata_exp.py -P2 -m (linear|mlp) -r imagenet_exp/ 
```

**`imagenet_analysis.py`**: Summarizes the information contained in a folder after a run.

* Usage: `python imagenet_analysis.py <folder>`

## Files that are not supposed to be run (backend)

Those are `data_utils.py`, `learning_utils.py`, `model_utils.py`, `plot_utils.py`,
`imagenet_data/build_db_with_strata.py`, `imagenet_data/convert_mapping_list.py`,
`imagenet_data/data_summaries.py`, `imagenet_data/fuse_resnet50_to_df.py`,
`imagenet_data/gen_correspondence_strata_synset.py`, `imagenet_data/img_to_features.py`,
`mnist_data/convert_mnist_to_npy.py`.
