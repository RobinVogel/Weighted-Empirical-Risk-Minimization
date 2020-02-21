"""Converts the meta file provided by ImageNet to a list of mappings."""
from scipy.io import loadmat

def main():
    m = loadmat("data/ILSVRC2012_devkit_t12/data/meta.mat")
    with open("data/ILSVRC2012_mapping.txt", "wt") as f:
        for a in m["synsets"]:
            ind = a[0][0][0][0]
            synset = a[0][1][0]
            f.write("{} {}\n".format(ind, synset))

if __name__ == "__main__":
    main()
