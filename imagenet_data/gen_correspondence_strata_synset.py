"""
    Interprets the xml file, and adds the strata information alongside the mat
    information.
    L-n strata (depth in the imageNet structure file that we have for the synset.)
"""
import sys
import json
import xml.etree.ElementTree
import numpy as np
import requests
from datetime import datetime
import pandas as pd

def wnid_below_list(synset):
    """
        Get all of the wnid under the synset by exploring all of the descendent
        nodes. Does not return the wnid of the synset.
    """
    res_list = [a.attrib["wnid"] for a in synset.findall("synset")]
    for s in synset.findall("synset"):
        res_list = res_list + wnid_below_list(s)
    return res_list

def get_strata_all_synsets(strata_txtfile="data/strata_list.txt",
                           struct_file="data/structure_released.xml"):
    """Returns the strata of all synsets."""
    print("Opening strata list... " + datetime.now().ctime())
    with open(strata_txtfile, "rt") as f:
        strata_list = list(map(lambda x: x.strip(), f.readlines()))


    print("Getting all data synsets... " + datetime.now().ctime())
    with open("data/all_data_synsets.txt", "rt") as f:
        all_synsets = set(map(lambda x: x.strip(), f.readlines()[1:]))

    print("Exploring tree... " + datetime.now().ctime())
    root = xml.etree.ElementTree.parse(struct_file
                                      ).getroot().findall('synset')[0]
    dres = dict()
    n_strata = len(strata_list)
    i_strata = 0
    for s_strata in strata_list:
        print("Strata {} out of {} - ".format(i_strata, n_strata)
              + datetime.now().ctime())
        cur_depth_synsets = root.findall('synset')
        # [a for a in root.findall('synset') if a.attrib["wnid"] != "fa11misc"]
        cur_depth = 0
        while cur_depth_synsets:
            for master_synset in cur_depth_synsets:
                master_wnid = master_synset.attrib["wnid"]
                list_wnid = wnid_below_list(master_synset)
                # print(s_strata, master_wnid)
                # print(master_wnid == s_strata)
                if master_wnid == s_strata:
                    synsets_under = all_synsets.intersection(list_wnid)
                    # print(list_wnid, all_synsets)
                    # import ipdb; ipdb.set_trace()
                    for syn in synsets_under:
                        dres[syn] = dres.get(syn, set()).union({s_strata})
            # print("Current depth: {} - ".format(cur_depth)
            #       + datetime.now().ctime())
            cur_depth_synsets = [a for s in cur_depth_synsets
                                 for a in s.findall("synset")]
            cur_depth += 1
        i_strata += 1
    for syn in all_synsets.difference(set(dres.keys())):
        dres[syn] = ["no_strata"]
    return dres

def main(in_fold="data", trial=False):
    d_res = get_strata_all_synsets()
    outfile = ("{}/strata_for_synsets.json".format(in_fold) if not trial else
               "{}/strata_for_synsets_trial.json".format(in_fold))
    outfile2 = ("{}/synsets_for_stratum.json".format(in_fold) if not trial else
                "{}/synsets_for_stratum_trial.json".format(in_fold))
    d_res2 = dict()
    for key in d_res:
        all_strata = sorted(list(d_res[key]))
        d_res[key] = all_strata
        meta_strata = "_".join(all_strata)
        d_res2[meta_strata] = d_res2.get(meta_strata, list()) + [key]
    with open(outfile, "wt") as f:
        json.dump(d_res, f)
    with open(outfile2, "wt") as f:
        json.dump(d_res2, f)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "trial":
        main(trial=True)
    else:
        main(trial=False)
