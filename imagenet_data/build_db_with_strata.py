"""
    Uses the file data/strata_for_synsets.json to generate the files
    data/df_train_st.csv and data/df_val_st.csv used for learning.
"""
import datetime
import json
import pandas as pd

def build_db_with_strata(in_db_name="df_train.csv",
                         strata_for_synsets="strata_for_synsets.json",
                         out_db_name="df_train_st.csv"):
    """Builds the strata for the database."""
    # Load the strata file
    with open(strata_for_synsets, "rt") as f:
        d_st_4_syn = json.load(f)
    print("Reading CSV...")
    df = pd.read_csv(in_db_name)
    print("Done reading CSV.")
    n_labels = len(df["SYNSET"])
    list_strata = list()
    for i, l in enumerate(df["SYNSET"]):
        list_strata.append("_".join(d_st_4_syn[l]))
        if i % 10000 == 0:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("{} - Done {:.2f} %".format(time, ((i+1)/n_labels)*100))
    df["STRATA"] = list_strata
    print("Saving the csv - {}".format(datetime.datetime.now().ctime()))
    df.to_csv(out_db_name, index=None)

if __name__ == "__main__":
    print("#### Working on train")
    build_db_with_strata(in_db_name="data/df_train.csv",
                         strata_for_synsets="data/strata_for_synsets.json",
                         out_db_name="data/df_train_st.csv")
    print("#### Working on val")
    build_db_with_strata(in_db_name="data/df_val.csv",
                         strata_for_synsets="data/strata_for_synsets.json",
                         out_db_name="data/df_val_st.csv")
    # print("#### Working on test")
    # build_db_with_strata(in_db_name="df_test.csv",
    #                      strata_for_synsets="strata_for_synsets.json",
    #                      out_db_name="df_test_st.csv")
