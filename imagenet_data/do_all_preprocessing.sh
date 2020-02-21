#!/bin/bash
gt_file="data/ILSVRC2012_devkit_t12.tar.gz"
train_file="data/data/ILSVRC2012_img_train.tar"
test_file="data/ILSVRC2012_img_val.tar"
if [ ! -f "$gt_file" ] || [ ! -f "$train_file" ] || [ ! -f "$test_file" ]; then
    echo "First step is to download the images of ILSVRC2012"
    echo "Get the files from http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads"
    echo "Training images (Task 1 & 2).  138GB. MD5: 1d675b47d978889d74fa0da5fadfb00e"
    echo "Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622"
    echo "Development kit (Task 1 & 2).  2.5MB."
    echo "And put them in the folder data using symbolic links if required."
fi

echo "1. Extract the files - Requires scipy"
python convert_mapping_list.py
tar xzvf "$gt_file"
tar xzvf "$train_file"
tar xzvf "$test_file"
mv "data/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt" "./data/"

echo "2. Encode the files - Requires keras, numpy"
find "ILSVRC2012/" | grep "\.JPEG" > "data/ILSVRC2012.txt"
find "ILSVRC2012_validation/" | grep "\.JPEG" > "data/ILSVRC2012_val.txt"
mkdir -p "data/ILSVRC2012_ResNet50_encodings/train/"
mkdir -p "data/ILSVRC2012_ResNet50_encodings/val/"
python img_to_features.py val &
python img_to_features.py train &
wait

echo "3. Fuses all of the npy files to create big DataFrames."
python fuse_resnet50_to_df.py val &
python fuse_resnet50_to_df.py train &
wait

echo "4. Generate the correspondance between data and strata."
wget "http://image-net.org/archive/words.txt"
mv "words.txt" "data/"
echo "The file data/strata_list.txt was created by hand by looking up"
echo "the categories in http://www.image-net.org/about-stats described "
echo "by words and finding the corresponding synsets in data/words.txt."
wget "http://www.image-net.org/api/xml/structure_released.xml"
mv "structure_released.xml" "data/"
{
    awk -F "," '(NR>1){print $2}' "data/df_train.csv" | uniq; 
    awk -F '(NR>1){print $2}' "data/df_val.csv" | uniq ;
} | sort | uniq  > "data/all_data_synsets.txt"
python gen_correspondence_strata_synset.py

echo "5. Generates the databases with the strata information using the"
echo "list of synsets in data/strata_list.txt."
python build_db_with_strata.py

echo "6. Generates the summaries of the data present in the paper."
awk -F "," '{print $NF}' "data/df_train_st.csv" > "data/df_train_st_only.csv"
awk -F "," '{print $NF}' "data/df_val_st.csv" > "data/df_val_st_only.csv"
python data_summaries.py
bash get_def_synsets.sh > summaries/def_synsets.txt
