"""
    Encoding of the ILSVRC images with ResNet50.
"""
import os
import sys
from datetime import datetime
import numpy as np

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

# Modify this folder,
INET_DIR = "data/"
# with a folder that contains:
# The following text files:
#   ILSVRC2012.txt
#   ILSVRC2012_val.txt
#   ILSVRC2012_test.txt if necessary
# The following folders with images, extracted from the zip file of
# 
# It will store the encodings in the following folders:
#   ILSVRC2012_ResNet50_encodings/train
#   ILSVRC2012_ResNet50_encodings/val
#   ILSVRC2012_ResNet50_encodings/test

def encode_example(output="logits"):
    """Encode an example, meant for testing."""
    assert output in {"logits", "features"}
    if output == "logits":
        model = ResNet50(weights='imagenet')
    if output == "features":
        base_model = ResNet50(weights='imagenet')
        model = Model(inputs=base_model.input,
                      outputs=base_model.get_layer("avg_pool").output)
        # Get the config with model.get_config()

    img_path = "example.jpg"
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    # Returns a 1,000 - dimensional array with the input image:
    features = model.predict(x)

    features.dump("example")

def convert_to_encodings(in_data_dir, in_data_list, out_data_dir,
                         mod_print=1000, output="features"):
    """Convert to deep neural network encodings a list of pictures."""
    #pylint: disable-msg=too-many-locals
    assert output in {"logits", "features"}
    if output == "logits":
        model = ResNet50(weights='imagenet')
    if output == "features":
        base_model = ResNet50(weights='imagenet')
        model = Model(inputs=base_model.input,
                      outputs=base_model.get_layer("avg_pool").output)

    n_files = 0
    for _ in open(in_data_list, "rt"):
        n_files += 1
    with open(in_data_list, "rt") as f_in_data:
        in_data_path = f_in_data.readline().strip()
        n_elems = 0
        while in_data_path:
            img_path = in_data_dir + "/" + in_data_path
            out_data_path = (out_data_dir + "/" +
                             os.path.splitext(in_data_path)[0] + ".npy")
            out_dir = os.path.split(out_data_path)[0]
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)

            # Returns a 1,000 - dimensional array with the input image:
            features = model.predict(x)

            features.dump(out_data_path)
            if n_elems % mod_print == 0:
                print("Processed {} elements out of {} / ".format(
                    n_elems, n_files) + datetime.now().ctime())
            n_elems += 1
            in_data_path = f_in_data.readline().strip()

if __name__ == "__main__":
    # encode_example()

    # Encode ImageNet:
    print("Working on {}".format(sys.argv[1]))
    if sys.argv[1] == "train":
        convert_to_encodings(INET_DIR, INET_DIR + "/ILSVRC2012.txt",
                             INET_DIR + "/ILSVRC2012_ResNet50_encodings/train")
    if sys.argv[1] == "val":
        convert_to_encodings(INET_DIR,
                             INET_DIR + "/ILSVRC2012_val.txt",
                             INET_DIR + "/ILSVRC2012_ResNet50_encodings/val")
    if sys.argv[1] == "test":
        convert_to_encodings(INET_DIR,
                             INET_DIR + "/ILSVRC2012_test.txt",
                             INET_DIR + "/ILSVRC2012_ResNet50_encodings/test")
