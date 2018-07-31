import argparse
print("...Importing Libraries...")

import matplotlib.pyplot as plt
import numpy as np

import cv2
from skimage.feature import hog
from skimage import data, color, exposure

from sklearn.cluster import DBSCAN
from sklearn import metrics

import pandas as pd

import string
import os

print("...Parsing Arguments...")
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, help="Folder containing sequence of images")
parser.add_argument("--csv_name", type=str, help="Name of CSV file to save, including extension")
parser.add_argument("--scaling_factor", type=float, help="Scaling factor to resize images", default=1.0)
parser.add_argument("--eps", type=float, help="EPS for DBScan", default=11.5)
parsed = parser.parse_args()
image_dir, csv_name, scaling_factor, eps = parsed.image_dir, parsed.csv_name, parsed.scaling_factor, parsed.eps

# Ensure consistency of directories
if image_dir[-1] != "/":
    image_dir += "/"

def extract_features_from_image(file, features_only=True, scaling_factor=0.2):

    """
    Extracts HoG features from an image in a given directory (after scaling)

    Inputs:
    - file: A directory to an image file in an accepted format by opencv

    Returns:
    - fd: A numpy array of features extracted from the image
    - img_gs: The original image in grayscale
    - hog_image: An image of the HoG feature representation of the input image
    """

    img = cv2.imread(file)

    if scaling_factor == 1.0:
        img_r = img
    elif scaling_factor > 1:
        img_r = cv2.resize(img, None, fx = scaling_factor, fy = scaling_factor, interpolation = cv2.INTER_CUBIC)
    else:
        img_r = cv2.resize(img, None, fx = scaling_factor, fy = scaling_factor, interpolation = cv2.INTER_AREA)

    img_gs = color.rgb2gray(img_r)
    fd, hog_image = hog(img_gs, orientations=16, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)

    if features_only:
        return(fd)
    else:
        return(img_gs, hog_image, fd)

def create_full_fp(imageName, dir=image_dir):
    image_path = dir + imageName
    return(image_path)

def create_output_csv(labels, filename):

    """
    Takes clustering results and creates a results dataframe and outputs a .csv file in the current directory.

    Inputs:
    - labels: A numpy array of labels assigned to frames (in chronological order).
    - filename: The name of the .csv file to be saved

    Returns:
    - None
    """

    keyframe_ind = [labels[i] != labels[i-1] for i, val  in enumerate(labels)]
    keyframe_idxs = [i for i, val in enumerate(keyframe_ind) if val==True]
    keyframe_filenames = ["%06d" % (i+1) + ".jpg" for i, val in enumerate(keyframe_ind) if val==True]
    keyframe_scenes = labels[keyframe_idxs]
    keyframe_scenes_ascii = [string.ascii_lowercase[i] for i in keyframe_scenes]
    result = pd.DataFrame([keyframe_filenames, keyframe_scenes_ascii]).transpose()
    result.columns = ['keyframe', 'scene id']
    filepath = os.getcwd()
    result.to_csv(filepath + '/' + filename)

if __name__ == "__main__":

    # Example unix input from parent directory of this .py file:
    # python keyframe_detection.py --image_dir=/Users/Sean/Documents/Wirewax/Sequence1/ --csv_name=Sequence1d.csv --scaling_factor=0.2 --eps=1.0

    images = map(create_full_fp, os.listdir(image_dir))

    print("...Extracting Features from Images...")
    feats = map(extract_features_from_image, images)
    features = np.asarray(feats)

    print("...Clustering Frames...")
    db = DBSCAN(eps=eps, min_samples=3).fit(features)
    labels = db.labels_

    create_output_csv(labels, csv_name)
    print("...Keyframe Detection Complete. Output saved to current directory " + image_dir + "/" + csv_name + "...")
