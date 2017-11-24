import nltk
import os
import sys
import pandas as pd

# set working directory
dataset_dir = 'bbc'


# check if directory exists and loop through subdirectories
if os.path.isdir(dataset_dir):
    for rdir, subdirs, filenames in os.walk(dataset_dir):
        for subdir in subdirs:
            for rdir1, subdirs1, filenames1 in os.walk(dataset_dir + '/' + subdir):
                for filename in filenames1:
                    newfilename = subdir + filename
                    print(newfilename)
else:
    print("Directory " + dataset_dir + " does not exist.")
    sys.exit(1)
# randomly shuffle files in directory
# for file in subdir:
