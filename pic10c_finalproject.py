import os
import sys
import pandas as pd
from shutil import copyfile
import string
import nltk
nltk.download('punkt')

# set working directory
dataset_dir = 'bbc'

# create empty lists
dirlist = []
filelist = []
newfilelist = []

# check if directory exists and loop through subdirectories
if os.path.isdir(dataset_dir):
    for rdir, subdirs, filenames in os.walk(dataset_dir):
        for subdir in subdirs:
            for rdir1, subdirs1, filenames1 in os.walk(dataset_dir + '/' + subdir):
                for filename in filenames1:
                    newfilename = subdir + filename
                    dirlist.append(subdir)
                    filelist.append(filename)
else:
    print("Directory " + dataset_dir + " does not exist.")
    sys.exit(1)
    
# create dataframe with original file name, new file name, and subdirectory
dataframe = pd.DataFrame({'subdir': dirlist, 'filename': filelist})
# print(dataframe)

# randomly shuffle all rows in dataframe and reindex
shuffled_df = dataframe.sample(frac=1)
shuffled_df['movedfilename'] = range(1, len(shuffled_df) + 1)
# converting int to string 
shuffled_df['movedfilename'] = shuffled_df['movedfilename'].astype(str)
# create new file names and directory
shuffled_df['movedfilename'] = 'shuffleddata/' + shuffled_df['movedfilename'] + '.txt'
print(shuffled_df)

# copy original content to new shuffleddata directory
if not os.path.exists('shuffleddata'):
    os.makedirs('shuffleddata')
text = []
tokentext = []
print(shuffled_df['movedfilename'][257])
# parse lines
tokenizer = nltk.RegexpTokenizer(r'\w+')
for index, row in shuffled_df.iterrows():
    copyfile('bbc/' + row['subdir'] + '/' + row['filename'], row['movedfilename'])
    file = open(row['movedfilename'], "r")
    lines = file.read()
    text.append(lines)
    #tokenize text files
    tokens = tokenizer.tokenize(lines.lower())
    tokentext.append(tokens)
shuffled_df['raw_text'] = pd.Series(text)
shuffled_df['tokens'] = pd.Series(tokentext)

# tokenize text files

    
    


