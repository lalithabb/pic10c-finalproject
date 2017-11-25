import os
import sys
import pandas as pd
from shutil import copyfile
import nltk
nltk.download('punkt')
import numpy as np

# def download_bbc_dataset():
    
# transform dataset into a dataframe, parse text, and tokenize
def prepare_bbc_dataset():
    # set working directory
    dataset_dir = 'bbc'
    
    # create empty lists
    dirlist = []
    filelist = []
    
    # check if directory exists and loop through subdirectories
    if os.path.isdir(dataset_dir):
        for rdir, subdirs, filenames in os.walk(dataset_dir):
            for subdir in subdirs:
                for rdir1, subdirs1, filenames1 in os.walk(dataset_dir + '/' + subdir):
                    for filename in filenames1:
                        dirlist.append(subdir)
                        filelist.append(filename)
    else:
        print("Directory '" + dataset_dir + "' does not exist.")
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
    
    # copy original content to new shuffleddata directory
    if not os.path.exists('shuffleddata'):
        os.makedirs('shuffleddata')
    text = []
    tokentext = []
    # parse lines
    # format tokenizer to remove punctuation
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    for index, row in shuffled_df.iterrows():
        copyfile('bbc/' + row['subdir'] + '/' + row['filename'], row['movedfilename'])
        file = open(row['movedfilename'], "r")
        lines = file.read()
        text.append(lines)
        #tokenize text files, convert words to lowercase
        tokens = tokenizer.tokenize(lines.lower())
        tokentext.append(tokens)
    shuffled_df['raw_text'] = pd.Series(text)
    shuffled_df['tokens'] = pd.Series(tokentext)
    # print(shuffled_df.head())
    return shuffled_df

# calculates word count; this function 
# assumes that the dataframe has the following columns: index, filename, subdir, movedfilename
def countwords(df):
    # extend shuffled_df dataframe 
    word_list = []
    for row in df[['filename', 'subdir', 'movedfilename', 'tokens']].iterrows():
        rown = row[1]
        for token in rown.tokens:
            word_list.append((rown.filename, rown.subdir, rown.movedfilename, token))
    word_table = pd.DataFrame(word_list, columns = ['filename', 'subdir', 'movedfilename', 'words'])
    
    # word count
    wordcount = word_table.groupby('movedfilename').words.value_counts().to_frame().rename(columns={'words':'nwords'})
    print(wordcount.head())
    return word_table

# calculates term frequency using the following equation: Tf = numwords/numtxtfiles
# calculates inverse document frequency using the following equation: idf = log(totaldocs/(numdocs with term t))
# assumes that the dataframe has the following columns: movedfilename, words, and nwords
def calculate_tfidf(wordtable):
    # calculate term frequency
    totalwords = wordtable.groupby(level=0).sum().rename(columns = {'nwords':'nfiles'})
    termfrequency = wordtable.join(totalwords)
    termfrequency['termfrequency'] = termfrequency.nwords/termfrequency.nfiles
    print(termfrequency.head())
    
    # calculate inverse document frequency
    # idf = wordtable.groupby('words').movedfilename.nunique().to_frame().rename(columns={'books':'id'}).sortvalues('id')
    
    
# main function
def main():
    shuffled_df = prepare_bbc_dataset()
    # print(shuffled_df.head())
    word_table = countwords(shuffled_df)
    # print(word_table.head())
    calculate_tfidf(word_table)

# call main function
if __name__ == "__main__":
    main()



    
    


