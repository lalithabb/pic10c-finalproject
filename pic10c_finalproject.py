import os
import sys
import pandas as pd
from shutil import copyfile
import nltk
# nltk.download('punkt')
import numpy as np
    
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
    return shuffled_df

# calculates word count; this function 
# assumes that the dataframe has the following columns: index, filename, subdir, movedfilename
def countwords(df):
    # extend shuffled_df dataframe 
    word_list = []
    for row in df[['movedfilename', 'tokens']].iterrows():
        rownum = row[1]
        for token in rownum.tokens:
            word_list.append((rownum.movedfilename, token))
    word_table = pd.DataFrame(word_list, columns = ['movedfilename', 'words'])
    
    # word count
    wordcount = word_table.groupby('movedfilename').words.value_counts().to_frame().rename(columns={'words':'termcounts'})
    # print(wordcount.head())
    return wordcount

# calculates term frequency using the following equation: Tf = numwords/numtxtfiles
# calculates inverse document frequency using the following equation: idf = log(totaldocs/(numdocs with term t))
# assumes that the incoming dataframe has the following columns: movedfilename, words, and nwords
def calculate_tfidf(word_table):
    # retrieve word count
    wordtable = countwords(word_table)
    
    # calculate term frequency
    totalwords = wordtable.groupby(level=0).sum().rename(columns = {'termcounts':'nwords'})
    termfrequency = wordtable.join(totalwords)
    termfrequency['termfrequency'] = termfrequency.termcounts/termfrequency.nwords
    
    # document count
    doc_count = len(totalwords)
    
    # number of unique documents that each word appears in
    idf_table = wordtable.reset_index(drop=False)
    idf_table = idf_table.groupby('words').movedfilename.nunique().to_frame().rename(columns={'movedfilename':'nfiles_word'}).sort_values('nfiles_word')
    
    # calculate idf
    idf_table['idf'] = np.log(doc_count/idf_table.nfiles_word.values)
    
    # join term frequency and idf in dataframe
    tfidf_table = termfrequency.join(idf_table)
    tfidf_table['tf_idf'] = tfidf_table.termfrequency * tfidf_table.idf
    tfidf_table = tfidf_table.reset_index(drop=False)
    # sort tf_idf in descending order
    tfidf_table = tfidf_table.groupby('movedfilename').apply(lambda x:x.sort_values('tf_idf', ascending=False))
    # print(tfidf_table[tfidf_table['movedfilename'] == 'shuffleddata/2.txt'])
    return tfidf_table
    
    # prepare tf_idf dataset
    
# initialize clusters
def kmeans_initialization():
    return

# main function
def main():
    shuffled_df = prepare_bbc_dataset()
    tfidf_table = calculate_tfidf(shuffled_df)
    print(tfidf_table.head())

# call main function
if __name__ == "__main__":
    main()



    
    


