# libraries for text parsing
import os
import sys
import pandas as pd
from shutil import copyfile
import nltk
# nltk.download('punkt')
import numpy as np

#packages for k-means clustering
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
    
# transform dataset into a dataframe, parse text, and tokenize
def prepare_bbc_dataset():
    # set working directory
    dataset_dir = 'bbcss'
    #dataset_dir = 'bbc'

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
    
    # randomly shuffle all rows (set seed to 1) in dataframe and reindex
    shuffled_df = dataframe.sample(frac=1, random_state=1)
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
    idf_table = idf_table.groupby('words').movedfilename.nunique().to_frame().rename(columns={'movedfilename':'nfiles_word'}).sort_values('nfiles_word', ascending=False)
    
    # calculate idf
    idf_table['idf'] = np.log(doc_count/idf_table.nfiles_word.values)
    
    # join term frequency and idf in dataframe
    tfidf_table = termfrequency.join(idf_table)
    tfidf_table['tf_idf'] = tfidf_table.termfrequency * tfidf_table.idf
    tfidf_table = tfidf_table.reset_index(drop=False)
    # sort tf_idf in descending order
    # tfidf_table = tfidf_table.groupby('movedfilename').apply(lambda x:x.sort_values('tf_idf', ascending=False))
    # print(tfidf_table[tfidf_table['movedfilename'] == 'shuffleddata/2.txt'])
    # tfidf_table = tfidf_table.reset_index(drop=False)
    # prepare tf_idf dataset
    #convert tf_idf table to dictionary
    tfidf_dataset = tfidf_table.groupby('movedfilename').apply(lambda x: dict(zip(x['words'], x['tf_idf']))).reset_index().rename(columns={0:'tfidf'})
    # tfidf_dataset = tfidf_dataset.merge(tfidf_dataset, word_table, on='movedfilename', how='inner')
    return tfidf_dataset

# Syntactic sugar for converting a set of dictionary keys to numeric indices 
# and vice versa
def dict_to_index_table (di):
    ''' do, dor = dict_to_index_table(di)
    
    Input parameters: 
        di:  Dictionary; e.g. {'a': 'John', 'b': 'Mary' ...}
        
    Output: 
        do: Dictionary; e.g. {'a': 0, 'b': 1, ...} 
        dor: Reverse dictionary; e.g. {0: 'a', 1:'b', ...}
        
    This function takes a dictionary and returns a mapping from the 
    dictionary keys to a set of numeric indices in the range of the length 
    of the dictionary. 
    
    '''
    doi = list({k for d in di for k in d.keys()})
    do = {x:i for i,x in enumerate(doi)}
    dor = {i:x for i,x in enumerate(do)}
    return do, dor
 
def df_to_sparse_matrix(df, column_name):
    ''' tfidf_numeric, wordmap, indexmap = \
                df_to_sparse_matrix(df, dict_column, wordindex)
        
        Input Parameters:
            df          : tfidf dataset with tfidf in a dictionary in column_name
            column_name : Column spec that has the tfidf values for each word
            
        Output:
            tfidf_numeric    : sparse matrix of tfidf values
            map_index_to_word: dictionary of indices mapping to word
            
        This function returns a sparse matrix of TF-IDF values and returns a 
        map from indices to words. The rows correspond to the rows of the 
        dataframe while the columns are the words in each document.
    '''
    row_indices = range(len(df))
    map_word_to_index, map_index_to_word = dict_to_index_table(df[column_name])
    
    return row_indices
    
def reformat_tfidf_table(tfidf_table):
    wordindex, indexword = dict_to_index_table(tfidf_table['tfidf'])
    lens=[len(item) for item in tfidf_table['tfidf']]
    tfidf_table['rownum'] = pd.Series(range(len(tfidf_table)))
    tfidf_dataset=pd.DataFrame(\
                               {'movedfilename': \
                                np.repeat(tfidf_table['movedfilename'].values, lens), \
                                'rownum': np.repeat(tfidf_table['rownum'].values, lens), \
                                'colnum': tfidf_table.tfidf.apply(lambda x: pd.Series([wordindex[k] for k in list(x.keys())]))\
                                .stack()\
                                .reset_index(level=1, drop=True).astype('int'), \
                                'tfidf': tfidf_table.tfidf.apply(lambda x: pd.Series([v for v in list(x.values())]))\
                                .stack()\
                                .reset_index(level=1, drop=True)})
    tfidf_dataset['word'] = tfidf_dataset.colnum.apply(lambda x: indexword[x])  
    return tfidf_dataset         

#reduce dictionary to top 10 unique words from each text file
#def reduce_dictionary(df, num_entries):
    #for row in df:
        #df['tfidf'] = {key:value for key,value in df['tfidf'].items()[0:num_entries]}
        #n_items = take(num_entries, df['tfidf'].items())
    #return n_items

# convert the dataframe into a sparse matrix (which has a lot of zero-entires and a few nonzero-entries, identifies row_index, col_index, and value of nonzero-entries alone)
def create_sparse_matrix(tfidf_dataset):
    #tfidf_dataset = tfidf_dataset.drop(['movedfilename', 'word'])
    #matrix = tfidf_dataset.as_matrix()
    #rownum = tfidf_dataset['rownum'].toarray()
    row = np.array(tfidf_dataset['rownum'])
    col = np.array(tfidf_dataset['colnum'])
    tfidf = np.array(tfidf_dataset['tfidf'])
    rownum = max(tfidf_dataset['rownum']) + 1
    colnum = max(tfidf_dataset['colnum']) + 1
    tfidf_matrix = csr_matrix((tfidf, (row, col)), shape=(rownum, colnum)).toarray()
    print(tfidf_matrix)
    return tfidf_matrix

# initialize clusters
#def kmeans_initialization():
#    return

# main function
#def main():
shuffled_df = prepare_bbc_dataset()
tfidf_table = calculate_tfidf(shuffled_df)
tfidf_dataset = reformat_tfidf_table(tfidf_table)
tfidf_dataset = create_sparse_matrix(tfidf_dataset)
#print(tfidf_dataset.head())

#reduced_tfidf = reduce_dictionary(tfidf_table, 10)
#print(reduced_tfidf.head())
#sparse_tfidf = df_to_sparse_matrix(tfidf_table, tfidf)

# call main function
#if __name__ == "__main__":
#    main()



    
    


