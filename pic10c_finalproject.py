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
from sklearn.metrics import pairwise_distances

#for visualization of results
# from sklearn.cluster import KMeans
    
# transform dataset into a dataframe, parse text, and tokenize (using nltk)
# returns "shuffled" BBC dataset
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
    print(shuffled_df.head())
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
    print(wordcount.head())
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
    print(tfidf_dataset.head())
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
 
# reformat the tfidf_dataset to have proper information for creation of sparse matrix
# input: tfidf_dataset created in calculate_tfidf()
# output: dataset with columns: # of columns, original file path, # of rows, tfidf value, word
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
    print(tfidf_dataset.head())
    return tfidf_dataset         

#reduce dictionary to top 10 unique words from each text file
#def reduce_dictionary(df, num_entries):
    #for row in df:
        #df['tfidf'] = {key:value for key,value in df['tfidf'].items()[0:num_entries]}
        #n_items = take(num_entries, df['tfidf'].items())
    #return n_items

# convert the dataframe into a sparse matrix (which has a lot of zero-entires and a few nonzero-entries, identifies row_index, col_index, and value of nonzero-entries alone)
# input: tfidf_dataset from reformat_tfidf_table()
# output: sparse matrix containing tfidf values
def create_sparse_matrix(tfidf_dataset):
    # convert rows into an array
    row = np.array(tfidf_dataset['rownum'])
    # convert
    col = np.array(tfidf_dataset['colnum'])
    tfidf = np.array(tfidf_dataset['tfidf'])
    rownum = max(tfidf_dataset['rownum']) + 1
    colnum = max(tfidf_dataset['colnum']) + 1
    tfidf_matrix = csr_matrix((tfidf, (row, col)), shape=(rownum, colnum))
    tfidf_matrix = normalize(tfidf_matrix)
    print(tfidf_matrix)
    return tfidf_matrix

# this function randomly initializes k centroids to cluster the rest of the data around
# input: tfidf: sparse matrix with tfidf values, k: number of desired centroids, and random seed 
# output: an array of initial centroids
def initialize_centroids(tfidf, k, seed=None):
    #the goal is to randomly choose k data values as the centroids
    #in this case, k will be 5 because there were 5 original themes in the bbc dataset
    if seed is not None: # used for debugging purposes - can set seed to not None to achieve consistent results
        np.random.seed(seed)
    # set n equal to number of data values in tfidf sparse matrix
    n = tfidf.shape[0]  
    # randomly pick 5 centroid values between 0 and n
    randomk = np.random.randint(0, n, k)
    centroids = tfidf[randomk,:].toarray()
    print(centroids)
    return centroids

# this function assigns data values to clusters by calculating the euclidean distance between each centroid/value pair
# input: tfidf sparse matrix, centroids calculated in initialize_centroids() function
# output: assigned clusters
def assign_clusters(tfidf, centroids):
    # use euclidean distance to determine how far each data value is from each centroid
    distFromCentroid = pairwise_distances(tfidf, centroids, metric='euclidean')
    #returns indices of the closest values to each centroid
    clustAssignment = np.argmin(distFromCentroid, axis=1)
    print(clustAssignment)
    return clustAssignment

# reassign each centroid to the mean of all the cluster datapoints
# input: tfidf sparse matrix, number of centroids, cluster assignment from assign_clusters()
# output: new centroids
def revise_centroids(tfidf, k, clustAssignment):
    newCentroids = []
    for i in range(k):
        member_data_points = tfidf[clustAssignment==i]
        # calculate mean of data
        centroid = member_data_points.mean(axis=0)
        # convert matrix to array
        centroid = centroid.A1
        newCentroids.append(centroid)
    newCentroids = np.array(newCentroids)
    print(newCentroids)
    return newCentroids

# initializes, assigns, and revises clusters; checks for cluster convergence
# input: tfidf sparse matrix, number of centroids, number of iterations, verbose=True prints changed data points 
# modeled after UofWash kmeans clustering explanation
def kmeans(tfidf, k, init_centroids, iterations, verbose=False):
    centroids = init_centroids[:]
    prev_assignment = None
    
    for iter in range(iterations):        
        if verbose:
            print(iter)
        # call assign_clusters()
        clust_assignment = assign_clusters(tfidf, centroids)
        # call revise_centroids()
        centroids = revise_centroids(tfidf, k, clust_assignment)
        # check for cluster convergence
        if prev_assignment is not None and \
          (prev_assignment==clust_assignment).all():
            break
        if prev_assignment is not None:
            num_changed = np.sum(prev_assignment!=clust_assignment)
            if verbose:
                print('{0:5d} changed cluster assignments.'.format(num_changed))   
        prev_assignment = clust_assignment[:]
    print(centroids)
    print(clust_assignment)
    return centroids, clust_assignment

#def plot_clusters(tfidf_matrix, centroids, clust_assignment):
#    
#    return

# main function
#def main():
shuffled_df = prepare_bbc_dataset()
tfidf_table = calculate_tfidf(shuffled_df)
tfidf_dataset = reformat_tfidf_table(tfidf_table)
tfidf_matrix = create_sparse_matrix(tfidf_dataset)
init_centroids = initialize_centroids(tfidf_matrix, 5, seed=1)
clust_assignment = assign_clusters(tfidf_matrix, init_centroids)
revised_centroids = revise_centroids(tfidf_matrix, 5, clust_assignment)
centroids, clust_assignment = kmeans(tfidf_matrix, 5, init_centroids, 3, verbose=True)


#print(tfidf_dataset.head())

#reduced_tfidf = reduce_dictionary(tfidf_table, 10)
#print(reduced_tfidf.head())
#sparse_tfidf = df_to_sparse_matrix(tfidf_table, tfidf)

# call main function
#if __name__ == "__main__":
#    main()



    
    


