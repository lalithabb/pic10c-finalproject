# PIC10C Project Description

**Goal:**
The idea is to classify the themes of documents using k-means clustering.

Please read below for updates made after the making of the video.
**Link to youtube video explanation of project: https://youtu.be/zYJkn0ypwFc

**Installation requirements (included in program):**
This program requires installation of 1) Python 3, 2) the Natural Language Toolkit, and 3) the BBC raw text dataset. Use of the BBC dataset included in this repository is recommended, as one character in BBC/sport/199.txt was change to comply with ASCII encoding (£ was changed to ‘pound’). 

A smaller version of this dataset, ‘bbcss’, has been included as well. The program was run on the ‘bbcss’ dataset due to the fact that the original dataset is quite large, and runtime is extremely long as a result. If you wish to test the program, I would suggest using the ‘bbcss’ dataset (the code should be set to this by default).

If you do wish to download the original dataset, it can be found here: http://mlg.ucd.ie/datasets/bbc.html
Under the ‘Datasets: BBC’ section, click on ‘Download raw text files’.

**The data:**
The BBC dataset contains raw text files sorted into one of 5 categories: politics, business, sports, entertainment, and tech. The first function (prepare_bbc_dataset()) moves all the text files out of these directories and randomly shuffles them in order to reclassify them later on.

**Program description:**
This program parses the words in the BBC text files, formats them, calculates the term frequency-inverse document frequency (TF-IDF), creates a sparse matrix, and performs k-means analysis on them. K-Means is a clustering algorithm that assigns cluster centers, or centroids, to classify similar values (in this case TF-IDF values). The algorithm aims to reduce cluster heterogeneity, and thus create tightly knit clusters of words. Formulas for tf-idf can be found in the comments above the calculate_tfidf() function. Further definitions and descriptions of the sparse matrix and other functions can also be found in the program comments.

The program prints the cluster themes upon completion.

**Connections to C++:**
The data is stored in containers, namely lists and dictionaries (which are equivalent to the std::map in C++). The preprocessing steps use a combination of generic algorithms (such as groupby, sort, etc.) and lambda functions to structure the data for easy analysis. These algorithms underlyingly use iterators to traverse through the containers.

**Progress:**
The formatting of text files and calculation of the TF-IDF took much longer than I expected, so I was not able to really manually code the k-means algorithm - instead I relied on libraries available in Python. If I had more time, I could have “manually” created the sparse matrix and performed the functions for k-means.

As of now, the program randomly initializes cluster centroids, calculates minimum Euclidean distances between each centroid and each data point, and recalculates the centroids based on the mean data points. This is problematic because if the randomly initialized centroids are too close together, the clusters will not be entirely accurate. Ideally, the k-means function should be performed iteratively so that all possible cluster centers/clusters are determined to converge on the “best” result. This means that the results as of now are not entirely accurate.

**Future directions:**
I originally intended to retrieve news/magazine articles from the Washington Post, New York Times, National Geographic websites, and classify those. The BBC dataset text files were easier to parse, so I used those instead. In the future, I would like to modify the program to handle articles retrieved from the internet. 

The kmeans() function could possibly have been implemented as a class rather than a function. I also could have hand-coded some of the functionality from scikit learn, but performance would have been compromised compared to the available libraries in Python. 

**UPDATE:** the below suggestion has been updated; the kmeans function returns the top data point for each cluster. The program identifies the following themes: sport, entertainment, tech, and business. It misclassifies the politics cluster as another sport cluster. 
Also, the intention is to compare the clusters generated by the algorithm to the original directory structure of the BBC dataset (see ‘The data’ section above). This can be achieved by comparing the top n tf-idf values in each cluster and determining whether they are of common themes to the BBC dataset directories or not. This has now been implemented.
