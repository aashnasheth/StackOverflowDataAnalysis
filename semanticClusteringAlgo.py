# Databricks notebook source
# MAGIC %sh
# MAGIC /databricks/python/bin/python -m pip install nltk
# MAGIC /databricks/python/bin/python -m pip install --upgrade pip
# MAGIC /databricks/python/bin/python -m nltk.downloader all
# MAGIC /databricks/python/bin/python -m pip install bs4

# COMMAND ----------

#import statements
from bs4 import BeautifulSoup
from nltk import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from time import time
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text
import pandas as pd
from sklearn.cluster import KMeans

# COMMAND ----------

#some errors have numbers and punctuation along with letters: like 0x800A03EC which is reason for custom tokenizer
def tokenize_only(text):
    text = BeautifulSoup(text, 'html.parser').get_text() #parse out the htmk
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:  
        if ((len(token) > 3) & (token.isupper() | token.islower())): #using is upper and is lower to check that there are letters in the string 
            if ((token == "add-in") | (token == "add-ins") | (token == "addins")): #standardizing words that cloud data 
                token = "addin"
            if ((token == "installed") | (token == "installing")):
                token = "install"
            if ((token == "outlook.mailitem")):
                token = "mailitem"
            if ((token == "js") | (token == "office.js")):
                token = "javascript"
            if ("=" not in token):
                filtered_tokens.append(token)
    return filtered_tokens

# COMMAND ----------

#performs lsa (tfidf + svd) and KMeans, takes dataframe, columnName, and number of clusters 
def lsaAndKMeans(dfName, columnName, nc, dimensions): 
    #TFIDF
    print("Extracting features from the training dataset "
          "using a sparse vectorizer")
    my_stop_words = text.ENGLISH_STOP_WORDS.union(["yyyy", "application"]) 
    vectorizer = TfidfVectorizer(max_df=0.5, #can only appear in max 1/2 of the docs (if appears more, likely not helpful)
                                 min_df=10, #must appear minimum of 10 times 
                                 tokenizer = tokenize_only,  
                                 stop_words=my_stop_words)
                                     #ngram_range=(1,3))
    X = vectorizer.fit_transform(dfName[columnName]).toarray() #tfidf matrix 
    
    dist = 1 - cosine_similarity(X) #used for 2D plotting later on 
    
    #NARRATION
    print("n_samples: %d, n_features: %d" % X.shape)
    print()
    print("Performing dimensionality reduction using LSA")
    t0 = time()
   
    #SVD
    svd = TruncatedSVD(dimensions) #ideal SVD for large data is between 100 and 500
    normalizer = Normalizer(copy=False)
    svd_pipeline = make_pipeline(svd, normalizer)
    print(X.shape)
    X = svd_pipeline.fit_transform(X) #running svd 
    
    #NARRATION
    print("done in %fs" % (time() - t0))
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))
    print(X.shape)
    
    #KMeans
    km = KMeans(n_clusters=nc)
    pred = km.fit_predict(X)
    clusters = km.labels_.tolist()
    docList = dfName[columnName].tolist() 
    docsClusterDict = {columnName: docList, 'cluster': clusters}
    frame = pd.DataFrame(docsClusterDict, index = [clusters] , columns = [columnName, 'cluster'])
    
    print (frame['cluster'].value_counts())
    
    return svd, km, vectorizer, frame, X, dist #returns a ton of objects that are helpful for other methods

# COMMAND ----------

def printKMeans(svd, km, vectorizer, nc, columnName):
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()

    for i in range(nc):
        print("Cluster %d: " % i, end='')
        for ind in order_centroids[i, :10]: #num of words outputted 
            print(' %s' % terms[ind], end=',')
        print()
        
        #comment out below code if you do not want documents to print along with the clusters they belong to 
        print("Cluster %d documents:" % i, end='') 
        for item in frame.ix[i][columnName].values.tolist()[:10]:
            print(' %s,' % item, end='')
            print()
        print()

# COMMAND ----------

from datetime import datetime, timedelta, date

currentDate = date.today()
previousDate = currentDate - timedelta(days=2) #ability to filter how which data you'd like to use 

#USE CODE BELOW TO RUN WITH SPECIFIC TAG TITLES/BODIES  
df = spark.table("default.table_stackoverflowdata")
docsdf = df.filter(df.current_date == previousDate).filter(df.queried_tag == "outlook-addin").select("title")
docs_df = docsdf.toPandas() 
docs_df.title[0] #checking data 

#USE THE CODE BELOW TO RUN ALGO WITH ALL ADDIN TITLES (duplicates manually removed... add yyyy and application as stop words)
#df = spark.table("default.nodupsaddintitles_csv")
#titlesdf = df.select("title")
#titles_df = titlesdf.toPandas() 
#titles_df.title[0]

# COMMAND ----------

#calling methods  and storing outputs 
svd, km, vectorizer, frame, X, dist = lsaAndKMeans(docs_df, "title", 8, 200)
printKMeans(svd, km, vectorizer, 8, "title") #perform elbow method to determins optmial number of clusters (or use guess and check)

# COMMAND ----------

#MAKING SENSE OF CLUSTERS 
new_df = frame[frame['cluster'] != 0] #exclude junk cluster (determine which cluster to excluse depending on most populated cluster)
svd, km, vectorizer, frame, X, dist = lsaAndKMeans(new_df, "title", 7, 200) #rerun methods with one less cluster
printKMeans(svd, km, vectorizer, 7, "title")

# COMMAND ----------

# DBTITLE 1,Elbow Method
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#dont want to do any sort of pre-procesing bc tfidif is supposed to be the "normalization"

Sum_of_squared_distances = []
K = range(1,20)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X) #data_transformed 
    Sum_of_squared_distances.append(km.inertia_)

plt.clf()
plt.figure(figsize=(8,6))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')

display(plt.show())

# COMMAND ----------

# DBTITLE 1,Generating some Frequency Plots
  import matplotlib.pyplot as plt

  plt.clf()
  plt.cla()
  plt.close()

  array = []
  df = new_df[new_df['cluster'] == 1] #choose a cluster to generate frequency plot of 
  titleList = list(df.body)
  for item in titleList:
    array.extend(tokenize_only(item))
  
  for word in array:
    if word not in my_stop_words:
        words_ns.append(word)
    
  plt.clf()
  freqdist1 = nltk.FreqDist(array)
  freqdist2 = nltk.FreqDist(dict(freqdist1.most_common()[:25]))
  fig = freqdist2.plot()
  display(fig.figure)
