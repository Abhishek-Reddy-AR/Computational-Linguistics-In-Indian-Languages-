#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required module

from nltk.tokenize import WhitespaceTokenizer
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from unidecode import unidecode
import math
import re
import os
import pickle
import pandas as pd
from sys import argv


# In[2]:


#loading the stored data required for running the three systems

tf= pickle.load(open('stored_tf', "rb"))
df= pickle.load(open('stored_df', "rb"))
idf= pickle.load(open('stored_idf', "rb"))
tf_idf= pickle.load(open('stored_tf_idf', "rb"))
idf_bm25= pickle.load(open('stored_idf_bm25', "rb"))
doc_len=pickle.load(open('stored_doc_len',"rb"))
files=pickle.load(open('file_names',"rb"))


# ### We perform preprocessing of the query in the same way we have done for a document in the corpus

# In[3]:


#function to convert query into list of tokens
def Tokenizer(query):
    
    #tokenizing based on whitespace character
    tk = WhitespaceTokenizer()
    query_tokens=tk.tokenize(query)
    return query_tokens


# In[4]:


#function to remove stopwords from query tokens
def RemoveStopWords(query_tokens):
    
    #getting the stopwords
    stop = stopwords.words('english')
    
    #to store tokens that are not stopwords
    new_list=[]
    
    #for each token in the list
    for i in query_tokens:
        
        #if the token is not a stopword adding it to new list
        if i not in stop:
            new_list.append(i)
    return new_list


# In[5]:


#function that does stemming on the query tokens
def Stemmer(query_tokens):
    
    #to store stemmed tokens
    stem_list=[]
    
    #we use porterstemmer
    ps = PorterStemmer()
    
    #for each token
    for i in query_tokens:
        
        #adding stemmed token to the stem list
        stem_list.append(ps.stem(i))
    return stem_list


# In[6]:


def preprocess(query):
    
    #replacing non ascii characters 
    query=unidecode(query)
    #converting to lowercase
    query= query.lower()
    
    #removing the html tags
    clean = re.compile('<.*?>') 
    query= re.sub(clean, '', query)
    
    #replacing '\n' ,'\r' and punctuations with ' '(space)
    query = query.replace("\n"," ").replace("\r"," ")
    query = query.replace("'s"," ")
    punctuationList = '!"#$%&\()*+,-./:;<=>?@[\\]^_{|}~'
    x = str.maketrans(dict.fromkeys(punctuationList," "))
    query = query.translate(x)
    
    #performing tokenization,stopword removal and stemming
    query_tokens=Tokenizer(query)
    query_tokens=RemoveStopWords(query_tokens)
    query_tokens=Stemmer(query_tokens)
    
    return query_tokens


# ### Q2. Running IR Systems

# In[7]:


#function to calculate the boolean score 
# args: query_tokens , documnent frequencies, top_k:no of top documents to be retrieved

def boolean_score(query_tokens,df,top_k):
    
    #to store the retrieved doc ids
    docs=[]
    
    #for each token in the query
    for token in query_tokens:
        #if the token is present in df
        if token in df.keys():
            #we add the doc ids that contain that token into the list
            docs.extend(df[token]['l'])
    
    #getting the unique docids in the list
    s=set(docs)
    #to store the count of each docid in the docs list
    final_count=[]
    #for each docid
    for i in s:
        #appending its count i.e the no of query tokens present in that partiular docid
        final_count.append([docs.count(i),files[i]])
    #sorting in dec order based on count values
    final_count.sort(reverse=True)
    #to store only the top k docids
    doc_ids=[]
    for i in range(top_k):
        doc_ids.append(final_count[i][1])
    return doc_ids


# In[8]:


#function to calculate td idf score
#args :  query_tokens , tf_idf ,idf ,top_k:no of top documents to be retrieved

def tf_idf_score(query_tokens,tf_idf,idf,top_k):
    #to store cosine similarity scores 
    cos_scores=[]
    #getting the normalized term freq of tokens in the query
    tf_q={}
    for i in set(query_tokens):
        tf_q[i]=query_tokens.count(i)/len(query_tokens)
    
    #getting the tf idf values for the tokens in the query
    tf_idf_q={}
    for i in tf_q.keys():
        if i in idf.keys():
            tf_idf_q[i]=tf_q[i]*idf[i]
    
    #getting the euclidean norm of query tokens  tf_idf values 
    a=math.sqrt(sum([x*x for x in tf_idf_q.values()]))
    
    #for each doc 
    for i in range(len(tf_idf)):
        #if there are no tokens in that doc 
        if(len(tf_idf[i])==0):
            #then ist cosine sim score is zero
            cos_scores.append([0,files[i]])
            continue
        #else if there are tokens in the doc
        curr_score=float(0.0)
        
        #for each token in the query
        for j in tf_idf_q.keys():
            #if that token is present in that doc
            if j in tf_idf[i].keys():
                #multiplying the tf idf values of that token (tf_idf of corpus * tf_idf of query)
                curr_score+=tf_idf[i][j]*tf_idf_q[j]
        
        #getting the euclidean norm of doc tokens  tf_idf values 
        b=math.sqrt(sum([x*x for x in tf_idf[i].values()]))
        
        #calculating the current documents cosine similarity score
        curr_score=(curr_score)/(a*b)
        
        #appending the docid and the score
        cos_scores.append([curr_score,files[i]])
    
    #sorting in dec order based on the cosine similarity scores
    cos_scores.sort(reverse=True)
    #returning the top k docids
    doc_ids=[]
    for i in range(top_k):
        doc_ids.append(cos_scores[i][1])
    return doc_ids


# In[9]:


#function to calculate the bm25 score
#args  :  query_tokens , tf ,idf_bm25 , doc lengths , top_k:no of top documents to be retrieved ,k1 (hyperparameter), b(hyperparameter)

def bm25_score(query_tokens,tf,idf_bm25,doc_len,top_k,k1,b):
    
    #calculating the avg doc len
    avg_doc_len = sum(doc_len) / len(doc_len)
    
    #to store scores
    bm25_scores=[]
    #for each doc
    for i in range(len(tf)):
        #getting curr docs term freqs dict and its length
        curr_score=0.0
        curr_doc_len = doc_len[i]
        frequencies = tf[i]
        
        #for each query token
        for token in query_tokens:
            #if token not in the curr doc , skip
            if token not in frequencies:
                continue
            #if the token is present in the curr doc
            freq = frequencies[token]
            
            #calculating bm25 score of that token and adding it to bm25 score of the curr doc
            numerator = idf_bm25[token] * freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * curr_doc_len / avg_doc_len)
            curr_score += (numerator / denominator)
        #appending the docid and the bm25 score of that doc
        bm25_scores.append([curr_score,files[i]])
    #sorting the list based on bm25 scores in dec order
    bm25_scores.sort(reverse=True)
    #returning the top k docids
    doc_ids=[]
    for i in range(top_k):
        doc_ids.append(bm25_scores[i][1])
    return doc_ids


# ### Q3. Reading Queries and storing the output from the systems in QRels format

# In[13]:


#reading the queries
query_file = argv[1]
query_f = open(query_file, 'r')
queries = query_f.readlines()


# In[19]:


#seperating the queryid and query
queries_list=[]
for query in queries:
    queries_list.append(query.split("\t",1))


# In[20]:


#removing last element of the list if it contains only '/n'
if len(queries_list[-1])==1:
    queries_list=queries_list[:-1]


# In[21]:


#creating dataframes to store the qrels format results of each of the systems
boolean_qrels=pd.DataFrame(columns=['QueryId', 'Iteration', 'DocId', 'Relevance'])
tf_idf_qrels=pd.DataFrame(columns=['QueryId', 'Iteration', 'DocId', 'Relevance'])
bm25_qrels=pd.DataFrame(columns=['QueryId', 'Iteration', 'DocId', 'Relevance'])


# In[22]:


#for each query
for query in queries_list:
    #preprocessing the query
    query_tokens=list(preprocess(query[1]))
    
    #getting the top 10 docs ids using boolen based, tf idf based , bm25 based retrieval systems
    boolean_docids=boolean_score(query_tokens,df,5)
    tf_idf_docids=tf_idf_score(query_tokens,tf_idf,idf,5)
    bm25_docids=bm25_score(query_tokens,tf,idf_bm25,doc_len,5,1.5,0.75)
    
    #creating temporary dataframes to store the current query's retrieved docis
    tempdf1=pd.DataFrame(columns=['QueryId', 'Iteration', 'DocId', 'Relevance'])
    tempdf2=pd.DataFrame(columns=['QueryId', 'Iteration', 'DocId', 'Relevance'])
    tempdf3=pd.DataFrame(columns=['QueryId', 'Iteration', 'DocId', 'Relevance'])
    
    #assings the docids
    tempdf1.DocId=boolean_docids
    tempdf2.DocId=tf_idf_docids
    tempdf3.DocId=bm25_docids
    
    #assigns queryid , iteration,relevance values
    tempdf1.QueryId=tempdf2.QueryId=tempdf3.QueryId=query[0]
    tempdf1.Iteration=tempdf2.Iteration=tempdf3.Iteration=1
    tempdf1.Relevance=tempdf2.Relevance=tempdf3.Relevance=1
    
    #appendig these temporary dataframes into their respective retrieval system dataframes
    boolean_qrels=boolean_qrels.append(tempdf1)
    tf_idf_qrels=tf_idf_qrels.append(tempdf2)
    bm25_qrels=bm25_qrels.append(tempdf3)


# In[23]:


#storing the dataframes into csv files
boolean_qrels.to_csv('boolean_qrels.csv',index=False,header=False)
tf_idf_qrels.to_csv('tf_idf_qrels.csv',index=False,header=False)
bm25_qrels.to_csv('bm25_qrels.csv',index=False,header=False)


# In[ ]:




