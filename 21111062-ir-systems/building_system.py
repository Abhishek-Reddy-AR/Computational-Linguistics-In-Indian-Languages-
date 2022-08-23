#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required module

from nltk.tokenize import WhitespaceTokenizer
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from unidecode import unidecode
from os.path import join
import math
import re
import os
import pickle
import io


# ### Q1.Functions for Processing of the Corpus

# In[6]:


#function to convert document into list of tokens
def Tokenizer(text):
    
    #tokenizing based on whitespace character
    tk = WhitespaceTokenizer()
    text=tk.tokenize(text)
    return text


# In[7]:


#function to remove stopwords from the token list
def RemoveStopWords(text_list):
    
    #getting the stopwords
    stop = stopwords.words('english')
    
    #to store tokens that are not stopwords
    new_list=[]
    
    #for each token in the list
    for i in text_list:
        
        #if the token is not a stopword adding it to new list
        if i not in stop:
            new_list.append(i)
    return new_list


# In[8]:


#function that does stemming on the token list
def Stemmer(text_list):
    
    #to store stemmed tokens
    stem_list=[]
    
    #we use porterstemmer
    ps = PorterStemmer()
    
    #for each token
    for i in text_list:
        
        #adding stemmed token to the stem list
        stem_list.append(ps.stem(i))
    return stem_list


# In[9]:


def preprocess(text):
    
    #replacing non ascii characters 
    text=unidecode(text)
    #converting to lowercase
    text= text.lower()
    
    #removing the html tags
    clean = re.compile('<.*?>') 
    text= re.sub(clean, '', text)
    
    #replacing '\n' ,'\r' and punctuations with ' '(space)
    text = text.replace("\n"," ").replace("\r"," ")
    text = text.replace("'s"," ")
    punctuationList = '!"#$%&\()*+,-./:;<=>?@[\\]^_{|}~'
    x = str.maketrans(dict.fromkeys(punctuationList," "))
    text = text.translate(x)
    
    #performing tokenization,stopword removal and stemming
    token_list=Tokenizer(text)
    token_list=RemoveStopWords(token_list)
    token_list=Stemmer(token_list)
    
    return token_list


# In[3]:


#getting the current working directory
cwd = os.getcwd() 
#storing all the documents names present in the corpus
files = os.listdir('english-corpora/') 

#list to store the contents of each document
corpus=[]
#for each document
for i in range(len(files)):
    #opening the doc
    #fileopen = open('english-corpora\\'+files[i], 'r',encoding="utf8")
    fileopen = io.open(join('english-corpora/',str(files[i])),'r',encoding='utf-8',errors='ignore')
    #reading and appending the contents of doc into text list
    source = fileopen.read()
    corpus.append(source)


# In[4]:


#removing the .txt extention and storing only the doc ids
#sfiles=[x[:-4] for x in files]


# In[11]:


#to store the token list of each doc 
processed_corpus=[]
#for each doc
for x,i in enumerate(corpus):
    #appending the token list returned by the preprocess function
    processed_corpus.append(preprocess(i))


# ### Q2. Building IR Systems

# In[20]:


#function to calculate term freq,document freq,doc length,corpus size 
def info(processed_corpus):
    #list to store term freq of each token in each doc in the corpus
    tf = []
    #dict to store the doc freq of each token in the corpus and also the list of docids that contain that token
    df = {}
    #list to store the len of each doc
    doc_len = []
    #to store the tot no of docs
    corpus_size = 0
    
    #for each doc in the processed corpus
    for document in processed_corpus:
        #increasing the corpus size by 1
        corpus_size += 1
        #storing the len of the doc 
        doc_len.append(len(document))

        #computing tf (term frequency) per document
        frequencies = {}
        for term in document:
            term_count = frequencies.get(term, 0) + 1
            frequencies[term] = term_count
        #appending the frequencies dict of that doc into tf list
        tf.append(frequencies)

        #computing df (document frequency) per token
        for term, _ in frequencies.items():
            if term in df.keys():
                df[term]['count']+=1
                df[term]['l'].append(corpus_size-1)
            else:
                df[term]={'count':1,'l':[corpus_size-1]}
    
    return tf,df,doc_len,corpus_size


# In[21]:


#calling info function on processed corpus
tf,df,doc_len,corpus_size=info(processed_corpus)


# In[ ]:





# In[22]:


#function to calculate inverse doc freq of each token
def cal_idf(df,corpus_size):
    #dict to store idf 
    idf={}
    #calculating idf of each token and storing it in dict with key as token and value as idf of that token
    for token,value in df.items():
        idf[token]=math.log(corpus_size/(value['count']+1))
    return idf


# In[23]:


#function to calculate tf idf value of each token in each doc
def cal_tf_idf(tf,idf):
    #list to store tf idf values of each token in each doc
    tf_idf=[]
    
    #for term freq dict of each doc
    for i in tf:
        #to store the tf idf values of the tokens in curr doc
        curr_dict={}
        #getting the total no of words in the curr doc
        no_of_words=sum(i.values())
        #for each token in the term freq dict of curr doc
        for token in i.keys():
            #calculating tf idf of that token in the curr doc 
            curr_dict[token]=(i[token]/no_of_words)*idf[token]
        #appending the tf idf dict of curr doc into tf idf list
        tf_idf.append(curr_dict)
    return tf_idf


# In[24]:


#calculating idf and tf idf
idf=cal_idf(df,corpus_size)
tf_idf=cal_tf_idf(tf,idf)


# In[ ]:





# In[25]:


#function to calculate the idf of bm25
def cal_idf_bm25(df,corpus_size):
    #dict to store idf of bm25
    idf_bm25={}
    #calculating bm25 idf of each token and storing it in dict with key as token and value as bm25 idf of that token
    for token, value in df.items():
            idf_bm25[token] = math.log(1 + (corpus_size - value['count'] + 0.5) / (value['count'] + 0.5))
    return idf_bm25


# In[26]:


#getting bm25 idf
idf_bm25=cal_idf_bm25(df,corpus_size)


# In[ ]:





# In[29]:


#storing all the calculated variables as pickle files

tf_file= open('stored_tf', "wb")
pickle.dump(tf, tf_file)
tf_file.close()

df_file= open('stored_df', "wb")
pickle.dump(df, df_file)
df_file.close()

idf_file= open('stored_idf', "wb")
pickle.dump(idf, idf_file)
idf_file.close()

tf_idf_file= open('stored_tf_idf', "wb")
pickle.dump(tf_idf, tf_idf_file)
tf_idf_file.close()

idf_bm25_file= open('stored_idf_bm25', "wb")
pickle.dump(idf_bm25, idf_bm25_file)
idf_bm25_file.close()

doc_len_file=open('stored_doc_len',"wb")
pickle.dump(doc_len,doc_len_file)
doc_len_file.close()

file_names_file=open('file_names',"wb")
pickle.dump(files,file_names_file)
file_names_file.close()


# In[ ]:




