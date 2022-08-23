**USING PYTHON 3.8.10 IS RECOMMENDED**
**INSTALL JUPYTER NOTEBOOK**
**KINDLY RE-RUN THE ENTIRE NOTEBOOK IF YOU FACE DIFFICULTIES, ALL AT ONCE UNLIKE CELL BY CELL EXECUTION, BECAUSE RUNNING A CELL TWICE AT
SOME POINT OF TIME MAY CAUSE INCONSISTENCIES**
**INSTALL JUPYTER NBCONVERT APP BY EXECUTING "sudo apt install jupyter-nbconvert"**
**BEFORE EXECUTING ANY SHELL SCRIPT, FIRST MAKE IT EXECUTABLE BY COMMAND "chmod +x <file_name>.sh"**
**FOR MORE SPECIFIC INFORMATION OF A QUESTION,PLEASE REFER THE COMMENTS IN 'building_system.ipynb' or 'running_system.ipynb' FILE**

------------------------   REQUIREMENTS   ---------------------
These are the addition modules that are required to be installed before executing scripts

    1. nltk
    2. unidecode
    3. math
    4. re
    5. os
    6. pickle
    7. pandas
    8. io

If you don't have stopwords in the nltk package downloaded, please run the command "nltk.download('stopwords')" in jupyter notebook
to download.

------------------------   ASSUMPTIONS   -------------------------------------

  *  Make sure you have "english-corpora" in the current working directory

---------------------- HOW TO RUN THE ASSIGNMENT -------------------------------

if you want to run from scratch:
     
     1. buildsys.sh is used to build the systems from scratch.
     2. make run testfile = "queries.txt" command is used to get the relevant doc ids.
     
if you only want to get the relavant docids then run only the second command



----------------------------------------------------------------------------------

----------------------- Implementation Details  --------------------------------

I Have Divided the Assignment into two parts. i.e 2 ipynb files
       1. processing document corpus and building IR Systems
       2. Running the IR Systems on the queries text file

---> first part is used to do the processing of the document corpus and
     calculating all the entites that are required for the running of 
     Three IR Systems i.e TF,IDF,BM25 IDF etc . next i stored all the 
     calculated entites in pickle files

--->second part loads all the pickle files and Reads the Queries file as input
    and does preprocessing of the each query and runs each  query on the ir systems
    and stores the output in QRels format in three different csv files , one for each 
    of the ir system i.e boolean_qrels.csv,tf_idf_qrels.csv,bm25_qrels.csv
    THERE ARE NO COLUMN HEADERS IN THE CSV FILES
    
------------------------------------------------------------------

(Q1)PREPROCESSING:
   
    performed
   
   * whitespace tokenizing
   * porter stemming
   * stopword removal
   * non ascii character substitution with ascii characters
   * html tags removal
   * removed punctuations
   * Replace all occurances of "'s" with a single space.

(Q2) preprocessed the query in the same way as the document corpus to get the query tokens

BOOLEAN RETRIEVAL MODEL:

     1.calculated the no of query tokens present in each of the document
     2.ranked the documents based on the no of query tokens present , in descending order
     3.returned the top 5 doc ids from the sorted list

     FOR BETTER UNDERSTANDING OF THIS IMPLEMENTATION YOU CAN HAVE A LOOK AT THE COMMENTS IN THE
     PYTHON NOTEBOOK.HERE I HAVE ONLY GIVEN THE CRUX OF THE IMPLEMENTATION

TF-IDF RETRIEVAL MODEL:

     1.normalized Term Frequency is calculated
     2.Inverse Document Frequency is calculated for a token as log(total no of documents/(no of documents the token is present in+ 1))
     3.next tf idf for every document is calculated
     4.normalized term frequency for query tokens is calculated
     5.now tf idf for every token in the query is calculated
     6.for each document cosine similarity is calculated between the tf idf values of document tokens and the tf idf values  of query tokens
     7.sorted the document ids based on the cosine similarity score in descending order
     8.returned the top 5 doc ids from the sorted list

     FOR BETTER UNDERSTANDING OF THIS IMPLEMENTATION YOU CAN HAVE A LOOK AT THE COMMENTS IN THE
     PYTHON NOTEBOOK.HERE I HAVE ONLY GIVEN THE CRUX OF THE IMPLEMENTATION

BM25 RETRIEVAL MODEL:

Ref: https://en.wikipedia.org/wiki/Okapi_BM25

    1.document length is calculated for each document
    2. average document length is calculated
    3. bm25 score is calculated using the formula
     
        if q1,q2,...,qn are the query tokens,then BM25 score of a document D is:
           
            score(D,Q)=sum(token_score(D,q_i)) for i=1 to n
                             

                                  bm25_idf(q_i) * tf(q_i,D)*(k+1)
            token_score(D,q_i)= -------------------------------------
                                  tf(q_i,D)+k*(1-b+b*( |D|/ avgdl))
           
            where 
                 tf(q_i,D) is the term frequency of token q_i in document D
                 |D| is the document length
                 avgdl is the average document length in the corpus
                 k,b are hyperparameters
                 and
                                     N-n(q_i)+0.5
                  bm25_idf(q_i)=log(--------------- +1)
                                       n(q_i)+0.5
                 where
                      N is the total no of documents
                      n(q_i) is the total no of  documents containing token q_i

    4.sorted the documents based on bm25 score in descending order
    5.returned the top 5 doc ids from the sorted list


 (Q3) queries.txt contains 20 queries,1 query in each line
      the ground truths for these 20 queries are present in ground_truths.csv , following the QREL's format
      for each query i have given 15 relavant document ids
 
 (Q4) The outputs of three ir systems will be stored in 3 different csv files namelt

       boolean_qrels.csv , tf_idf_qrels.csv , bm25_qrels.csv