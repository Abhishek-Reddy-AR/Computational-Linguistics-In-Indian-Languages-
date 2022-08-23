**USING PYTHON 3.8.10 IS RECOMMENDED**
**INSTALL JUPYTER NOTEBOOK**
**KINDLY RE-RUN THE ENTIRE NOTEBOOK IF YOU FACE DIFFICULTIES, ALL AT ONCE UNLIKE CELL BY CELL EXECUTION, BECAUSE RUNNING A CELL TWICE AT
SOME POINT OF TIME MAY CAUSE INCONSISTENCIES**
**INSTALL JUPYTER NBCONVERT APP BY EXECUTING "sudo apt install jupyter-nbconvert"**
**BEFORE EXECUTING ANY SHELL SCRIPT, FIRST MAKE IT EXECUTABLE BY COMMAND "chmod +x <file_name>.sh"**
**FOR MORE SPECIFIC INFORMATION OF A QUESTION,PLEASE REFER THE COMMENTS IN 'Q1.ipynb' or 'Q2.ipynb' or 'Q3.ipynb' FILES**

------------------------   REQUIREMENTS   ---------------------
These are the addition modules that are required to be installed before executing scripts

    1. numpy
    2. pandas
    3. math
    4. gensim
    5. warnings
    6. pickle
    7. pytorch_lightning
    8. transformers
    9. datasets
   10. typing
   11. random
   12. torch
   13. seqeval
   14. dataclasses
   15. os
   16. nltk
   17. re
   18. matplotlib
   19. io


------------------------   ASSUMPTIONS   -------------------------------------

  *  Make sure you have the below files in the current working directory as that of the scripts
       
       for Q1:IM ASSUMING ALL THESE FILES FOR Q1 ARE AVAILABLE WITH YOU,SO IM NOT INCLUDING THEM IN THE SUBMISSION FILE

         1.hindi.txt
         2.hi-d100-m2-cbow.model
         3.hi-d100-m2-cbow.model.trainables.syn1neg.npy
         4.hi-d100-m2-cbow.model.wv.vectors.npy
         5.hi-d50-m2-cbow.model.wv.vectors.npy
         6.hi-d100-m2-sg.model.wv.vectors.npy
         7.hi-d50-m2-sg.model.wv.vectors.npy
         8.hi-d100-m2-fasttext.model.wv.vectors.npy
         9.hi-d50-m2-fasttext.model.wv.vectors.npy
        10.hi-d100-glove.txt
        11.hi-d50-glove.txt
       
      for Q2
        
        IF YOU WANT TO BUILD THE MODEL FROM SCRATCH :IM ASSUMING ALL THESE FILES ARE AVAILABLE WITH YOU,SO IM NOT INCLUDING THEM IN THE SUBMISSION FILE
        
            1.hi_train.conll
            2.hi_dev.conll
          
        IF YOU WANT TO RUN THE TUNED MODEL ONLY : I WILL INCLUDE THE BELOW FILES IN THE SUBMISSION FILE

            1.processed_dataset
            2.tuned.ckpt
     
     for Q3:IM ASSUMING ALL THESE FILES ARE AVAILABLE WITH YOU,SO IM NOT INCLUDING THEM IN THE SUBMISSION FILE
    
        1.hi.txt

---------------------- HOW TO RUN THE ASSIGNMENT -------------------------------

Use command "make run" to run all the 3 questions. However, you can also run each question
by executing individual shell files. ex: q1.sh,q2.sh,q3.sh.



----------------------------------------------------------------------------------

----------------------- Implementation Details  --------------------------------

QUESTION 1: in the final columns of output csv files i.e label
            i have assigned label as 1 , when either both ground truth and similarity score are greater than threshold or less than threshold
            i have assigned label as 0 , in the other cases
     i have followed output file format as Q1_ModelName_similarity_value_dimension.csv
           
  ***if you need anything else regarding question 1 , you can refer to q1.ipynb notebook ,it is well documented***

QUESTION 2: it returns a text file names f_score.txt which contains train, test and validation f scores

   *first i have converted the input files into 'sentance' and 'ner tags' format and divided them into test,train and validation splits,this is done by 
    load_data function
   
   *next i have converted these sentance,ner tags into the input feature format of indic-bert model using indic-bert tokenizer,this is done by 
    convert_examples_to_features function
  
   *next i have loaded indic-bert model with softmax of 13 outputs,since 13 different ner tags are present in our input,and defined the forward function
    to the loaded model
   
   *next i defined train_dataloader,test_dataloader and val_dataloader functions to divide the train,test and validation data into batches

   *next i defined training_step,validation_step and test_step functions to perform train ,validation and test loop on each batch
   
   *next i defined validation_epoch_end,test_epoch_end to take the outputs of all the batches from validation and test step and return f score

  
***if you need anything else regarding question 2 , you can refer to q2.ipynb notebook ,it is well documented***

QUESTION 3: for 3.a,3.b,3.c ALL THE OUTPUTS ARE STORED IN TEXT FILES EX:character_unigrams.txt,word_bigrams.txt,syllable_trigrams.txt etc
            for 3.d the outputs are images of distributions named characters_zipf.png,syllables_zipf.png,words_zipf.png

  * i have read only some part of the corpus due to my hardware constraints
  * initially i have removed all the punctuations from the corpus
  * i performed white space tokenization
  * removed words that are either ascii,digits or decimal
  * from each word removed all the characters that are not in hindi

3.a : corrected Unicode mistake of not encoding the halanta character and represented it by 'h'
      
***if you need anything else regarding question 3 , you can refer to q3.ipynb notebook ,it is well documented***


PLEASE DO CHECK THE COMMENTS IN THE NOTEBOOKS,IM NOT INCLUDING THE DETAILS THAT ARE PRESENT IN THE COMMENTS IN THE NOTEBOOKS HERE,AS IT CAUSES REDUNDANCY
