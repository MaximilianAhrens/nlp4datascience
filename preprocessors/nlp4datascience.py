"""
@author: Maximilian Ahrens, Oxford 2020 | maximilian.ahrens@outlook.com
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
tqdm.pandas(desc="Progress:")
from string import digits
import itertools
import collections
import scipy.sparse
import nltk
import re
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    print("NLTK wordnet data package not yet installed. Download initiated.")
    nltk.download('wordnet')
    import nltk   
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.phrases import Phrases, Phraser

from nlp4datascience.datahandling.largepickle import pickle_dump

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# whether future warnings should be displayed
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# =============================================================================
# Bag of Words
# =============================================================================
class BagOfWords():
    '''
    DOCSTRING: Information about the function
    INPUT: BagOfWords is a function to pre-process text documents in a bag-of-words style.
    It is largely based on the functions from NLTK and extends and makes the creation process
    of (tf, df, tf-idf weighted) unigrams and bigrams easy and fast.
    '''
    
    def __init__(self, raw_data, min_length = 1 , ngram_length = 1, ngram_connector = "."):
        self.version = "last update: 31 May 2020"
        self.min_length = min_length
        self.ngram_length = ngram_length
        self.ngram_connector = ngram_connector
        self.N = len(list(raw_data)) 
        self.raw_data = raw_data
        self.custom_stopwords = list()
        if ngram_connector == "." or ngram_connector == "," or ngram_connector == "_":
          pass
        else:
          raise ValueError("invalid input for ngram_connector. ""."" or "","" allowed. ")
          
    def sentence_split(self):
        '''
        future updates:
        - expression splits after puctuation and next word lowercase. This might be improved in future versions.
        - convert for loop into list comprehension for faster processing
        '''
        temp = [re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', x) for x in tqdm(self.raw_data)] 
        temp_corpus = pd.DataFrame([item for sublist in temp for item in enumerate(sublist)], columns=["sentence_id","sentence"])
        doc_id = [0]*len(temp_corpus)
        i = -1
        for idx in tqdm(range(len(temp_corpus))):
            if temp_corpus.sentence_id[idx] != 0:
                doc_id[idx] = i
            else:
                i +=1
                doc_id[idx] = i
        temp_corpus["doc_id"] = doc_id
        self.corpus_ssplit = temp_corpus[["doc_id","sentence_id","sentence"]]
    
    def clean(self, remove_numbers=True):                        
        # removing stopwords (NLTK stopword list)
        self.stop_words = stopwords.words('english')+stopwords.words('french')+stopwords.words('german')+self.custom_stopwords
        print("\n1/3: Stopword removal")
        # check whether sentence-splitting occured
        if hasattr(self,"corpus_ssplit"):
            corpus = self.corpus_ssplit["sentence"].progress_apply(lambda x: " ".join(x for x in x.split() if x not in self.stop_words))
        else:
            corpus = self.raw_data.progress_apply(lambda x: " ".join(x for x in x.split() if x not in self.stop_words))
        # lowercasing
        print("\n2/3: Lowercasing")
        corpus = corpus.progress_apply(lambda x: " ".join(x.lower() for x in x.split())) 
        # removing punctuation
        corpus = corpus.str.replace('[^\w\s]','')
        # removing special characters (can be extended)    
        corpus = corpus.str.replace('â','')
        corpus = corpus.str.replace('ô','')
        corpus = corpus.str.replace('_',' ')
        # removing numbers
        if remove_numbers == True:
            remove_digits = str.maketrans('', '', digits)
            print("\n3/3: Removing numbers")
            self.corpus = corpus.progress_apply(lambda x: x.translate(remove_digits))
        else:
            self.corpus = corpus   
    
    def tokenize(self):
        # tokenizing
        tokens = [[word for word in doc.split()] for doc in tqdm(self.corpus)]
        # removing min_length words
        self.tokens = [[word for word in doc if len(word) > self.min_length] for doc in tqdm(tokens)]
        
    
    def stemm(self):
        print("\nCreating unigrams:")
        # stemming (Porter Stemmer)
        self.stems = [[PorterStemmer().stem(word) for word in doc] for doc in tqdm(self.tokens)]
        # removing stopwords from stems
        self.stems = [[word for word in doc if word not in self.stop_words] for doc in tqdm(self.stems)]
        self.unigrams = self.stems
        self.unigrams_unadjust = self.stems.copy()
        # count frequency of unigrams
        self.unigrams_all = [item for sublist in self.unigrams for item in sublist]
        # check for bigrams        
        if self.ngram_length > 1:
            print("\nCreating bigrams:")
            bigrams = []
            for d in tqdm(self.unigrams):
                try:
                    bigrams.append(list(ngrams(d, self.ngram_length)))
                except RuntimeError:
                    bigrams.append(list(""))
            del d     
            # count frequency of bigrams
            self.bigrams_all = [item for sublist in bigrams for item in sublist]
            self.bigrams = bigrams
            self.bigrams_unadjust = bigrams.copy()       
    
    def lemmatize(self):
        print("\nCreating unigrams:")
        # lemmatizing (WordNet Lemmatizer)
        self.lemmas = [[WordNetLemmatizer().lemmatize(word) for word in doc] for doc in tqdm(self.tokens)]
        # removing stopwords from stems
        self.lemmas = [[word for word in doc if word not in self.stop_words] for doc in tqdm(self.lemmas)]
        self.unigrams = self.lemmas
        self.unigrams_unadjust = self.lemmas.copy()        
        # count frequency of unigrams
        self.unigrams_all = [item for sublist in self.unigrams for item in sublist]
        # check for bigrams  
        if self.ngram_length > 1:
            print("\nCreating bigrams:")
            bigrams = []
            for d in self.unigrams:
                try:
                    bigrams.append(list(ngrams(d, 2)))
                except RuntimeError:
                    bigrams.append(list(""))
            del d     
            # count frequency of bigrams
            self.bigrams_all = [item for sublist in bigrams for item in sublist]
            self.bigrams = bigrams
            self.bigrams_unadjust = bigrams.copy()
    
    def ngrams(self, ngram_type):
        if ngram_type == "unigrams":
            ngram_list = self.unigrams
        elif ngram_type == "bigrams":
            ngram_list = self.bigrams

        def tf_idf_compute(t):
            return (1 + np.log(counts[t]))*np.log(self.N/counts_d[t])
        
        v = ngram_list
        agg = itertools.chain(*v)
        counts = collections.Counter(agg)
        v_unique = list(map(lambda x: set(x), v))
        agg_d = itertools.chain(*v_unique)
        counts_d = collections.Counter(agg_d)
        unique_tokens = set(itertools.chain(*v))
        
        unsorted_tf = [counts[t] for t in unique_tokens]
        unsorted_tf_idf = [tf_idf_compute(t) for t in unique_tokens]
        unsorted_df = [counts_d[t] for t in unique_tokens]
        
        if ngram_type == "unigrams":
            self.unigrams_tf = sorted(zip(unique_tokens, unsorted_tf),
                                     key=lambda x: x[1], reverse=True)
            self.unigrams_tf_idf = sorted(zip(unique_tokens, unsorted_tf_idf),
                                     key=lambda x: x[1], reverse=True)
            self.unigrams_df = sorted(zip(unique_tokens, unsorted_df),
                                         key=lambda x: x[1], reverse=True)
            self.unigrams_tf_adj = self.unigrams_tf.copy()
            self.unigrams_df_adj = self.unigrams_df.copy()
            self.unigrams_tf_idf_adj = self.unigrams_tf_idf.copy()
            
        if ngram_type == "bigrams":
            self.bigrams_tf = sorted(zip(unique_tokens, unsorted_tf),
                                     key=lambda x: x[1], reverse=True)
            self.bigrams_tf_idf = sorted(zip(unique_tokens, unsorted_tf_idf),
                                     key=lambda x: x[1], reverse=True)
            self.bigrams_df = sorted(zip(unique_tokens, unsorted_df),
                                     key=lambda x: x[1], reverse=True)
            
            self.bigrams_tf_adj = self.bigrams_tf.copy()
            self.bigrams_df_adj = self.bigrams_df.copy()
            self.bigrams_tf_idf_adj = self.bigrams_tf_idf.copy()
          
    
    def remove_tokens(self, weight, items, cutoff_min, cutoff_max=False):
        
        """
        remove tokens or stems (specified in items) based on weight's ('tf',
        'tfidf', or 'df' value being less than cutoff to remove all words with rank R or
        less, specify cutoff = self.xxx_ranking[R][1]
        """

        def remove(tokens):
            return [t for t in tokens if t not in to_remove]
                   
        if items == "uni":
            if weight == "tf":
                metric = self.unigrams_tf_adj
            elif weight == "tf-idf":    
                metric = self.unigrams_tf_idf_adj
            elif weight == "df":    
                metric = self.unigrams_df_adj  
            
            to_remove = set()
            if cutoff_min != False:
                to_remove.update(set([t[0] for t in metric if t[1] <= cutoff_min]))
            if cutoff_max != False:
                to_remove.update(set([t[0] for t in metric if t[1] >= cutoff_max]))
  
            self.unigrams = list(map(remove, self.unigrams))
            self.unigrams_tf_adj = [t for t in self.unigrams_tf_adj if t[0] not in to_remove]
            self.unigrams_df_adj = [t for t in self.unigrams_df_adj if t[0] not in to_remove]
            self.unigrams_tf_idf_adj = [t for t in self.unigrams_tf_idf_adj if t[0] not in to_remove]
            print("Total number of all unique unigrams:", len(set(self.unigrams_all)))
            print("Total number of all unigrams:", len(self.unigrams_all))
              
            self.unigrams_all_adj = [item for sublist in self.unigrams for item in sublist]
            print("Total number of all unique unigrams after document-occurance cut-off:", len(set(self.unigrams_all_adj)))
            print("Total number of all unigrams after document-occurance cut-off:", len(self.unigrams_all_adj))

        elif items == "bi":
            if weight == "tf":
                metric = self.bigrams_tf_adj
            elif weight == "tf-idf":    
                metric = self.bigrams_tf_idf_adj
            elif weight == "df":    
                metric = self.bigrams_df_adj  
                
            to_remove = set()
            if cutoff_min != False:
                to_remove.update(set([t[0] for t in metric if t[1] <= cutoff_min]))
            if cutoff_max != False:
                to_remove.update(set([t[0] for t in metric if t[1] >= cutoff_max]))
                 
            self.bigrams = list(map(remove, self.bigrams))
            self.bigrams_tf_adj = [t for t in self.bigrams_tf_adj if t[0] not in to_remove]
            self.bigrams_df_adj = [t for t in self.bigrams_df_adj if t[0] not in to_remove]
            self.bigrams_tf_idf_adj = [t for t in self.bigrams_tf_idf_adj if t[0] not in to_remove]    
            print("Total number of all unique bigrams:", len(set(self.bigrams_all)))
            print("Total number of all bigrams:", len(self.bigrams_all))
            
            self.bigrams_all_adj = [item for sublist in self.bigrams for item in sublist]
            print("Total number of all unique bigrams after document-occurance cut-off:", len(set(self.bigrams_all_adj)))
            print("Total number of all bigrams after document-occurance cut-off:", len(self.bigrams_all_adj))

                
    def visualize(self, weight):
        if weight == "tf":
            unigram_type = self.unigrams_tf
            if hasattr(self,"bigrams_tf"):
                bigram_type = self.bigrams_tf
            ftitle = "term frequency ranking"
        elif weight == "df":
            unigram_type = self.unigrams_df
            if hasattr(self,"bigrams_df"):
                bigram_type = self.bigrams_df
            ftitle = "document frequency ranking"
        elif weight == "tf-idf":
            unigram_type = self.unigrams_tf_idf
            if hasattr(self,"bigrams_tf_idf"):
                bigram_type = self.bigrams_tf_idf
            ftitle = "tf-idf ranking"
            
        if self.ngram_length == 1:
            f = plt.figure(figsize = (14,8))
            f.suptitle(str(ftitle+" unigrams"))
            sp = f.add_subplot(111)
            sp.plot([x[1] for x in unigram_type])
            plt.show()
            
          
        elif self.ngram_length > 1:
            f = plt.figure(figsize = (14,8))
            sp = f.add_subplot(211)
            sp.plot([x[1] for x in unigram_type])
            plt.title(str(ftitle+" unigrams"))
            if hasattr(self,"bigrams_tf"):
                sp2 = f.add_subplot(212)
                sp2.plot([x[1] for x in bigram_type])
                plt.title(str(ftitle+" bigrams"))
                plt.show()
                

    def visualize_adj(self, weight):
        if weight == "tf":
            unigram_type = self.unigrams_tf_adj
            if hasattr(self,"bigrams_tf_adj"):
                bigram_type = self.bigrams_tf_adj
            ftitle = "term frequency ranking"
        elif weight == "df":
            unigram_type = self.unigrams_df_adj
            if hasattr(self,"bigrams_df_adj"):
                bigram_type = self.bigrams_df_adj
            ftitle = "document frequency ranking"
        elif weight == "tf-idf":
            unigram_type = self.unigrams_tf_idf_adj
            if hasattr(self,"bigrams_tf_idf_adj"):
                bigram_type = self.bigrams_tf_idf_adj
            ftitle = "tf-idf ranking"                
            
        if self.ngram_length == 1:
            f = plt.figure()
            f.suptitle(str(ftitle+" unigrams"))
            sp = f.add_subplot(111)
            sp.plot([x[1] for x in unigram_type], color ="r")
            plt.show()
          
        elif self.ngram_length > 1:
            f = plt.figure()
            sp = f.add_subplot(211)

            sp.plot([x[1] for x in unigram_type],color ="r")
            plt.title(str(ftitle+" unigrams after cut-off"))
            if hasattr(self,"bigrams_tf"):
                sp2 = f.add_subplot(212)
                sp2.plot([x[1] for x in bigram_type],color ="r")
                plt.title(str(ftitle+" bigrams after cut-off"))
            plt.show()


    def save(self, output_dir, file_name, data_format ="pkl"): # save preprocessed dataset
        results = pd.DataFrame()
        if hasattr(self, "corpus_ssplit"):
            results["doc_id"] = self.corpus_ssplit["doc_id"]
            results["sentence_id"] = self.corpus_ssplit["sentence_id"]
        results["uni"] = self.unigrams_unadjust
        results["uni_adj"] = self.unigrams
        if hasattr(self,"bigrams_tf"):
            len(self.bigrams)
            results["bi"] = self.bigrams_unadjust
            results["bi_adj"] = self.bigrams

        if data_format == "pkl":
            pickle_dump(results, output_dir + str(file_name + "." + data_format))
            print("File saved in pickle-format.")
        if data_format == "csv":
            results.to_csv(output_dir + str(file_name + "." + data_format), encoding="utf-8")
            print("File saved in csv-format.")
        if data_format != "pkl" and data_format != "csv":
            print("ERROR: specified data-format not supported.")  
                
# =============================================================================
# Bag of Words: DTM
# =============================================================================
class DTM():
    '''
    DOCSTRING: Information about the function
    INPUT:DTM is a function to create document-term-matrices of preprocessed
    text documents. It uses the sklearn-library. Unigrams or bigrams can 
    be used as inputs.
    
    raw_data has to be in form of "preprocessed_data.unigrams or 
    preprecessed_data.bigrams". It may need to be "."-connected 
    if used as input in R.
    
    binDTM is either True or False. True creates an additional binary DTM.
    '''
 
    def __init__(self, raw_data, tfidfDTM=False, binDTM=False):
        
      # core attributes
      #self.ngram_length = ngram_length
      self.raw_data = raw_data
      
    def create_dtm(self, ngrams):
      '''
      Create a document-term-matrix
      '''       
      # define preprocessor and tokenizer for sklearn DTM vectorizer  
      def no_preprocessing(doc):
          pass
          return(doc)
      
      def no_tokenizing(corpus):
          pass
          return(corpus)
      
      # define DTM vectorizer
      count_vectorizer = CountVectorizer(lowercase = False,
                               preprocessor=no_preprocessing,
                               tokenizer=no_tokenizing,
                               stop_words= None,
                               analyzer = "word",
                               ngram_range=(1,1)
                               )
      self.countvectorizer = count_vectorizer
        
      # create the DTM
      start_time = time.time()
      dtm = count_vectorizer.fit_transform(ngrams)
      print("--- %s seconds ---" % round((time.time() - start_time),4))
      print("Dimension of document-term-maxtrix (counts):", dtm.shape)
      self.dtm = dtm
      
      # define the tokens
      tokens = count_vectorizer.get_feature_names()
      self.tokens = tokens
              
    def unigram_dtm(self):
        self.create_dtm(self.raw_data)
                
    def bigram_dtm_naive(self): # create each combination of 2 unigrams. Requires creation of bigrams with BagOfWords class above.
        self.create_dtm(self.raw_data)    
        
    def bigram_dtm(self, min_count=5, threshold=10): # feed in unigrams to detect bigrams via collocation method
        bigram_model = Phrases(self.raw_data, min_count=min_count, threshold=threshold, delimiter=b' ')
        bigram_model_final = Phraser(bigram_model)
        bigrams_list = [0]*self.raw_data.shape[0]
        for i, doc in enumerate(tqdm(self.raw_data)):
            bigrams_list[i] = bigram_model_final[doc]
        self.bigrams = bigrams_list
        self.create_dtm(self.bigrams)

    def ngram_dtm(self, min_count=5, threshold=10):
        if hasattr(self, 'bigrams'):
            ngram_model = Phrases(self.bigrams, min_count=min_count, threshold=threshold, delimiter=b' ')
            ngram_model_final = Phraser(ngram_model)
            ngrams_list = [0]*self.raw_data.shape[0]
            for i, doc in enumerate(tqdm(self.bigrams)):
                ngrams_list[i] = ngram_model_final[doc]
            self.ngrams = ngrams_list
        else:
            self.bigram_dtm(min_count=min_count, threshold=threshold)
            ngram_model = Phrases(self.bigrams, min_count=min_count, threshold=threshold, delimiter=b' ')
            ngram_model_final = Phraser(ngram_model)
            ngrams_list = [0]*self.raw_data.shape[0]
            for i, doc in enumerate(tqdm(self.bigrams)):
                ngrams_list[i] = ngram_model_final[doc]
            self.ngrams = ngrams_list
        self.create_dtm(self.ngrams)
        
        
    def save(self, output_dir, file_name): # save preprocessed dataset
        '''
        input variables: output_dir, file_name
        '''
        scipy.sparse.save_npz(output_dir + "/" + file_name +"_dtm.npz", self.dtm)
        pd.Series(self.tokens).to_pickle(output_dir + "/" + file_name + "_tokens.pkl")
        

# =============================================================================
# Word Vectors
# =============================================================================

class WordVectors():
    '''
    Currently supported are word2vec, GloVe, and BERT.
    '''
        
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data
        
          
    def word2vec(self, vector_dim = 200, min_count = 1):
        '''
        size (int, optional) – Dimensionality of the word vectors.
        min_count (int, optional) – Ignores all words with total frequency lower than this.
        '''
        from gensim.models import Word2Vec
        self.word2vec = Word2Vec(self.tokenized_data,
                                 size = vector_dim,
                                 min_count=min_count)
        self.w2v_vectors = self.word2vec.wv
        
    def savew2v(self, output_dir, file_name):
        self.word2vec.save(str(output_dir + "/"+ str(file_name+".model")))

    # once no more (re-)training needed, model state can be discarded to speed up processing
    def deletew2vmodel(self):
        del self.word2vec   
    




