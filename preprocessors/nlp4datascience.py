
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
tqdm.pandas(desc="Progress:")

from string import digits
import itertools
import collections
import nltk

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
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer

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
    
    def __init__(self, raw_data, min_length, ngram_length, ngram_connector = "."):
        self.min_length = min_length
        self.ngram_length = ngram_length
        self.ngram_connector = ngram_connector
        self.N = len(list(raw_data)) 
        
        if ngram_connector == "." or ngram_connector == "," or ngram_connector == "_":
          pass
        else:
          raise ValueError("invalid input for ngram_connector. ""."" or "","" allowed. ")
          
        # removing stopwords (NLTK stopword list)
        self.stop_words = stopwords.words('english')+stopwords.words('french')+stopwords.words('german')
        print("\n\n1/3: Stopword removal\n\n:")
        corpus = raw_data.progress_apply(lambda x: " ".join(x for x in x.split() if x not in self.stop_words))
        # lowercasing
        print("\n\n2/3: Lowercasing\n\n:")
        corpus = corpus.progress_apply(lambda x: " ".join(x.lower() for x in x.split())) 
        # removing punctuation
        corpus = corpus.str.replace('[^\w\s]','')
        # removing special characters (can be extended)    
        corpus = corpus.str.replace('â','')
        corpus = corpus.str.replace('ô','')
        # removing numbers
        remove_digits = str.maketrans('', '', digits)
        print("\n\n3/3: Removing numbers\n\n:")
        self.corpus = corpus.progress_apply(lambda x: x.translate(remove_digits))
    
    
    def tokenize(self):
        # tokenizing
        tokens = [[word for word in doc.split()] for doc in tqdm(self.corpus)]
        # removing min_length words
        self.tokens = [[word for word in doc if len(word) > self.min_length] for doc in tqdm(tokens)]
        
    
    def stemm(self):
        print("\n\nCreating unigrams\n\n:")
        # stemming (Porter Stemmer)
        self.stems = [[PorterStemmer().stem(word) for word in doc] for doc in tqdm(self.tokens)]
        # removing stopwords from stems
        self.stems = [[word for word in doc if word not in self.stop_words] for doc in tqdm(self.stems)]
        self.unigrams = self.stems
        # count frequency of unigrams
        self.unigrams_all = [item for sublist in self.unigrams for item in sublist]
        # check for bigrams        
        if self.ngram_length > 1:
            print("\n\nCreating bigrams\n\n:")
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
        
    
    def lemmatize(self):
        print("\n\nCreating unigrams\n\n:")
        # lemmatizing (WordNet Lemmatizer)
        self.lemmas = [[WordNetLemmatizer().lemmatize(word) for word in doc] for doc in tqdm(self.tokens)]
        # removing stopwords from stems
        self.lemmas = [[word for word in doc if word not in self.stop_words] for doc in tqdm(self.lemmas)]
        self.unigrams = self.lemmas
        # count frequency of unigrams
        self.unigrams_all = [item for sublist in self.unigrams for item in sublist]
        # check for bigrams  
        if self.ngram_length > 1:
            print("\n\nCreating bigrams\n\n:")
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
     
    
    def weighted_ngrams(self, ngram_type):
        if ngram_type == "unigrams":
            ngram_list = self.unigrams
        if ngram_type == "bigrams":
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
        if ngram_type == "bigrams":
            self.bigrams_tf = sorted(zip(unique_tokens, unsorted_tf),
                                     key=lambda x: x[1], reverse=True)
            self.bigrams_tf_idf = sorted(zip(unique_tokens, unsorted_tf_idf),
                                     key=lambda x: x[1], reverse=True)
            self.bigrams_df = sorted(zip(unique_tokens, unsorted_df),
                                     key=lambda x: x[1], reverse=True)
          
    
    def rank_remove(self, rank, items, cutoff):
        
        """
        remove tokens or stems (specified in items) based on rank's ('tf',
        'tfidf', or 'df' value being less than cutoff to remove all words with rank R or
        less, specify cutoff = self.xxx_ranking[R][1]
        """

        def remove(tokens):
            return [t for t in tokens if t not in to_remove]
                   
        if items == "uni":
            if rank == "tf":
                to_remove = set([t[0] for t in self.unigrams_tf if t[1] <= cutoff])
            elif rank == "tf-idf":
                to_remove = set([t[0] for t in self.unigrams_tf_idf if t[1] <= cutoff])
            elif rank == "df":
                to_remove = set([t[0] for t in self.unigrams_df if t[1] <= cutoff])    
            self.unigrams_adj = list(map(remove, self.unigrams))
            self.unigrams_tf_adj = [t for t in self.unigrams_tf if t[0] not in to_remove]
            self.unigrams_df_adj = [t for t in self.unigrams_df if t[0] not in to_remove]
            self.unigrams_tf_idf_adj = [t for t in self.unigrams_tf_idf if t[0] not in to_remove]
            
            print("Total number of all unique unigrams:", len(set(self.unigrams_all)))
            print("Total number of all unigrams:", len(self.unigrams_all))
              
            self.unigrams_all_adj = [item for sublist in self.unigrams_adj for item in sublist]
            print("Total number of all unique unigrams after document-occurance cut-off:", len(set(self.unigrams_all_adj)))
            print("Total number of all unigrams after document-occurance cut-off:", len(self.unigrams_all_adj))

        if items == "bi":
            if rank == "tf":
                to_remove = set([t[0] for t in self.bigrams_tf if t[1] <= cutoff])
            elif rank == "tf-idf":
                to_remove = set([t[0] for t in self.bigrams_tf_idf if t[1] <= cutoff])
            elif rank == "df":
                to_remove = set([t[0] for t in self.bigrams_df if t[1] <= cutoff])    
            self.bigrams_adj = list(map(remove, self.bigrams))
            self.bigrams_tf_adj = [t for t in self.bigrams_tf if t[0] not in to_remove]
            self.bigrams_df_adj = [t for t in self.bigrams_df if t[0] not in to_remove]
            self.bigrams_tf_idf_adj = [t for t in self.bigrams_tf_idf if t[0] not in to_remove]
            
            print("Total number of all unique bigrams:", len(set(self.bigrams_all)))
            print("Total number of all bigrams:", len(self.bigrams_all))
            
            self.bigrams_all_adj = [item for sublist in self.bigrams_adj for item in sublist]
            print("Total number of all unique bigrams after document-occurance cut-off:", len(set(self.bigrams_all_adj)))
            print("Total number of all bigrams after document-occurance cut-off:", len(self.bigrams_all_adj))

                
    def visualize(self, weight):
        if weight == "tf":
            unigram_type = self.unigrams_tf
            try:
                bigram_type = self.bigrams_tf
            except AttributeError:
                pass
            ftitle = "term frequency ranking"
        if weight == "df":
            unigram_type = self.unigrams_df
            try:
                bigram_type = self.bigrams_df
            except AttributeError:
                pass
            ftitle = "document frequency ranking"
        if weight == "tf-idf":
            unigram_type = self.unigrams_tf_idf
            try:
                bigram_type = self.bigrams_tf_idf
            except AttributeError:
                pass
            ftitle = "tf-idf-ranking"
            
        if self.ngram_length == 1:
            f = plt.figure()
            f.suptitle(str(ftitle+" unigrams"))
            sp = f.add_subplot(111)
            sp.plot([x[1] for x in unigram_type])
            return f
          
        if self.ngram_length > 1:
            f = plt.figure()
            sp = f.add_subplot(211)
            sp.plot([x[1] for x in unigram_type])
            plt.title(str(ftitle+" unigrams"))
            sp2 = f.add_subplot(212)
            sp2.plot([x[1] for x in bigram_type])
            plt.title(str(ftitle+" bigrams"))
        return f
                

    def visualize_adj(self, weight):
        if weight == "tf":
            unigram_type = self.unigrams_tf_adj
            bigram_type = self.bigrams_tf_adj
            ftitle = "term frequency ranking"
        if weight == "df":
            unigram_type = self.unigrams_df_adj
            bigram_type = self.bigrams_df_adj
            ftitle = "document frequency ranking"
        if weight == "tf-idf":
            unigram_type = self.unigrams_tf_idf_adj
            bigram_type = self.bigrams_tf_idf_adj
            ftitle = "tf-idf-ranking"
            
        if self.ngram_length == 1:
            f = plt.figure()
            f.suptitle(str(ftitle+" unigrams"))
            sp = f.add_subplot(111)
            sp.plot([x[1] for x in unigram_type])
            return f
          
        if self.ngram_length > 1:
            f = plt.figure()
            sp = f.add_subplot(211)
            sp.plot([x[1] for x in unigram_type])
            plt.title(str(ftitle+" unigrams"))
            sp2 = f.add_subplot(212)
            sp2.plot([x[1] for x in bigram_type])
            plt.title(str(ftitle+" bigrams"))
        return f

    def save(self, data_format, output_dir, file_name): # save preprocessed dataset
        results = pd.DataFrame()
        results["uni"] = self.unigrams
        results["uni_adj"] = self.unigrams_adj
        
        try:
            len(self.bigrams)
            results["bi"] = self.bigrams
            results["bi_adj"] = self.bigrams_adj
        except AttributeError:
            pass
            
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
    preprecessed_data.bigrams". It needs to be "."-connected for R-input.
    
    binDTM is either True or False. True creates an additional binary DTM.
    '''
    
    def __init__(self, raw_data, ngram_length, nlp4datascience2sklearn=True,
                 tfidfDTM=False, binDTM=False):
      # core attributes
      self.ngram_length = ngram_length
      self.nlp4datascience2sklearn = nlp4datascience2sklearn
      
      # check whether data is already preprocessed via nlp4datascience
      if self.nlp4datascience2sklearn == True:
      
        # for unigrams
        if ngram_length == 1:
            unigrams_sentences= []
            for doc in raw_data:
                temp = " ".join(doc)
                unigrams_sentences.append(temp)  
            self.unigrams = unigrams_sentences
        
        # for bigrams
        elif ngram_length == 2:
            bigrams_dot = []
            for doc in raw_data:
                new_doc = []
                for tup in doc:
                    new_doc.append(str(tup[0]) + '.' + str(tup[1]))
                    bigrams_dot.append(new_doc)   
              
            bigrams_dot_sentences= []
            for doc in bigrams_dot:
                temp = " ".join(doc)
                bigrams_dot_sentences.append(temp)
            self.bigrams = bigrams_dot_sentences
            self.bigram_tokens = bigrams_dot
        
        # if more than bigrams
        else:
            raise ValueError("Specified ngram length currently not yet supported.")
      
      # if un-preprocessed text data is used
      else: 
        if ngram_length == 1:
          self.ngrams = raw_data
        if ngram_length == 2:
          self.ngrams = raw_data         
        else:
          raise ValueError("Specified ngram length currently not yet supported.")
    
    
    def create_dtm(self, ngrams):
      '''
      Create a document-term-matrix
      '''
      if self.nlp4datascience2sklearn == True:
        
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
                 

    def sparsify(self, DTM, cut_off = 0.1):
      '''
      Make DTM sparser
      '''
      self.cut_off = cut_off
      old_dict = DTM.sum(axis=0, skipna = True)
      old_dict.sort_values(ascending = False, inplace = True)
      cut_off_threshold = round(len(old_dict)*(1-cut_off))
      new_dict = old_dict[:cut_off_threshold]
      DTM_sparsified = DTM[new_dict.index]
      return(DTM_sparsified)      


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
    

# =============================================================================
#     # TBD# 
#     def bigram2vec(self, vector_dim = 200, min_count = 1):
#         '''
#         size (int, optional) – Dimensionality of the word vectors.
#         min_count (int, optional) – Ignores all words with total frequency lower than this.
#         '''
#         bigram_transformer = Phrases(common_texts)
#         model = Word2Vec(bigram_transformer[common_texts], min_count=1)
#         
#         
#     # tbd    
#     def glovevec(self):
#         pass
#     
#     # tbd
#     def bertvec(self):
#         pass
# =============================================================================
        


  
      


# =============================================================================
# Further documentation
# =============================================================================
### --> see also: https://stanford.edu/~rjweiss/public_html/IRiSS2013/text2/notebooks/cleaningtext.html


        
# =============================================================================
# WORK IN PROGRESS: ADDITIONAL OPTIONS
# =============================================================================

        
    # choose tokenizer
        
# =============================================================================
#     def my_preprocessor(doc):
#         #corpus = unescape(doc).lower()
#         corpus = unescape(doc)
#         return corpus
#     
#     def my_tokenizer(corpus):
#         # create a spaCy tokenizer
#         spacy.load("en_core_web_sm")
#         lemmatizer = spacy.lang.en.English()
#         tokens = lemmatizer(corpus)
#         return([token.lemma_ for token in tokens])
#         
#         
#     count_vectorizer = CountVectorizer(preprocessor=my_preprocessor,
#                                  tokenizer=my_tokenizer,
#                                  ngram_range=(1,1),
#                                  #stop_words='english'
#                                  )
# =============================================================================
        
 # =============================================================================
#     # Word-Matrix to Dataframe
#     def wm2df(wm, feat_names): 
#         # create an index for each row
#         doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
#         df = pd.DataFrame(data=wm.toarray(), index=doc_names,
#                           columns=feat_names)
#         return(df)
# =============================================================================   
    
    
# =============================================================================
#         start_time = time.time()
#         dtm_count = count_vectorizer.fit_transform(raw_data)
#         print("--- %s seconds ---" % round((time.time() - start_time),4))
#         print("Dimension of document-term-maxtrix (counts):", dtm_count.shape)
#         self.count = dtm_count
#         self.shape = dtm_count.shape
#         self.toarray = dtm_count.toarray
#         
#         tokens = count_vectorizer.get_feature_names()
#         #wm2df(dtm_count, tokens)
#         #word2id = dict((v, idx) for idx, v in enumerate(tokens))
#         self.tokens = tokens
#         
#         if tfidfDTM==True:
#             # term frequencies (count/doc-length)
#             tf_transformer = TfidfTransformer(use_idf=False)
#             dtm_tf = tf_transformer.fit_transform(dtm_count)
#             dtm_tf.shape
#             self.tf = dtm_tf
#             
#             # tf-idf score
#             tfidf_transformer = TfidfTransformer(use_idf=True)
#             dtm_tfidf = tfidf_transformer.fit_transform(dtm_count)
#             dtm_tfidf.shape
#             self.tfidf = dtm_tfidf
# 
#         if binDTM==True:
#             # word occurences (binary dtm)
#             bin_vectorizer = CountVectorizer(preprocessor=my_preprocessor,
#                                      tokenizer=my_tokenizer,
#                                      ngram_range=(1,1),
#                                      #stop_words='english',
#                                      binary = True
#                                      )
# 
#             start_time = time.time()
#             dtm_bin = bin_vectorizer.fit_transform(raw_data)
#             print("--- %s seconds ---" % round((time.time() - start_time),4))
#             print("Dimension of document-term-maxtrix (counts):", dtm_bin.shape)
#             self.bin = dtm_bin
# =============================================================================
       


