#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:58:55 2019

@author: Maximilian
"""

# =============================================================================
# Literature and References
# =============================================================================
"""
complete beginner's guide to DL: https://towardsdatascience.com/intro-to-deep-learning-c025efd92535

done - https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65

https://towardsdatascience.com/everything-you-need-to-know-about-neural-networks-and-backpropagation-machine-learning-made-easy-e5285bc2be3a

https://towardsdatascience.com/how-to-build-a-simple-neural-network-from-scratch-with-python-9f011896d2f3


"""
## pipeline building
https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976

## feature selection with sklearn and pandas
https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b

# =============================================================================
# Set up pipeline (load modules, define classes and function)
# =============================================================================


# =============================================================================
# UPSTREAM: raw data input, PreProDocs preprocessing, BERT, GloVe, Word2Vec
# =============================================================================
# general tips
https://medium.com/@rrfd/cleaning-and-prepping-data-with-python-for-data-science-best-practices-and-helpful-packages-af1edfbe2a3

# speed up of preprocessing for NLP: spaCy and cython
https://medium.com/huggingface/100-times-faster-natural-language-processing-in-python-ee32033bdced

# =============================================================================
# BERT Tokenizer
# =============================================================================
""" 
source: https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784
"""
#pytorch BERT implementation from huggingface group
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(some_text)

# Then once you convert a string to a list of tokens you have to convert it to a list of IDs that match to words in the BERT vocabulary
tokenizer.convert_tokens_to_ids(tokenized_text)

# next bit (look up on source link for explanation)
max_seq_length = 256
class text_dataset(Dataset):
    def __init__(self,x_y_list):
self.x_y_list = x_y_list
        
    def __getitem__(self,index):
        
        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])
        
        if len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]
            
        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length - len(ids_review))
        
        ids_review += padding
        
        assert len(ids_review) == max_seq_length
        
        #print(ids_review)
        ids_review = torch.tensor(ids_review)
        
        sentiment = self.x_y_list[1][index] # color        
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        
        return ids_review, list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])

class text_dataset(Dataset):
    def __init__(self,x_y_list):
        self.x_y_list = x_y_list
        
def __getitem__(self,index):
        
        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])
        
        if len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]
            
        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length - len(ids_review))
        
        ids_review += padding
        
        assert len(ids_review) == max_seq_length
        
        #print(ids_review)
        ids_review = torch.tensor(ids_review)
        
        sentiment = self.x_y_list[1][index]      
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        
        return ids_review, list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])
    
    
#### Training
"""
At this point the training pipeline is pretty standard (now that BERT is just another Pytorch model).     
"""
outputs = F.softmax(outputs,dim=1)

lrlast = .001
lrmain = .00001
optim1 = optim.Adam(
    [
        {"params":model.bert.parameters(),"lr": lrmain},
        {"params":model.classifier.parameters(), "lr": lrlast},       
   ])

optimizer_ft = optim1

    
# =============================================================================
# =============================================================================
# # DOWNSTREAM: ML models for TS prediction
# =============================================================================
# =============================================================================

# =============================================================================
# OLS REGRESSION
# =============================================================================

# =============================================================================
# LASSO
# =============================================================================

# =============================================================================
# ELASTIC NET
# =============================================================================

# =============================================================================
# =============================================================================
# # TOPIC MODELS
# =============================================================================
# =============================================================================

# =============================================================================
# LSA
# =============================================================================
"""
source: https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImRiMDJhYjMwZTBiNzViOGVjZDRmODE2YmI5ZTE5NzhmNjI4NDk4OTQiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE1NzQ2MTAxNTIsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjEwNzU0MTYzMzc2NjQxNjI3NjI1MCIsImVtYWlsIjoibWF4aW1pbGlhbi5haHJlbnNAb3V0bG9vay5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6Ik1heGltaWxpYW4gQWhyZW5zIiwicGljdHVyZSI6Imh0dHBzOi8vbGg0Lmdvb2dsZXVzZXJjb250ZW50LmNvbS8tSWtjMFJBSGZqdmsvQUFBQUFBQUFBQUkvQUFBQUFBQUFBQUEvQUNIaTNyZlZWYVdQd0otcEJSS1VfX29fbVE0YVcyMTVEdy9zOTYtYy9waG90by5qcGciLCJnaXZlbl9uYW1lIjoiTWF4aW1pbGlhbiIsImZhbWlseV9uYW1lIjoiQWhyZW5zIiwiaWF0IjoxNTc0NjEwNDUyLCJleHAiOjE1NzQ2MTQwNTIsImp0aSI6ImQ1OGMxNWEyNGNjYjJjNDIyMWMwMDE5ZGJiMTVmNzg3ZjNmMzQzOGIifQ.dAzfgFmB_KstTHducsc2laev3RqgoRg796fexz7DCNmJSaHSpzZ1TNxgvsxIPq7dGvejaT_oyBK5xKm7NR0DlXPCeC_E5ns17N4NX-kyEtGwM0nyEUoCBl_s5ZZojxUHWaMSW7YU5WJDDgH2YYlKvlqxuYmvIS8eIMWLNqh9YnaDTA6cr2CAQpTbKbINttoWuMzqHpF5C_ffRpiENerDF1P8seOGur-sagfPsq1gLt_gg1y92N1NIAqsfL7cQqogm8BLhWFTSKzmptHY6bvcDBEedDDrbBOrvHLl2uF3p7cgLm6mgAfcmMDx5saeuy3QSKNr8tcuGLKboTXuB3h7Qg

LSA is quick and efficient to use, but it does have a few primary drawbacks:
lack of interpretable embeddings (we don’t know what the topics are, and the components may be arbitrarily positive/negative)
need for really large set of documents and vocabulary to get accurate results
less efficient representation

"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
documents = ["doc1.txt", "doc2.txt", "doc3.txt"] 
  
# raw documents to tf-idf matrix: 
vectorizer = TfidfVectorizer(stop_words='english', 
                             use_idf=True, 
                             smooth_idf=True)
# SVD to reduce dimensionality: 
svd_model = TruncatedSVD(n_components=100,         // num dimensions
                         algorithm='randomized',
                         n_iter=10)
# pipeline of tf-idf + SVD, fit to and applied to documents:
svd_transformer = Pipeline([('tfidf', vectorizer), 
                            ('svd', svd_model)])
svd_matrix = svd_transformer.fit_transform(documents)

# svd_matrix can later be used to compare documents, compare words, or compare queries with documents

# =============================================================================
# PLSA (probablistic LSA)
# =============================================================================
"""
source: https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImRiMDJhYjMwZTBiNzViOGVjZDRmODE2YmI5ZTE5NzhmNjI4NDk4OTQiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE1NzQ2MTAxNTIsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjEwNzU0MTYzMzc2NjQxNjI3NjI1MCIsImVtYWlsIjoibWF4aW1pbGlhbi5haHJlbnNAb3V0bG9vay5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6Ik1heGltaWxpYW4gQWhyZW5zIiwicGljdHVyZSI6Imh0dHBzOi8vbGg0Lmdvb2dsZXVzZXJjb250ZW50LmNvbS8tSWtjMFJBSGZqdmsvQUFBQUFBQUFBQUkvQUFBQUFBQUFBQUEvQUNIaTNyZlZWYVdQd0otcEJSS1VfX29fbVE0YVcyMTVEdy9zOTYtYy9waG90by5qcGciLCJnaXZlbl9uYW1lIjoiTWF4aW1pbGlhbiIsImZhbWlseV9uYW1lIjoiQWhyZW5zIiwiaWF0IjoxNTc0NjEwNDUyLCJleHAiOjE1NzQ2MTQwNTIsImp0aSI6ImQ1OGMxNWEyNGNjYjJjNDIyMWMwMDE5ZGJiMTVmNzg3ZjNmMzQzOGIifQ.dAzfgFmB_KstTHducsc2laev3RqgoRg796fexz7DCNmJSaHSpzZ1TNxgvsxIPq7dGvejaT_oyBK5xKm7NR0DlXPCeC_E5ns17N4NX-kyEtGwM0nyEUoCBl_s5ZZojxUHWaMSW7YU5WJDDgH2YYlKvlqxuYmvIS8eIMWLNqh9YnaDTA6cr2CAQpTbKbINttoWuMzqHpF5C_ffRpiENerDF1P8seOGur-sagfPsq1gLt_gg1y92N1NIAqsfL7cQqogm8BLhWFTSKzmptHY6bvcDBEedDDrbBOrvHLl2uF3p7cgLm6mgAfcmMDx5saeuy3QSKNr8tcuGLKboTXuB3h7Qg
pLSA, or Probabilistic Latent Semantic Analysis, uses a probabilistic method instead of SVD to tackle the problem. The core idea is to find a probabilistic model with latent topics that can generate the data we observe in our document-term matrix. In particular, we want a model P(D,W) such that for any document d and word w, P(d,w) corresponds to that entry in the document-term matrix.
Recall the basic assumption of topic models: each document consists of a mixture of topics, and each topic consists of a collection of words. pLSA adds a probabilistic spin to these assumptions:

It is a far more flexible model than LSA, but still has a few problems. In particular:
Because we have no parameters to model P(D), we don’t know how to assign probabilities to new documents
The number of parameters for pLSA grows linearly with the number of documents we have, so it is prone to overfitting

We will not look at any code for pLSA because it is rarely used on its own. In general, when people are looking for a topic model beyond the baseline performance LSA gives, they turn to LDA. LDA, the most common type of topic model, extends PLSA to address these issues.
"""

# =============================================================================
# LDA
# =============================================================================
"""
source: https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImRiMDJhYjMwZTBiNzViOGVjZDRmODE2YmI5ZTE5NzhmNjI4NDk4OTQiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE1NzQ2MTAxNTIsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjEwNzU0MTYzMzc2NjQxNjI3NjI1MCIsImVtYWlsIjoibWF4aW1pbGlhbi5haHJlbnNAb3V0bG9vay5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6Ik1heGltaWxpYW4gQWhyZW5zIiwicGljdHVyZSI6Imh0dHBzOi8vbGg0Lmdvb2dsZXVzZXJjb250ZW50LmNvbS8tSWtjMFJBSGZqdmsvQUFBQUFBQUFBQUkvQUFBQUFBQUFBQUEvQUNIaTNyZlZWYVdQd0otcEJSS1VfX29fbVE0YVcyMTVEdy9zOTYtYy9waG90by5qcGciLCJnaXZlbl9uYW1lIjoiTWF4aW1pbGlhbiIsImZhbWlseV9uYW1lIjoiQWhyZW5zIiwiaWF0IjoxNTc0NjEwNDUyLCJleHAiOjE1NzQ2MTQwNTIsImp0aSI6ImQ1OGMxNWEyNGNjYjJjNDIyMWMwMDE5ZGJiMTVmNzg3ZjNmMzQzOGIifQ.dAzfgFmB_KstTHducsc2laev3RqgoRg796fexz7DCNmJSaHSpzZ1TNxgvsxIPq7dGvejaT_oyBK5xKm7NR0DlXPCeC_E5ns17N4NX-kyEtGwM0nyEUoCBl_s5ZZojxUHWaMSW7YU5WJDDgH2YYlKvlqxuYmvIS8eIMWLNqh9YnaDTA6cr2CAQpTbKbINttoWuMzqHpF5C_ffRpiENerDF1P8seOGur-sagfPsq1gLt_gg1y92N1NIAqsfL7cQqogm8BLhWFTSKzmptHY6bvcDBEedDDrbBOrvHLl2uF3p7cgLm6mgAfcmMDx5saeuy3QSKNr8tcuGLKboTXuB3h7Qg
LDA stands for Latent Dirichlet Allocation. LDA is a Bayesian version of pLSA.
In particular, it uses dirichlet priors for the document-topic and word-topic distributions, lending itself to better generalization.
"""

from gensim.corpora.Dictionary import load_from_text, doc2bow
from gensim.corpora import MmCorpus
from gensim.models.ldamodel import LdaModel
document = "This is some document..."
# load id->word mapping (the dictionary)
id2word = load_from_text('wiki_en_wordids.txt')
# load corpus iterator
mm = MmCorpus('wiki_en_tfidf.mm')
# extract 100 LDA topics, updating once every 10,000
lda = LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)
# use LDA model: transform new doc to bag-of-words, then apply lda
doc_bow = doc2bow(document.split())
doc_lda = lda[doc_bow]
# doc_lda is vector of length num_topics representing weighted presence of each topic in the doc

# =============================================================================
# SUPERVIDED LDA
# =============================================================================

# =============================================================================
# Hierachical Bayesian Model
# =============================================================================


# =============================================================================
# BAYESIAN TOPIC REGRESSION (MA, JA)
# =============================================================================

# =============================================================================
# BAYESIAN SLDA
# =============================================================================




# =============================================================================
# LDA2Vec
# =============================================================================
# topic modelling overview
"""
source: https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05

lda2vec is an extension of word2vec and LDA that jointly learns word, document, and topic vectors.

lda2vec specifically builds on top of the skip-gram model of word2vec to generate word vectors. If you’re not familiar with skip-gram and word2vec, you can read up on it here, but essentially it’s a neural net that learns a word embedding by trying to use the input word to predict surrounding context words.

With lda2vec, instead of using the word vector directly to predict context words, we leverage a context vector to make the predictions. This context vector is created as the sum of two other vectors: the word vector and the document vector.
"""

# code
https://github.com/cemoody/lda2vec

# =============================================================================
# Bayesian LSTM with attention (MA)
# =============================================================================
# bayes by backprop
https://medium.com/neuralspace/probabilistic-deep-learning-bayes-by-backprop-c4a3de0d9743


https://medium.com/neuralspace/when-machine-learning-meets-complexity-why-bayesian-deep-learning-is-unavoidable-55c97aa2a9cc

https://medium.com/@laumannfelix/reflections-on-bayesian-inference-in-probabilistic-deep-learning-416376e42dc0


Further reading:
MacKay, D. J. (1992). Bayesian interpolation. Neural computation, 4(3), 415–447.
Hinton, G. E., & Van Camp, D. (1993). Keeping the neural networks simple by minimizing the description length of the weights. In Proceedings of the sixth annual conference on Computational learning theory (pp. 5–13). ACM.
Graves, A. (2011). Practical variational inference for neural networks. In Advances in Neural Information Processing Systems, pages 2348–2356.
Blundell, C., Cornebise, J., Kavukcuoglu, K., and Wierstra, D. (2015). Weight uncertainty in neural networks. arXiv preprint arXiv:1505.05424.
219

Machine Learning
Probability
Deep Learning
Artificial Intelligence
# =============================================================================
# LSTM (bidirectional)
# =============================================================================

#general re LSTM
https://towardsdatascience.com/reading-between-the-layers-lstm-network-7956ad192e58

# =============================================================================
# LSTM (bi, attentional)
# =============================================================================
## attention
"""
https://towardsdatascience.com/an-introduction-to-attention-transformers-and-bert-part-1-da0e838c7cda
"""


# ULMFiT

# ELMo

# =============================================================================
# Transformers: BERT
# =============================================================================
https://medium.com/synapse-dev/understanding-bert-transformer-attention-isnt-all-you-need-5839ebd396db
"""
[Transformers] in contrast to RNNs — relies purely on attention mechanisms, and does not have an explicit notion of word order beyond marking each word with its absolute-position embedding. This reliance on attention may lead one to expect decreased performance on syntax-sensitive tasks compared to RNN (LSTM) models that do model word order directly, and explicitly track states across the sentence.
"""
https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1

https://towardsdatascience.com/bert-to-the-rescue-17671379687f

# step by step explanation and comparison to LSTMs
https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03

https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

https://towardsdatascience.com/bert-text-classification-in-3-lines-of-code-using-keras-264db7e7a358
# =============================================================================
# Transformers: GPT2
# =============================================================================

# =============================================================================
# XLNet
# =============================================================================
https://towardsdatascience.com/what-is-two-stream-self-attention-in-xlnet-ebfe013a0cf3
https://towardsdatascience.com/what-is-xlnet-and-why-it-outperforms-bert-8d8fce710335



# =============================================================================
# =============================================================================
# # VISUALIZATION 
# =============================================================================
# =============================================================================
