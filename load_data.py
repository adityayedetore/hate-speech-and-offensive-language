# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import ipdb
#from nltk.tokenize import word_tokenize
import re

def load_dataset(test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    
    # TODO: examine tokenization outputs
    train_file = "train.csv"
    val_file = "val.csv"
    test_file = "test.csv"

    def twitter_tokenize(x):
        # stolen from https://towardsdatascience.com/use-torchtext-to-load-nlp-datasets-part-i-5da6f1c89d84
        x = re.sub(
                        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", 
                            str(x))
        x = re.sub(r"[ ]+", " ", x)
        x = re.sub(r"\!+", " !", x)
        x = re.sub(r"\,+", " ,", x)
        x = re.sub(r"\?+", " ?", x)
        x = re.sub(r"\.", " .", x)
        return x.split()

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=twitter_tokenize, lower=True, include_lengths=True, batch_first=True) #fix_length=50)
    LABEL = data.LabelField()
    #LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    #train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data, test_data = data.TabularDataset.splits(
        path='data/', format='csv',
        train=train_file, validation=val_file, test=test_file,
        fields={'tweet': ('text', TEXT),
                'class': ('label', LABEL)},
        skip_header=False)
    #train_data, test_data = ds.split(split_ratio=0.9)
    #ipdb.set_trace()
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))
    #ipdb.set_trace()
    #print ("Example: ", train_data[234].TEXT)

    #train_data, valid_data = train_data.split(split_ratio=0.8888888889) # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    #ipdb.set_trace()
    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
