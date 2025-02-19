'''
For theory of vector embedding, refer to OneNote on the IPad

To convert a word to vector embedding, we must train a neural network which is a very complex task.
We can make use of pretrained models from companies such as Google, Facebook, OpenAI, etc.

Eg. Google has an opensource model -- word2vec-google-news-300
Consists of pre trained vectors trained on a part of the Google News dataset (100 billion words)
Model consists of 300 dimensional vectors for 3 million words and phrases.

'''

'''
import numpy as np

import gensim.downloader as api
model = api.load("word2vec-google-news-300")

word_vector = model

# print(word_vector["computer"])
# print(word_vector["India"])
print(word_vector["Dog"].shape)
print(word_vector.most_similar(positive=["King", "Woman"], negative=["Man"], topn=3))

print(word_vector.similarity('woman', 'man'))
print(word_vector.similarity('paper', 'water'))
print(word_vector.similarity('God', 'Devil'))

# Vector difference
word_1 = 'man'
word_2 = 'woman'

vector_difference = model[word_1] - model[word_2]

# calculate the magnitude of the vector difference
magnitude_of_difference = np.linalg.norm(vector_difference)
print("Magnitude of difference b/w man and woman is : ", magnitude_of_difference)

# If the magnitude of difference is small then the words are semanticaly close to each other.
'''

# Simple Vector Embedding weight matrix using Torch

'''
vocab:      Today is a beautiful day
Token ID:     5   4  1     2      3  (Words are sorted in alphabetical order before creating token ids). For this process of tokenization usually libraries such as 'tiktoken' which uses Bite-Pair encoding.
'''

import torch

inputs_ids = torch.tensor([4, 1, 2, 3]) # creating a tensor of the sentence

# creating vector embedding weight matrix

vocab_size = 6 # total number of words. Eg. ChatGPT-2 had a vocab size of 50267
output_dim = 3 # dimension of the vector embedding for each token. ChatGPT-2 had dimension of 768 for each word to capture the complex semantic meaning of words

torch.manual_seed(232)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# Embedding is a simple lookup table that stores embeddings of a fixed dictionary and size -- definition from pytorch
print(embedding_layer(torch.tensor([3]))) # to retrieve the vector embedding of token id 4 (since indexing starts from 0)

# if we want vector embeddings for our sentence, then we simple give our input ids
print(embedding_layer(inputs_ids)) # dimension will be 4 * 3 

