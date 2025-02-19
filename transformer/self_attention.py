'''
Consider the sentence: "Your Journey starts with one stop"

We will create a context vector for this sentence using self-attention mechanism
'''

import tiktoken
import torch

tokenizer = tiktoken.get_encoding("gpt2")

sentence = "Your Journey starts with one stop"
print("senetence: ", sentence)

tokenized_sentence = tokenizer.encode(sentence)
print("Tokenized version: ", tokenized_sentence)

# creating 3 dimensional vector embedding
vocab_size = 50257
output_dim = 3 

# creating dictionary
torch.manual_seed(231)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# convert the tokenized version of sentence into tensor
tensors = torch.tensor(tokenized_sentence)
print("tokens converted into tensors: ", tensors)

vector_embeddings = embedding_layer(tensors)
print("3 dimensional vector embeddings: \n", vector_embeddings) # shape - 6x3

'''
tensor([[-0.1969, -0.3639,  0.0624],
        [-1.8691,  0.7923,  0.4234],
        [-0.6269,  0.7969,  1.3223],
        [ 0.1139,  1.1128, -0.0928],
        [ 2.1134,  0.7708,  0.9993],
        [ 0.0270,  0.3606, -0.0730]], grad_fn=<EmbeddingBackward0>)
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
# Visualization

# corresponding words
words = ["Your", "Journey", "starts", "with", "one", "stop"]

# extract x, y, z coordinates
x_coords = vector_embeddings[:, 0].detach().numpy()
y_coords = vector_embeddings[:, 1].detach().numpy()
z_coords = vector_embeddings[:, 2].detach().numpy()

# create 3D plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot for each point and annotate with corresponding word

for x, y, z, word in zip(x_coords, y_coords, z_coords, words):
    ax.scatter(x, y, z)
    ax.text(x, y, z, word, fontsize=10)


# set lables
ax.set_label('X')
ax.set_label('Y')
ax.set_label('Z')

plt.title('3D plot for word embedding')
plt.savefig("word_embedding.jpeg")
'''

'''
# consider the second input token as the query to calculate self-attention
query = vector_embeddings[1]

attn_scores_2 = torch.empty(vector_embeddings.shape[0])
for i, x in enumerate(vector_embeddings):
    attn_scores_2[i] = torch.dot(x, query)

print("Attention score w.r.t 2nd word : ", attn_scores_2)

# Normalization of attention scores
# Goal is to obtain attention weights that sum up to 1
# Normalization is useful for interpretation and maintaining traning stability in an LLM.

attn_scores_2_temp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights: ", attn_scores_2_temp)
print("Sum : ", attn_scores_2_temp.sum())

# Attention weights are always positive

# In practice it is better to use Softmax instead of sum for normalizing.
# naive softmax function
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_scores_2_naive = softmax_naive(attn_scores_2)
print("Attention weights softmax naive : ", attn_scores_2_naive)
print("Sum : ", attn_scores_2_naive.sum())

# Using pytorch softmax
attn_scores_2_softmax = torch.softmax(attn_scores_2, dim=0)
print("Attention weights softmax pytorch : ", attn_scores_2_softmax)
print("Sum : ", attn_scores_2_softmax.sum())

# context vector
context_vec_2 = torch.zeros(query.shape)
for i, x in enumerate(vector_embeddings):
    context_vec_2 += attn_scores_2_softmax[i] * x # for each vector embedding, multiply the attention weight

print("context vector w.r.t second word: ", context_vec_2)
'''

# In the above section we learned how to create a attention weights and context vector w.r.t second word
# In this section, we will do it for all the tokens in the input.
# we will create a attention weights matrix

attn_scores = torch.empty(vector_embeddings.shape[0], vector_embeddings.shape[0])

# dot product
# for i, x in enumerate(vector_embeddings):
#     for j, y in enumerate(vector_embeddings):
#         attn_scores[i, j] = torch.dot(x, y)

# print(attn_scores)

# We can replace the above for loop with @ - matrix multiplication
attn_scores = vector_embeddings @ vector_embeddings.T
print(attn_scores)

# Normalization
attn_scores_norm = torch.softmax(attn_scores, dim=-1)
print(attn_scores_norm)
# print(attn_scores_norm[1].sum())

# Context vector
# context_vec = torch.zeros(vector_embeddings.shape)
# for i, x in enumerate(vector_embeddings):
#     context_vec_each_word = torch.zeros(1, 3)
#     for j, y in enumerate(vector_embeddings):
#         context_vec_each_word += attn_scores_norm[i][j] * y
#     context_vec[i] = context_vec_each_word

# we can replace the above for loop with @
context_vec = attn_scores_norm @ vector_embeddings


print(context_vec)