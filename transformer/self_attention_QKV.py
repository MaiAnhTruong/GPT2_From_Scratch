import tiktoken
import torch

# Initialize the BPE
tokenizer = tiktoken.get_encoding("gpt2")

sentence = "Your Journey starts with one stop"
print("considered sentence: ", sentence)

tokenized_sentence = tokenizer.encode(sentence)
print("tokenized sentence: ", tokenized_sentence)

# defining vocab and vector embedding dim size
vocab = 50267 # gpt-2 has 50267 words/tokens
output_dim = 3 # gpt-2 has 748 dimension size. for our example, we will restrict to 3.

# creating dictionary
embedding_layer = torch.nn.Embedding(vocab, output_dim)

# convert tokenized sentence into tensor
tensors = torch.tensor(tokenized_sentence)
print("tokens converted into tensors: ", tensors)

# obtaining the vector embedding of the sentence based on tokens
vector_embeddings = embedding_layer(tensors)
print("3d dims vector embeddings: ", vector_embeddings)

# lets create QKV vectors
# Query vector
W_q = torch.nn.Parameter(torch.rand(3, 3), requires_grad=False)

# Key vector
W_k = torch.nn.Parameter(torch.rand(3, 3), requires_grad=False)

# Value vector
W_v = torch.nn.Parameter(torch.rand(3, 3), requires_grad=False)

Query = vector_embeddings @ W_q
Key = vector_embeddings @ W_k
Value = vector_embeddings @ W_v

print("Query: ", Query)
print("Key", Key)
print("Value: ", Value)

# Computing Attention scores
attn_score = Query @ Key.T
print(attn_score)
# will result in a 6x6 matrix. for eg. First row contains the attention score b/w itself and all other words.
# Similar to dot product shown in the self_attention.py script.

# Computing Attention weights - Normalization. 
# before normalization, we should scale the matrix by square_root(no. of columns). In our case 3
# This is also called scaled attention scores
# # Sum of each row should = 1.

d_k = Key.shape[-1]
attn_weight = torch.softmax(attn_score / d_k**0.5, dim=1)
print(attn_weight)
print(d_k)

'''
why divide by sqrt(Dimension)
1. For stability in learning
    Softmax is sensitive to magnitudes of its inputs. When the inputs are large, the difference b/w the exponential values of each i/p becomes more pronounced.
    IN attention mechanisms, particulary transformers, if the dot product b/w Q and K vectors become too large this will result in a very sharp softmax distribution.
    Making the model over confident

2. To make Variance of the dot product stable.
    Ideally we need the variance close to 1. But if we multiply 2 random numbers, variance can increase.
'''

# Computing Context vector
context_vector = attn_weight @ Value
print(context_vector)