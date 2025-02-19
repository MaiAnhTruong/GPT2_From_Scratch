import torch
import torch.nn as nn
import tiktoken


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out) # Linear layer to combine head outputs
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)) # to mask the upper diagonal matrix. To make sure  data leakage doesn't occur.

    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # we implicitly split the matrix by adding a 'num_heads' dimension
        # unroll last dim: (batch, num_tokens, d_out) --> (batch, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim)  --> (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute Scaled dot product attention (aka self-attention) with causal mask
        attn_scores = queries @ keys.transpose(2, 3) # Dot product for each head

        # original mask truncated to number of tokens and converted into boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        # use mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
        attn_weights = self.dropout(attn_weights)

        # Change the final Shape to original : (batch, num_heads, num_tokens, head_dim) --> (batch, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection.

        return context_vec


'''
If the:
input shape = 2,6,6 = (batch, no_words/tokens, dim_per_word)

let's say, num of attention heads = 2
Then the dim_per_word of each word is split into 2
i.e.
self.head_dim = d_out // num_heads

d_in = number of dimension of each word.

usually in LLMs d_in = d_out (d_out is decided by user, it is a hyperparameter)

head_dim = 6 // 2 = 3
so the input shape is reshaped to (batch, no_words/tokens, num_heads, head_dim) --> (2, 6, 2, 3)
i.e dimension size of each word vector is split equally by number of attention heads
more specifically, If there are 2 attention heads, the vectore embedding of the word is split by half and given to each head.

At the end, After all operations, ie. after calculating the context vector, they are concatenated back to original dimension.

Formaula for Attention is given as :

Attention(Q,K,V) = softmax((Q.K^T) / sqrt(d_out)) * V

head(i) = Attention(Q.Wi^Q, KWi^K, VWi^V)

'''


def main():
    # Initialize the BPE
    tokenizer = tiktoken.get_encoding("gpt2")

    sentence = "Your Journey starts with one stop"
    print("considered sentence: ", sentence)

    tokenized_sentence = tokenizer.encode(sentence)
    print("tokenized sentence: ", tokenized_sentence)

    # defining vocab and vector embedding dim size
    vocab = 50267 # gpt-2 has 50267 words/tokens
    output_dim = 6 # gpt-2 has 748 dimension size. for our example, we will restrict to 3.

    # creating dictionary
    embedding_layer = nn.Embedding(vocab, output_dim)

    # convert tokenized sentence into tensor
    tensors = torch.tensor(tokenized_sentence)
    print("tokens converted into tensors: ", tensors)

    # obtaining the vector embedding of the sentence based on tokens
    vector_embeddings = embedding_layer(tensors)
    print("3d dims vector embeddings: ", vector_embeddings.shape)

    # usually the input is given in batches to the model. Current shape is 3 x 6 x 6 (B x No_of_words x Dimension_per_word)
    batch = torch.stack((vector_embeddings, vector_embeddings, vector_embeddings), dim=0)
    print(batch.shape) # batch x num_tokens x dimension_size_of_each_token

    multihead_attention = MultiHeadAttention(batch.shape[2], batch.shape[2], batch.shape[1], 0.1, 2)
    cntxt_vec_2 = multihead_attention(batch) # context vector shape is 3 x 6 x 6. 
    print(cntxt_vec_2.shape) 

 
if __name__ == "__main__":
    main()

