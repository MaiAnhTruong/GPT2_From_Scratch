'''
Multi-Head Attention is implementing multiple single head attention (causal attention) mechanism parallelly.

For pictorial expalnation refer to Ipad for notes. For comments of single head causal attention mechanism refer to the script.
'''
import torch
import torch.nn as nn
import tiktoken

# Implementation of Single-head Attention
class causalAttention_v3(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        Query = self.W_q(x)
        Key = self.W_k(x)
        Value = self.W_v(x)

        attn_score = Query @ Key.transpose(1, 2) # changed transpose

        print("attention score - ", attn_score.shape)
        # fill the matrix above the diagonal with negative infinity
        masked_attn_scores = attn_score.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
        
        # scale the matrix and normalize
        d_k = Key.shape[-1]
        attn_weights = torch.softmax(masked_attn_scores / d_k**0.5, dim=1)
    
        attn_weights = self.dropout(attn_weights)
                        
        context_vector = attn_weights @ Value

        return context_vector
    
'''
In code, stacking multiple instances of causalAttention_v3 will give us Multi-Head Attention.
'''

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [causalAttention_v3(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    

def main():
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
    embedding_layer = nn.Embedding(vocab, output_dim)

    # convert tokenized sentence into tensor
    tensors = torch.tensor(tokenized_sentence)
    print("tokens converted into tensors: ", tensors)

    # obtaining the vector embedding of the sentence based on tokens
    vector_embeddings = embedding_layer(tensors)
    print("3d dims vector embeddings: ", vector_embeddings.shape)

    # usually the input is given in batches to the model. Current shape is 3 x 6 x 3 (B x No_of_words x Dimension_per_word)
    batch = torch.stack((vector_embeddings, vector_embeddings, vector_embeddings), dim=0)
    print(batch.shape) # batch x num_tokens x dimension_size_of_each_token

    multihead_attention = MultiHeadAttentionWrapper(3, 3, batch.shape[1], 0.1, 4)
    cntxt_vec_2 = multihead_attention(batch) # context vector shape is 3 x 6 x 12
    print(cntxt_vec_2.shape) 

 
if __name__ == "__main__":
    main()