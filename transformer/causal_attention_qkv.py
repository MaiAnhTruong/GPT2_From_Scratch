import torch.nn as nn
import tiktoken
import torch


'''
# uses Linear layer rather than Parameter module
First implementation of Causal Attention
'''
class causalAttention_v1(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        Query = self.W_q(x)
        Key = self.W_k(x)
        Value = self.W_v(x)

        attn_score = Query @ Key.T

        # we will scale the matrix and normalize
        d_k = Key.shape[-1]
        attn_weights = torch.softmax(attn_score / d_k**0.5, dim=1)

        # In Causal attention weights matrix everything above the diagonal will be zero. Since the model shouldn't see the next, we mask them
        context_length = attn_weights.shape[0]
        mask_simple = torch.tril(torch.ones(context_length, context_length))
        
        causal_attn_weights = attn_weights*mask_simple
        
        # every row should sum up to 1, hence normalizing
        row_sums = causal_attn_weights.sum(dim=1, keepdim=True)
        causal_attn_weights = causal_attn_weights / row_sums
                
        context_vector = causal_attn_weights @ Value

        return context_vector


'''
The above implementation of causal attention causes something known as data leakage.
i.e although we are masking the future words to avoid their influence before sending it to the model, 
we apply softmax (ie. we normalize) to the entire row before masking such that the future words influence is already applied on the current word

To resolve this issue, we come with a neat trick. i.e
we replace all the values above the diagonal in the attn_scores matrix with -inf.
This way when we apply softmax they become zero and thus do not influence the previous words.
'''
class causalAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        Query = self.W_q(x)
        Key = self.W_k(x)
        Value = self.W_v(x)

        attn_score = Query @ Key.T

        context_length = attn_score.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked_attn_scores = attn_score.masked_fill(mask.bool(), -torch.inf) 
        
        # we will scale the matrix and normalize
        d_k = Key.shape[-1]
        attn_weights = torch.softmax(masked_attn_scores / d_k**0.5, dim=1)
        print(attn_weights)
                        
        context_vector = attn_weights @ Value

        return context_vector

'''
We will further update the above causal attention network with dropout
For theory, refer to oneNote on Ipad

Along with implementation of dropout, modified code to accept batches of data instead of single sequence of tokens
'''

class causalAttention_v3(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
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

    # usually the input is given in batches to the model. Current shape is 2 x 6 x 3 (B x No_of_words x Dimension_per_word)
    batch = torch.stack((vector_embeddings, vector_embeddings), dim=0)
    print(batch.shape) # batch x num_tokens x dimension_size_of_each_token

    causal_attention_qkv = causalAttention_v3(3, 3, batch.shape[1], 0.1)
    cntxt_vec_2 = causal_attention_qkv(batch)
    print(cntxt_vec_2.shape)

 
if __name__ == "__main__":
    main()