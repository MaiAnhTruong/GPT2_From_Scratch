import torch.nn as nn
import tiktoken
import torch


class selfAttention_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Parameter(torch.rand(3, 3), requires_grad=False)
        self.W_k = nn.Parameter(torch.rand(3, 3), requires_grad=False)
        self.W_v = nn.Parameter(torch.rand(3, 3), requires_grad=False)

    def forward(self, x):
        Query = x @ self.W_q
        Key = x @ self.W_k
        Value = x @ self.W_v

        attn_score = Query @ Key.T

        # we will scale the matrix and normalize
        d_k = Key.shape[-1]
        attn_weight = torch.softmax(attn_score / d_k**0.5, dim=1)

        context_vector = attn_score @ Value

        return context_vector

# uses Linear layer rather than Parameter module
class selfAttention_v2(nn.Module):
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
        attn_weight = torch.softmax(attn_score / d_k**0.5, dim=1)

        context_vector = attn_score @ Value

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
    print("3d dims vector embeddings: ", vector_embeddings)

    cntxt_vec = selfAttention_v1()
    print(cntxt_vec(vector_embeddings))

    cntxt_vec_2 = selfAttention_v2(3, 3)
    print(cntxt_vec_2(vector_embeddings))


if __name__ == "__main__":
    main()