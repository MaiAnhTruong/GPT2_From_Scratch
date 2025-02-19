'''
context length = 4

consider text: "In the heart of the city stood the old library, a relic from a bygone era."

input tensor: x = tensor(["In", "the", "heart", "of"],
                         ["the", "city", "stood", "the"],
                         ["old", "library", ",", "a"],
                         ["relic", "from", "a", "bygone"])

output tensor: y = tensor(["the", "heart", "of", "the"],
                          ["city", "stood", "the", "old"],
                          ["library", "a", "relic", "from"],
                          ["from", "a", "bygone", "era"])

'''
import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken



class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        
        # sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            output_chunk = token_ids[i+1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index] # __getitem__ function is an iterator. will output one input-output pair at a time.
    

if __name__ == "__main__":

    tokenizer = tiktoken.get_encoding("gpt2")

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    GPT_Dataset = GPTDataset(raw_text, tokenizer=tokenizer, max_length=4, stride=16)
    GPT_Dataloader = DataLoader(dataset=GPT_Dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    # creating a vector embedding and positional embedding

    vocab_size = 50267 # total number of words. Eg. ChatGPT-2 had a vocab size of 50267
    output_dim = 256 # dimension of the vector embedding for each token. ChatGPT-2 had dimension of 768 for each word to capture the complex semantic meaning of words
    torch.manual_seed(232)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # 50267 x 256

    context_length = max_length = 4
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) # 4 x 256

    pos_embeddings = pos_embedding_layer(torch.arange(max_length)) # 4 x 256
    
    for (x, y) in GPT_Dataloader:
        token_embeddings = embedding_layer(x) # 8 x 4 x 256 (Batch x context_size x output_dim)
        
        input_embeddings = token_embeddings + pos_embeddings # broadcasting --> 8 x 4 x 256 # the shape of pos_embedding is 4x256. shape of token_embedding is 8x4x256.
                                                             # when we broadcast, pos_embedding is duplicated to match the shape of token_embedding

        print(input_embeddings.shape)
        
        exit()