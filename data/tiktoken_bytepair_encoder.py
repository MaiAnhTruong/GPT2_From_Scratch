import tiktoken

print("tiktoken version: ", tiktoken.__version__)

tokenizer = tiktoken.get_encoding("gpt2")

 # sample sentence to tokenize
sentence = "The brown dog playfully chased the swift fox "
sentence_two = " I had always thought Jack"
full_sentence = "<|endoftext|>".join((sentence, sentence_two))
print(full_sentence)

tokenized_sentence = tokenizer.encode(sentence)
print(tokenized_sentence)

tokenized_full_sentence = tokenizer.encode(full_sentence, allowed_special={'<|endoftext|>'})
print(tokenized_full_sentence)

decoded_full_sentence = tokenizer.decode(tokenized_full_sentence)
print(decoded_full_sentence)


# If BPE algorithm encounters an unfamiliar word, it will break it down into either sub words or characters
