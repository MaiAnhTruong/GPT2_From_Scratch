import re

class Simpletokenzier():
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}


    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # print(preprocessed)
        # ids = [self.str_to_int[s] if self.str_to_int[s] else 0 for s in preprocessed ] # case where the word is not present in the dictionary not covered

        ids = []
        for each in preprocessed:
            id = self.str_to_int.get(each)
            if id == None: id = self.str_to_int.get("<|unk|>")
            ids.append(id)


        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # replace ids with words
        text = re.sub(r'\s+([,.?"()\'])', r'\1', text)
        return text


def main():

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("total number of characters in the text : ", len(raw_text))
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

    # adding two special context tokens
    preprocessed.append("<|endoftext|>")
    preprocessed.append("<|unk|>")

    preprocessed = sorted(list(set(preprocessed)))

    vocab = {token:integer for integer, token, in enumerate(preprocessed)}

    tokenizer = Simpletokenzier(vocab=vocab)

    # sample sentence to tokenize
    sentence = "The brown dog playfully chased the swift fox "
    sentence_two = " I had always thought Jack"
    full_sentence = "<|endoftext|>".join((sentence, sentence_two))
    print(full_sentence)
    id_list = tokenizer.encode(full_sentence)
    print(tokenizer.encode(full_sentence))

    # getting back word from ids
    print(tokenizer.decode(id_list))

if __name__ == "__main__":
    main()

# Special context tokens:
# If there are some words that are not present in the vocabulary, then the encoder will throw an error.
# |unk| and |endoftext|

# if words are not present, then a constant id is given for all those unknown words |unk|
# |<endoftext>| --> when working with multiple text sources, we add this token between these texts.
# These <endoftext> tokens act as markers, signaling the start of end of a particular segment.
# This leads to more effective processing and understanding in the LLM


# GPT doesn't use unknown token to encode unknown words, instead it uses something called Byte-Pair Encoding
# i.e. the words are broken down into subword units.
