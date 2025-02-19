
import tiktoken

print("tiktoken version: ", tiktoken.__version__)

tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:] # removing first 50 words/tokens

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print("x: ", x)
print("y:       ", y)

# iterating over the enc_sample to get x & y
for i in range(1, context_size+1):
        context = enc_sample[:i] # input(x) to the LLM model
        desired = enc_sample[i] # output(y) that the model should predict

        # print(context, "----> ",  desired)
        print(tokenizer.decode(context), "---> ", tokenizer.decode([desired])) # input to tokenizer.decode() function is a list



# implementing efficient dataloader
