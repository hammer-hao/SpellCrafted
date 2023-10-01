import pandas as pd
import numpy as np
import re
import torch
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(69)
batch_size=32
block_size=8
max_iters=10000
eval_interval=300
learning_rate=1e-2
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200

mtg_df=pd.read_csv('mtg_data.csv', index_col=0)
mtg_df=mtg_df.dropna(subset='text')

#pre-processing to get rid of unregonizable characters
rare_char={
    '¡®°²½˝̶π—―’„•…™−∞☐œŠ':'',
    'Äàáâãä':'a',
    'Éèéêë':'e',
    'Ææ':'ae',
    'Óóö':'o',
    'úûü':'u',
    'íī':'i',
    'Ññ':'n'
}
for rarechar, target in rare_char.items():
    for char in [*rarechar]:
        mtg_df['text']=mtg_df['text'].str.replace(char, target)

text_list=list(mtg_df['text'])
text_len=np.array([len(desc) for desc in text_list])

text_total='\n'.join(text_list)
chars=sorted(list(set(text_total)))
vocab_size=len(chars)

#create the mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] #encode: take a string, output a list of integers
decode = lambda li: ''.join([itos[i] for i in li]) #decode: take a list of integers, output a string

#convert data to 2d tensor
encoded_text_list=[torch.Tensor(encode(text)) for text in text_list]
max_len=max([len(item) for item in encoded_text_list])
padded_text_list=[torch.cat((item, torch.ones(max_len-len(item)))) for item in encoded_text_list]

data = pad_sequence(padded_text_list, batch_first=True).long() # N_cards * Char_length

#train test split
n_train=int(0.9*data.shape[0])
train_data=data[:n_train]
val_data=data[n_train:]

def get_batch(split):
    #generates a small batch of data input x and target y
    data = train_data if split == 'train' else val_data
    ix = torch.stack([torch.randint(data.shape[0], (batch_size, )), torch.randint(256 - block_size, (batch_size, ))]).T
    x = torch.stack(tuple(data[i[0]][i[1]:i[1] + block_size] for i in ix))
    y = torch.stack(tuple(data[i[0]][i[1] + 1:i[1] + block_size + 1] for i in ix))
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split]=losses.mean()
    model.train()
    return out

class BigranLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size, vocab_size) #each token directly look up the logit of the next token from a lookup table
    
    def forward(self, idx, targets=None):
        #idx and targets are both (B,T) tensors of integers, where B=batch number, T=position in batch
        logits=self.token_embedding_table(idx) #look up value corresponding to own position in the token embedding table to form C (channel value)

        if targets is None:
            loss=None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            #logits are therefore values associated with each character
            loss=F.cross_entropy(logits, targets) #evaluate loss

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #get the predictions
            logits, loss = self(idx)
            #use logits only, focus only on last time step
            logits = logits[:, -1, :] #keep only last time step ---> (B, C)
            #apply softmax on logit to get distribution
            probs = F.softmax(logits, dim=-1) #get a (B, C) matrix of probabilities, sum(prob) of each B = 1
            #sample from the distribution
            idx_next=torch.multinomial(probs, num_samples=1) #get a (B, 1) array of predictions
            #append prediction to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #now a (B, T+1) matrix of returned results
        return idx

model=BigranLanguageModel()
m=model.to(device)

#create new optimizer
optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):
    # every once in a while evaluate the loss of train and val
    if iter % eval_interval == 0:
        losses=estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    
    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context= torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=250)[0].tolist()))