model_path='mtggenerator_v3.pt'

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.nn import functional as F
import re

batch_size=512
block_size=36
sampling_size=24
max_iters=7000
eval_interval=300
learning_rate=3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_heads = 4
n_layers = 5
dropout=0.3
vocab_size=29500

class Head(nn.Module):
    #one self attention head

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias= False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v=self.value(x)
        out=wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """multi head attention"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(head_size*num_heads, n_embd)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)
    
class FeedForward(nn.Module):
    """simple feedforward perceptron layer"""
    def __init__(self, n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block: multihead self attention followed by one Feedforward layer"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa=MultiHeadAttention(n_head, head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x+self.sa(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x


class MTGCardGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size, n_embd) #each token directly look up the logit of the next token from a lookup table
        self.lmhead=nn.Linear(n_embd, vocab_size)
        self.position_embedding_table=nn.Embedding(block_size, n_embd) #each token gets a position embeding of block_size, stores the relative position of token in the block
        self.block=nn.Sequential(*[Block(n_embd, n_head=n_heads) for _ in range(n_layers)])
    
    def forward(self, idx, targets=None):
        
        B, T = idx.shape

        #idx and targets are both (B,T) tensors of integers, where B=batch number, T=position in batch
        token_embeddings=self.token_embedding_table(idx) #look up value corresponding to own position in the token embedding table to form C (channel value)
        position_embeddings=self.position_embedding_table(torch.arange(T, device=device)) #add position embeddings to token embedding
        x= token_embeddings + position_embeddings
        x = self.block(x)
        logits=self.lmhead(x)

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
            #crop idx to max block size
            idx_cond=idx[:, -block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            #use logits only, focus only on last time step
            logits = logits[:, -1, :] #keep only last time step ---> (B, C)
            #apply softmax on logit to get distribution
            probs = F.softmax(logits, dim=-1) #get a (B, C) matrix of probabilities, sum(prob) of each B = 1
            #sample from the distribution
            idx_next=torch.multinomial(probs, num_samples=1) #get a (B, 1) array of predictions
            #append prediction to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #now a (B, T+1) matrix of returned results
        return idx

model = MTGCardGenerator()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
model.eval()
m=model.to(device)

tokenizer = Tokenizer.from_file("mtggenerator.json")
vocab_size=tokenizer.get_vocab_size()

#create the mapping from characters to integers
encode = lambda text: tokenizer.encode(text).ids #encode: take a string, output a list of integers
decode = lambda list: tokenizer.decode(list) #decode: take a list of integers, output a string

def generate(cardname):
    context= torch.tensor([encode(f'[CLS] {cardname}: [SEP]')], dtype=torch.long, device=device)
    response=m.generate(context, max_new_tokens=250)[0].tolist()
    indices = [i for i, x in enumerate(response) if x == 2]
    slices = [response[i+1:j] for i, j in zip([0] + indices, indices + [None])]
    out = [re.sub(r'{+\s', r'{', re.sub(r'\s+}', r'}', re.sub(r'\s+([.,!?:])', r'\1', decode(slice)))).replace('~', cardname) for slice in slices]
    try:
        mana=out[1]
        type=out[2]
        desc=out[3]
    except IndexError:
        mana='Error'
        type='Error'
        desc='Try Again'
    return cardname, mana, type, desc
generate('boy')