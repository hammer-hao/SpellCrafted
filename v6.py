import numpy as np
import torch
import torch.nn as nn

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim.downloader

from torch.distributions import Categorical
from tokenizers import Tokenizer
from torch.nn import functional as F
import re

torch.manual_seed(69)
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

tokenizer = Tokenizer.from_file("mtggenerator.json")
vocab_size=tokenizer.get_vocab_size()

#create the mapping from characters to integers
encode = lambda text: tokenizer.encode(text).ids #encode: take a string, output a list of integers
decode = lambda list: tokenizer.decode(list) #decode: take a list of integers, output a string

type_dict={
    0:'Creature',
    1:'Sorcery',
    2:'Artifact',
    3:'Enchantment',
    4:'Instant',
    5:'Land',
}

color_dict={
    0:'W',
    1:'U',
    2:'R',
    3:'G',
    4:'B',
    5:'Colorless',
}

wiki_vectors = gensim.downloader.load('glove-wiki-gigaword-50')

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


class MLPColorClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPColorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out


# Define the input size, hidden layer size, and number of classes
input_size = 50
hidden_size = 128
num_classes = 6

# Create an instance of the MLPClassifier
model = MLPClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('type.pt'))
model.eval()

# Create an instance of the MLPClassifier
color_model = MLPColorClassifier(input_size, hidden_size, num_classes)
color_model.load_state_dict(torch.load('color.pt'))
color_model.eval()

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

def generate_embedding(phrase):
    words = word_tokenize(phrase)
    words = [word.lower() for word in words]
    lemmatizer=WordNetLemmatizer()
    #stemmer=PorterStemmer()
    words = [lemmatizer.lemmatize(word) for word in words]
    #words = [stemmer.stem(word) for word in words]
    total_vector=[]
    for word in words:
        try:
            total_vector.append(wiki_vectors.word_vec(word))
        except KeyError:
            pass
    if len(total_vector)!=0:
        out = np.mean(total_vector, axis=0)
    else:
        out = np.zeros(50)
    return out

def generate_type(cardname):
    probs = model(torch.tensor(generate_embedding(cardname)))
    distribution = Categorical(probs)
    sampled_index = distribution.sample()
    for key, type in type_dict.items():
        print(f'{type}:{probs[key]}')
    return type_dict[int(sampled_index)]

def generate_color(cardname):
    probs = color_model(torch.tensor(generate_embedding(cardname)))
    distribution = Categorical(probs)
    sampled_index = distribution.sample()
    for key, type in color_dict.items():
        print(f'{type}:{probs[key]}')
    return color_dict[int(sampled_index)]

def generate(cardname):
    type=generate_type(cardname=cardname)
    color=generate_color(cardname=cardname)
    return f'[CLS] {cardname}: [SEP] {{{color}}} [SEP] {type} —'

model_path='mtggenerator_v3.pt'

model = MTGCardGenerator()
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.eval()
m=model.to(device)