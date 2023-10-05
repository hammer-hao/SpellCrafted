import pandas as pd
import numpy as np
import re

mtg_df=pd.read_csv('..\mtg_data.csv', index_col=0)
mtg_df=mtg_df.dropna(subset='text')
mtg_df.head(5)

#pre-processing to get rid of unregonizable characters
rare_char={
    '¡®°²½˝̶π’„•…™−∞☐œŠ':'',
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

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

#pre-split into items seperated by whitespaces
tokenizer.pre_tokenizer = Whitespace()

#pre-processing the strings to create special tokens

tokenizer.train_from_iterator(text_list, trainer=trainer)

vocab_size=tokenizer.get_vocab_size()
print(f'vocab size:{vocab_size}')
tokenizer.save('mtggenerator.dict')