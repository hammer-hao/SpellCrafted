# SpellCrafted: Your Custom MTG Card Conjurer

## What is SpellCrafted?

[Spellcrafted](http://spellcrafted.hammerhao.net) is a small language model trained with ~25,000 cards ever printed in Magic: The Gathering. It has 10 million parameters, and specializes at generating card texts with any given card game as prompt!

## How does it work?

When you input any arbitrary card name to the model, it first gets lammentized and converted into a vector using [Word2Vec](https://github.com/dav/word2vec). This vector gets passed down to multiple networks. It gets passed into multiple classification networks to determine the color, mana cost, and type of your card. It gets passed into a tranformer decoder which sequentially generates output tokens for card text.

For the transformer decoder, we trained our own tokenizers using the [🤗 Tokenizers](https://huggingface.co/docs/transformers/main_classes/tokenizer) implementation. 

## Who made SpellCrafted?

The model architecture for SpellCrafted was conjured by human planeswalker Zihe Hao. Training was done on a RTX4090 owned by Yuan Zhai. Human engineer Jianpeng Yin developed the frontend website.

## How do I train this model on my own?

The card text data we used are readily available at [Scryfall API](https://scryfall.com/docs/api/bulk-data). There exists card data files for all languages the cards are available in. You can train the model in another language or even train a card text translator!