#!/usr/bin/env bash

# Download MS_MARCO dataset
MSMARCO_DIR=~/data/MS_MARCO
mkdir -p $MSMARCO_DIR
wget https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz -O $MSMARCO_DIR/train_v2.1.json.gz
wget https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz -O $MSMARCO_DIR/dev_v2.1.json.gz
wget https://msmarco.blob.core.windows.net/msmarco/eval_v2.1_public.json.gz -O $MSMARCO_DIR/eval_v2.1_public.json.gz

gzip -d $MSMARCO_DIR/train_v2.1.json.gz > $MSMARCO_DIR/train_v2.1.json
gzip -d $MSMARCO_DIR/dev_v2.1.json.gz > $MSMARCO_DIR/dev_v2.1.json
gzip -d $MSMARCO_DIR/eval_v2.1_public.json.gz > $MSMARCO_DIR/eval_v2.1_public.json



# Download GloVe
GLOVE_DIR=~/data/glove
mkdir -p $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR

# Download Glove Character Embedding
# wget https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt -O $GLOVE_DIR/glove.840B.300d-char.txt

# Download fasttext
# FASTTEXT_DIR=~/data/fasttext
# mkdir -p $FASTTEXT_DIR
# wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip -O $FASTTEXT_DIR/wiki-news-300d-1M.vec.zip
# unzip $FASTTEXT_DIR/wiki-news-300d-1M.vec.zip -d $FASTTEXT_DIR

# Download Spacy language models
python3 -m spacy download en
