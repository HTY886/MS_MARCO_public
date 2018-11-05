# MS_MARCO_public
  * Work in progress
  * A Tensorflow implementation of S-Net-like(-to-be) model on [MS MARCO](http://www.msmarco.org/) dataset based on [R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) and [V-Net](https://arxiv.org/abs/1805.02220)
 
## Requirements

#### General
  * Python >= 3.4
  * gunzip, wget
#### Python Packages
  * tensorflow-gpu >= 1.5.0
  * spaCy >= 2.0.0
  * tqdm
  * ujson
  * jsonl

## Usage

To download and preprocess MS_MARCO data, run

```bash
# download MS MARCO and Glove
sh download.sh
# preprocess the data
python config.py --mode prepro
```
Hyper parameters are stored in config.py. To train/test the model, run

```bash
python config.py --mode train/test
```

To get the official score, clone the MSMARCOV2 official repository and run the evaluation script. (See README.md in the repository for more information)
```bash
git clone https://github.com/dfcf93/MSMARCOV2/tree/master/Q%2BA/Evaluation
cd MSMARCOV2/Q+A/Evaluation/
# run evaluation script
sh run.sh $answer_for_evl.json $answer/ref.json
```
`answer_for_evl.json` and `answer/ref.json` can be found in `log/answer`

The default directory for tensorboard log file is `log/event`

## Extensions

These settings may increase the score but not used in the model by default. You can turn these settings on in `config.py`. 

 * [Pretrained GloVe character embedding](https://github.com/minimaxir/char-embeddings). Contributed by yanghanxy.
 * [Fasttext Embedding](https://fasttext.cc/docs/en/english-vectors.html). Contributed by xiongyifan. May increase the F1 by 1% (reported by xiongyifan).

OR pretrain fasttext word embedding by yourself
```bash
# turn MS MARCO dataset into clean sentences as fasttext inputs
python MSMARCO2FASTTEXT.py
```
then train [fasttext](https://fasttext.cc/docs/en/unsupervised-tutorial.html) model 
on `corpus_for_emb` with dims=300

## Experiments
#### Multi-passages
|dataset|size of train-set|word-embedding|hidden-dims|bleu-1|rouge-l|
|---|---|---|---|---|---|
|subset of dev-set w/ target span|1.3M|GloVe w/ OOV|150|0.293|0.317|
|subset of dev-set w/ target span|1.3M|Fasttext|256|0.299|0.270|
|subset of dev-set w/ answer|1.3M|Fasttext|256|0.270|0.267|
|subset of dev-set w/ answer|1.3M|Fasttext|300|0.291|0.303|
|subset of dev-set w/ answer|2.1M|GloVe w/ OOV|256|0.251|0.252|


