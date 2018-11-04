import ujson as json 
import os.path
import spacy
import config
from tqdm import tqdm
import re

nlp = spacy.blank("en") 

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

filename = config.train_file 
out_filename = 'corpus_for_emb_lower' 

with open(filename, "r") as fh:
    source = json.load(fh)
    query_ids = source['query_id']
    queries = source['query']
    passages = source['passages']
    answers = source.get('answers', {})

print(type(queries) is list)
print(type(passages) is list)

def clean(sent):
    sent = " ".join(word_tokenize(sent))
    #remove_list = ""
    #re.sub(r'[^\w'+remove_list+']', ' ', sent.replace('-',' ')).lower()+'\n'
    return sent.lower()+'\n'

with open(out_filename, 'w') as fo:
    for qid in query_ids:
        
        if answers.get(qid)[0].find("No Answer Present.")<0:
            fo.write(clean(answers.get(qid)[0]))
        
        for passage in passages.get(qid):
            fo.write(clean(passage['passage_text']))

        fo.write(clean(queries.get(qid)))












