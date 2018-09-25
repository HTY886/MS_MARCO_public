import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
import os.path

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}

    with open(filename, "r") as fh:
        source = json.load(fh)
        query_ids = source['query_id']
        queries = source['query']
        passages = source['passages']
        answers = source.get('answers', {})

        flat = ((qid, passages[qid], queries[qid], answers.get(qid)) for qid in query_ids)
        q_num = len(source['query_id'])

        organized, filtered_out = _organize(flat, 0, q_num)

        for qid, passage, query, ans, position in tqdm(organized):
            context = passage['passage_text'].replace(
                "''", '" ').replace("``", '" ')
            context_tokens = word_tokenize(context)
            context_chars = [list(token) for token in context_tokens]
            spans = convert_idx(context, context_tokens)
            for token in context_tokens:
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1

            #for qa in para["qas"]:
            ques = query.replace(
                "''", '" ').replace("``", '" ')
            ques_tokens = word_tokenize(ques)
            ques_chars = [list(token) for token in ques_tokens]
            for token in ques_tokens:
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1

            y1s, y2s = [], []
            answer_texts = []
            #for answer in answers.get(qid):
            answer_text = ans
            answer_start = position[0]
            answer_end = position[-1]
            answer_texts.append(answer_text)
            answer_span = []
            is_select = False
            if answer_end - answer_start:
                is_select = True
                print('a {0} {1}'.format(answer_start, answer_end - answer_start))

            for idx, span in enumerate(spans):
                if not (answer_end <= span[0] or answer_start >= span[1]):
                    answer_span.append(idx)

            if(len(answer_span)):
                y1, y2 = answer_span[0], answer_span[-1]
            else:
                y1, y2 = 0, 0
            
            y1s.append(y1)
            y2s.append(y2)
            print(y1)
            example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                    "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": int(qid), "is_select": is_select}
            examples.append(example)
            eval_examples[str(qid)] = {
                "context": context, "spans": spans, "answers": answer_texts, "uuid": int(qid)}
        # random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


# PASTED FROM MSMARCOV2/BidafBaseline/scripts/dataset.py
def _organize(flat, span_only, q_num):
    """
    Filter the queries and consolidate the answer as needed.
    """
    filtered_out = set()

    organized = []

    total = q_num

    for qid, passages, query, answers in tqdm(flat):

        '''
            for muli-p
        '''

        if len(passages) is not 10 or answers is None or answers[0] is "No Answer Present.":
            filtered_out.add(qid)
            continue  # Skip non-answered queries
        
        matching = set()

        ans = answers[0]
        
        if len(ans) == 0:
            continue
        
        for ind, passage in enumerate(passages):
            pos = passage['passage_text'].find(ans)
            if pos >= 0:
                matching.add(ind)
                organized.append((qid, passage, query, ans, (pos, pos+len(ans))))
    
            if not span_only:
                if ind in matching:
                    continue
                if passage.get('is_selected', False):
                    matching.add(ind)
                    organized.append((qid, passage, query, ans, (0, len(passage))))
                else:
                    matching.add(ind)
                    organized.append((qid, passage, query, ans, (0, 0)))
        # Went through the whole thing. If there's still not match, then it got
        # filtered out.
        if len(matching) == 0:
            filtered_out.add(qid)
        if len(organized)> 0.1*total:
            break

    '''
    if len(filtered_out) > 0:
        assert span_only  
    '''
    return organized, filtered_out


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None, token2idx_dict=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 2)} if token2idx_dict is None else token2idx_dict
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):

    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    char_limit = config.char_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    qid_example_dict = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)
        is_select = np.zeros([1], dtype=np.float32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example["y1s"][-1], example["y2s"][-1]
        print(start)
        y1[start], y2[end] = 1.0, 1.0

        if example["is_select"]:
            is_select[0] = 1.0

        q_id = example["id"]

        if q_id not in qid_example_dict.keys():
            qid_example_dict[q_id] = {"context_idxs":[], "ques_idxs":[], "context_char_idxs":[], "ques_char_idxs":[], "y1":[], "y2":[], "is_select":[], "count":0 }

        qid_example_dict[q_id]["context_idxs"].append(context_idxs)
        qid_example_dict[q_id]["ques_idxs"].append(ques_idxs)
        qid_example_dict[q_id]["context_char_idxs"].append(context_char_idxs)
        qid_example_dict[q_id]["ques_char_idxs"].append(ques_char_idxs)
        qid_example_dict[q_id]["y1"].append(y1)
        qid_example_dict[q_id]["y2"].append(y2)
        qid_example_dict[q_id]["is_select"].append(is_select)
        qid_example_dict[q_id]["count"]+= 1


    for qid in tqdm(qid_example_dict.keys()):
        
        if qid_example_dict[q_id]["count"] is not config.passage_num:
            continue

        context_idxs_b = np.asarray(qid_example_dict[q_id]["context_idxs"])
        ques_idxs_b = np.asarray(qid_example_dict[q_id]["ques_idxs"])
        context_char_idxs_b =  np.asarray(qid_example_dict[q_id]["context_char_idxs"])
        ques_char_idxs_b = np.asarray(qid_example_dict[q_id]["ques_char_idxs"])
        y1_b = np.asarray(qid_example_dict[q_id]["y1"])
        y2_b = np.asarray(qid_example_dict[q_id]["y2"])
        print(y1_b)
        print(y2_b)
        is_select_b = np.asarray(qid_example_dict[q_id]["is_select"])
        
        record = tf.train.Example(features=tf.train.Features(feature={
                                  "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs_b.tostring()])),
                                  "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs_b.tostring()])),
                                  "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs_b.tostring()])),
                                  "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs_b.tostring()])),
                                  "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1_b.tostring()])),
                                  "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2_b.tostring()])),
                                  "is_select": tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_select_b.tostring()])),
                                  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[qid]))
                                  }))
        writer.write(record.SerializeToString())
    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(
        config.train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(
        config.dev_file, "dev", word_counter, char_counter)
    test_examples, test_eval = process_file(
        config.test_file, "test", word_counter, char_counter)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word2idx_dict = None
    if os.path.isfile(config.word2idx_file):
        with open(config.word2idx_file, "r") as fh:
            word2idx_dict = json.load(fh)
    word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", emb_file=word_emb_file,
                                                size=config.glove_word_size, vec_size=config.glove_dim, token2idx_dict=word2idx_dict)

    char2idx_dict = None
    if os.path.isfile(config.char2idx_file):
        with open(config.char2idx_file, "r") as fh:
            char2idx_dict = json.load(fh)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim, token2idx_dict=char2idx_dict)

    build_features(config, train_examples, "train",
                   config.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev",
                              config.dev_record_file, word2idx_dict, char2idx_dict)
    test_meta = build_features(config, test_examples, "test",
                               config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.word2idx_file, word2idx_dict, message="word2idx")
    save(config.char2idx_file, char2idx_dict, message="char2idx")
    save(config.test_meta, test_meta, message="test meta")
