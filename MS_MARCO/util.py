import tensorflow as tf
import numpy as np
import re
from collections import Counter
import string
import jsonlines
import os

def get_record_parser(config, is_test=False):
    def parse(example):
        passage_num = config.passage_num
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.string),
                                               "y2": tf.FixedLenFeature([], tf.string),
                                               "in_answer": tf.FixedLenFeature([], tf.string),
                                               "is_select": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(
            features["context_idxs"], tf.int32), [passage_num, para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
            features["ques_idxs"], tf.int32), [passage_num, ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(
            features["context_char_idxs"], tf.int32), [passage_num, para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(
            features["ques_char_idxs"], tf.int32), [passage_num, ques_limit, char_limit])
        y1 = tf.reshape(tf.decode_raw(
            features["y1"], tf.float32), [passage_num, para_limit])
        y2 = tf.reshape(tf.decode_raw(
            features["y2"], tf.float32), [passage_num, para_limit])
        qa_id = tf.tile(tf.expand_dims(features["id"], axis=0) , [passage_num])
        is_select = tf.reshape(tf.decode_raw(
            features["is_select"], tf.float32), [passage_num, 1]) 
        in_answer = tf.reshape(tf.decode_raw(
            features["in_answer"], tf.float32), [passage_num, para_limit])
        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id, is_select, in_answer
    return parse


def get_batch_dataset(record_file, parser, config, is_test =False):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    
    if config.is_group:
        assert(config.batch_size%config.passage_num==0)

        def concat_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id, is_select, in_answer):
            query_num = int(config.batch_size/config.passage_num)
            passage_num = config.passage_num
            para_limit = config.test_para_limit if is_test else config.para_limit
            ques_limit = config.test_ques_limit if is_test else config.ques_limit
            char_limit = config.char_limit
            context_idxs = tf.reshape(context_idxs, shape=[query_num*passage_num, para_limit])
            ques_idxs = tf.reshape(ques_idxs, shape=[query_num*passage_num, ques_limit])
            context_char_idxs = tf.reshape(context_char_idxs, shape=[query_num*passage_num, para_limit, char_limit])
            ques_char_idxs = tf.reshape(ques_char_idxs, shape=[query_num*passage_num, ques_limit, char_limit])
            y1 = tf.reshape(y1, shape=[query_num*passage_num, para_limit])
            y2 = tf.reshape(y2, shape=[query_num*passage_num, para_limit])
            qa_id = tf.reshape(qa_id, shape=[query_num*passage_num])
            is_select = tf.reshape(is_select, shape= [query_num*passage_num, 1]) 
            in_answer = tf.reshape(in_answer, shape=[query_num*passage_num, para_limit])

            return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id, is_select, in_answer
        dataset = dataset.batch(int(config.batch_size/config.passage_num)).map(concat_func)

    '''
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id, is_select):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            buckets_min = [np.iinfo(np.int32).min] + buckets
            buckets_max = buckets + [np.iinfo(np.int32).max]
            conditions_c = tf.logical_and(
                tf.less(buckets_min, c_len), tf.less_equal(c_len, buckets_max))
            bucket_id = tf.reduce_min(tf.where(conditions_c))
            return bucket_id

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    '''
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat()
    return dataset


def convert_tokens(eval_file, qa_id, pp1, pp2, yy1, yy2, ssp, ss):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2, y1, y2, sp, s in zip(qa_id, pp1, pp2, yy1, yy2, ssp, ss):
        qid = str(qid)+'P'+str(sp%10) #
        context = eval_file[qid]["context"]
        spans = eval_file[qid]["spans"]
        uuid = eval_file[qid]["uuid"]
        p1 = max(0, min(p1, len(spans)-1))
        p2 = max(0, min(p2, len(spans)-1))
        if y2:
            print("************qa_id: {}***************".format(qid))
            print("select_right: {}".format(str(sp == s)))
            print('AFTER CLIPPING=== p1: {0}, p2: {1}'.format(p1,p2))
            print('GROUND TRUTH==== y1: {0}, y2: {1}'.format(y1,y2))
        
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[qid] = context[start_idx: end_idx]
        if sp == s: 
            remapped_dict[qid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict, filter=False):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        ground_truths = eval_file[key]["answers"]

        if filter and prediction.find("NoAnSweR")>=0 :
            continue

        total += 1
        prediction = value
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    print('prediction: {0}'.format(prediction))
    print('ground-truth: {0}'.format(ground_truth))
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def write_prediction(eval_file, answer_dict, file_name, config):
    with jsonlines.open(os.path.join(config.answer_dir, 'ref.json'), mode='w') as writer1:
        with jsonlines.open(os.path.join(config.answer_dir, file_name), mode='w') as writer2:
            for key, values in answer_dict.items():
                writer2.write({"query_id":key, "answers":[answer_dict[key]]})
                writer1.write({"query_id":key, "answers":eval_file[key]["answers"]})
            
            

