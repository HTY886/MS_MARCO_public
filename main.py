import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os

from model import Model
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset, write_prediction


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["num_batches"]

    print("Building model...")
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    dev_dataset = get_batch_dataset(config.dev_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, dev_dataset.output_types, dev_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    model = Model(config, iterator, word_mat, char_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    lr = config.init_lr

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if config.load: 
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            print('loading model')
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))

        for _ in tqdm(range(1, config.num_steps + 1)):
            global_step = sess.run(model.global_step) + 1
            loss, train_op, loss_dict, pse = sess.run([model.loss, model.train_op, model.loss_dict, model.pse_is_select], feed_dict={
                                      handle: train_handle})
            
            #print(pse)
            if global_step % config.period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
                
            
                for loss_type in loss_dict.keys():
                    loss_sum_d = tf.Summary(value=[tf.Summary.Value(
                    tag="model/"+loss_type, simple_value=loss_dict[loss_type]), ])
                    writer.add_summary(loss_sum_d, global_step)
            
            if global_step % config.checkpoint == 0:
                
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))
                
                _, summ = evaluate_batch(
                    config, model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                for s in summ:
                    writer.add_summary(s, global_step)
                metrics, summ = evaluate_batch(
                    config, model, int(dev_total*config.passage_num/config.batch_size), dev_eval_file, sess, "dev", handle, dev_handle)
                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))
                dev_loss = metrics["loss"]
                print('dev_loss {0}'.format(dev_loss))
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(
                    config.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)
            

def evaluate_batch(config, model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    remapped_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches+1 )):
        try:    
            qa_id, loss, yp1, yp2 , y1, y2, is_select_p, is_select= sess.run(
                    [model.qa_id, model.loss, model.yp1, model.yp2, model.y1, model.y2, model.is_select_p, model.is_select], feed_dict={ handle:str_handle })
        except tf.errors.OutOfRangeError:
            break

        y1 = np.argmax(y1, axis=-1)
        y2 = np.argmax(y2, axis=-1)
        sp = np.argmax(is_select_p, axis=-1)
        s = np.argmax(is_select, axis=-1)
        sp = [ n+i*config.passage_num for i,n in enumerate(sp.tolist()) ]
        s = [ m+i*config.passage_num for i,m in enumerate(s.tolist()) ]

        answer_dict_, remapped_dict_ = convert_tokens(
            eval_file, [qa_id[n] for n in sp], [yp1[n] for n in sp], [yp2[n] for n in sp], [y1[n] for n in sp], [y2[n] for n in sp], sp, s)

        answer_dict.update(answer_dict_)
        remapped_dict.update(remapped_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict, filter=False)
    sp_metrics = evaluate(eval_file, remapped_dict, filter=False)

    metrics["loss"] = loss

    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])

    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    
    sp_f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/sp_f1".format(data_type), simple_value=sp_metrics["f1"]), ])
    sp_em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/sp_em".format(data_type), simple_value=sp_metrics["exact_match"]), ])
   
    return metrics, [loss_sum, f1_sum, em_sum, sp_f1_sum, sp_em_sum]


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["num_batches"]

    print("Loading model...")
    test_batch = get_batch_dataset(config.test_record_file, get_record_parser(
        config, is_test=True), config, is_test=True).make_one_shot_iterator()

    model = Model(config, test_batch, word_mat, char_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        answer_dict = {}
        select_right = []
        for step in tqdm(range(1, total + 1)):
            qa_id, loss, yp1, yp2 , y1, y2, is_select_p, is_select= sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2, model.y1, model.y2, model.is_select_p, model.is_select])
            y1 = np.argmax(y1, axis=-1)
            y2 = np.argmax(y2, axis=-1)
            sp = np.argmax(is_select_p, axis=-1)
            s = np.argmax(is_select, axis=-1)
            sp = [ n+i*config.passage_num for i,n in enumerate(sp.tolist()) ]
            s = [ m+i*config.passage_num for i,m in enumerate(s.tolist()) ]
            select_right.append(len(set(s).intersection(set(sp))))

            answer_dict_, _ = convert_tokens(
                eval_file, [qa_id[n] for n in sp], [yp1[n] for n in sp], [yp2[n] for n in sp], [y1[n] for n in sp], [y2[n] for n in sp], sp, s)
            answer_dict.update(answer_dict_)
            losses.append(loss)
        loss = np.mean(losses)
        select_accu = sum(select_right)/ (len(select_right)*(config.batch_size/config.passage_num))
        write_prediction(eval_file, answer_dict, 'answer_for_evl.json', config)
        metrics = evaluate(eval_file, answer_dict, filter=False)
        metrics['Selection Accuracy'] = select_accu
        
        print("Exact Match: {}, F1: {}, selection accuracy: {}".format(
            metrics['exact_match'], metrics['f1'], metrics['Selection Accuracy']))
