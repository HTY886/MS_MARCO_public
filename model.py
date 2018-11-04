import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, weighted_loss, content_model


class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id, self.is_select, self.in_answer= batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=True)  
        self.char_mat = tf.get_variable(
            "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        if opt:
            N, CL = config.batch_size, config.char_limit
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
            self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
            self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
            self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
            self.in_answer = tf.slice(self.in_answer, [0, 0], [N, self.c_maxlen])
        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

        self.ch_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        self.ready()

        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
        gru = cudnn_gru if config.use_cudnn else native_gru

        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                ch_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.ch), [N * PL, CL, dc])
                qh_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.qh), [N * QL, CL, dc])
                ch_emb = dropout(
                    ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                qh_emb = dropout(
                    qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                cell_fw = tf.contrib.rnn.GRUCell(dg)
                cell_bw = tf.contrib.rnn.GRUCell(dg)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
                ch_emb = tf.concat([state_fw, state_bw], axis=1)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
                qh_emb = tf.concat([state_fw, state_bw], axis=1)
                qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
                ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])

            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            att = rnn(qc_att, seq_len=self.c_len)

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            match = rnn(self_att, seq_len=self.c_len) #[10, ?,300]

        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
            )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            logits1, logits2 = pointer(init, match, d, self.c_mask)

        with tf.variable_scope("content_modeling"):

            logits4, c_semantics = content_model(init, match, config.hidden)

        with tf.variable_scope("cross_passage_attention"):
            self.query_num = int(config.batch_size/config.passage_num)
            c_semantics = tf.reshape(c_semantics, shape=[self.query_num, config.passage_num, -1])
            attnc_key = tf.tile(tf.expand_dims(c_semantics, axis=2), [1, 1, config.passage_num, 1])
            attnc_mem = tf.tile(tf.expand_dims(c_semantics, axis=1), [1, config.passage_num, 1, 1])
            attnc_w = tf.reduce_sum(attnc_key*attnc_mem, axis=-1)
            attnc_mask = tf.ones([config.passage_num, config.passage_num])-tf.diag([1.0]*config.passage_num)
            attnc_w = tf.nn.softmax(attnc_w*attnc_mask, axis=-1)
            attncp = tf.reduce_sum(tf.tile(tf.expand_dims(attnc_w, axis=-1), [1, 1, 1, 2*config.hidden])*attnc_mem, axis= 2)
        
        
        with tf.variable_scope("pseudo_label"):
            self.is_select = tf.reshape(tf.squeeze(self.is_select), shape=[self.query_num, config.passage_num])
            self.is_select = self.is_select/tf.tile(tf.reduce_sum(self.is_select, axis=-1, keepdims=True), [1, config.passage_num])
            sim_matrix = attnc_w
            lb_matrix = tf.tile(tf.expand_dims(self.is_select, axis=1), [1, config.passage_num, 1])
            self.pse_is_select = tf.reduce_sum(sim_matrix*lb_matrix, axis=-1) + tf.constant([0.00000001]*config.passage_num, dtype=tf.float32)    # avoid all zero
            self.pse_is_select = self.pse_is_select/tf.tile(tf.reduce_sum(self.pse_is_select, axis=-1, keepdims=True), [1,config.passage_num])
            alpha = 0.7
            self.fuse_label = alpha*self.is_select + (1-alpha)*tf.stop_gradient(self.pse_is_select)
        

        with tf.variable_scope("predict_passage"):
            init = tf.reshape(init, shape=[self.query_num, config.passage_num, -1])
            attn_concat = tf.concat([init, attncp, c_semantics], axis=-1)
            d1 = tf.layers.dense(attn_concat, 2*config.hidden, activation= tf.nn.leaky_relu, bias_initializer= tf.glorot_uniform_initializer()) #150
            d2 = tf.layers.dense(d1, config.hidden, activation= tf.nn.leaky_relu, bias_initializer= tf.glorot_uniform_initializer()) #75
            logits3 = tf.squeeze(tf.layers.dense(d2, 1, activation= None, bias_initializer= tf.glorot_uniform_initializer()))
        
        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 30)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            #logits3 = tf.reduce_max(tf.reduce_max(outer, axis=2), axis=1)
            self.is_select_p = tf.nn.sigmoid(logits3)

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits1, labels=tf.stop_gradient(self.y1))
            losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits2, labels=tf.stop_gradient(self.y2))
           
            weighted_losses = weighted_loss(config, 0.000001, self.y1, losses) #0.01
            weighted_losses2 = weighted_loss(config, 0.000001, self.y2, losses2) #0.01
            
            losses3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits3, labels=tf.stop_gradient(self.fuse_label)))
            
            in_answer_weight = tf.ones_like(self.in_answer) + 3*self.in_answer
            
            losses4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits4, labels=tf.stop_gradient(self.in_answer))*in_answer_weight, axis=-1)

            weighted_losses4 = weighted_loss(config, 0.000001, self.in_answer, losses4)
            
            self.loss_dict = {'pos_s loss':losses, 'pos_e loss':losses2, 'select loss':losses3, 'in answer':losses4}
            for key, values in self.loss_dict.items():
                self.loss_dict[key] = tf.reduce_mean(values)
            
            self.loss = tf.reduce_mean(weighted_losses + weighted_losses2 + losses3+ weighted_losses4)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
