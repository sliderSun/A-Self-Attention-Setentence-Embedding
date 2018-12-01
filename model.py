# coding: utf-8
import logging
import tensorflow as tf
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

class SelfAttentionSetenceEmbedding(object):
    """TensorFlow implementation of 'A Structured Self Attentive Sentence Embedding'"""
    def __init__(self, config, embedding_matrix):
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.num_classes = config['num_classes']
        self.embedding_size = config['embedding_size']
        self.hidden_layer_size = config['hidden_layer_size']
        self.beta_l2 = config['beta_l2']
        # hyperparameter from paper
        # n: sentence length
        # d: word embedding dimension
        # u : hidden state size
        self.n = config['n']
        self.d_a = config['d_a']
        self.u = config['u']
        self.r = config['r']
        # load word embedding
        self.embedding_matrix = embedding_matrix

    def add_placeholders(self):
        self.X = tf.placeholder('int32', [None, self.n])
        self.y = tf.placeholder('int32', [None, ])

    def inference(self):
        # define xavier initializer
        initializer=tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('embedding_layer'):
            # fine-tune embedding matrix
            W = tf.Variable(self.embedding_matrix, trainable=True, name='embedding_matrix', dtype='float32')
            # shape is (None, n, d)
            embedding_words = tf.nn.embedding_lookup(W, self.X)
        with tf.variable_scope('dropout_layer'):
            pass
        with tf.variable_scope('bi_lstm_layer'):
            cell_fw = tf.contrib.rnn.LSTMCell(self.u)
            cell_bw = tf.contrib.rnn.LSTMCell(self.u)
            H, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                embedding_words,
                dtype=tf.float32)
            # hidden state, shape = (batch_size, n, 2*u)
            H = tf.concat([H[0], H[1]], axis=2)
        with tf.variable_scope("attention_layer"):
            W_s1 = tf.get_variable('W_s1', shape=[self.d_a, 2*self.u],initializer=initializer)
            W_s2 = tf.get_variable('W_s2', shape=[self.r, self.d_a],initializer=initializer)
            # attention
            # shape = (r, batch_size*n)
            A = tf.nn.softmax(
                tf.matmul(W_s2,
                          tf.tanh(
                              tf.matmul(W_s1, tf.reshape(H, [2*self.u, -1]))
                          )
                )
            )
            # shape = (batch_size, r, n)
            A = tf.reshape(A, shape=[-1, self.r, self.n])
            # shape = (batch_size, r, 2*u)
            M = tf.matmul(A, H)
        with tf.variable_scope('fully_connected_layer'):
            # shape = (batch_size, 2*u*r)
            M_flatten = tf.reshape(M, shape=[-1, 2*self.u*self.r])
            # first hidden layer
            W_f1 = tf.get_variable('W_f1', shape=[self.r*self.u*2, self.hidden_layer_size], initializer=initializer)
            b_f1 = tf.get_variable('b_f1', shape=[self.hidden_layer_size], initializer=tf.zeros_initializer())
            hidden_output = tf.nn.relu(tf.matmul(M_flatten, W_f1) + b_f1)
            # output layer
            W_f2 = tf.get_variable('W_f2', shape=[self.hidden_layer_size, self.num_classes], initializer=initializer)
            b_f2 = tf.get_variable('b_f2', shape=[self.num_classes], initializer=tf.zeros_initializer())
            # shape = (batch_size, num_classes)
            self.y_output = tf.matmul(hidden_output, W_f2) + b_f2

        with tf.variable_scope('penalization_layer'):
            # shape = (batch_size, n, r)
            A_T = tf.transpose(A, perm=[0,2,1])
            # shape = (r, r)
            unit_matrix = tf.eye(self.r, dtype='float32')
            # penalization
            # subtract with broadcast
            self.penalty = tf.norm(
                tf.square(tf.matmul(A, A_T) - unit_matrix), axis=[-2,-1], ord='fro'
            )

    def add_loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_output)
        loss = loss + self.beta_l2 * self.penalty
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', self.loss)

    def add_metric(self):
        pass

    def train(self):
        # Applies exponential decay to learning rate
        self.global_step = tf.Variable(0, trainable=False)
        # define optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        """build graph for model"""
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_metric()
        self.train()
