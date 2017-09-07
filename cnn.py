import tensorflow as tf
from common import *
import datetime


class TextCNN(object):
    def __init__(self,
                 n_sents,
                 n_words,
                 vocab_size,
                 embedding_size,
                 sent_filter_sizes=[2, 3, 4, 5],
                 sent_nb_filter=15,
                 sent_embed_size=None,
                 doc_filter_sizes=[1, 2, 3],
                 doc_nb_filter=10,
                 doc_embed_size=None,
                 sent_kmax=10,
                 doc_kmax=10,
                 learning_rate=0.001,
                 phase=True):
        with tf.name_scope('init_model'):
            self.n_sents = n_sents
            self.n_words = n_words
            self.vocab_size = vocab_size
            self.embedding_size = embedding_size
            self.sent_filter_sizes = sent_filter_sizes
            self.sent_nb_filter = sent_nb_filter
            self.sent_embed_size = sent_embed_size
            self.doc_filter_sizes = doc_filter_sizes
            self.doc_nb_filter = doc_nb_filter
            self.doc_embed_size = doc_embed_size
            self.sent_kmax = sent_kmax
            self.doc_kmax = doc_kmax
            self.learning_rate = learning_rate
            self.phase = tf.placeholder(tf.bool, name='phase') # batch norm phase - train or test
            self.sess = tf.get_default_session()
            
            assert (self.sess is not None and 
                    not self.sess._closed), 'tensorflow session should be active'
            
            self.global_step = tf.get_variable("global_step", initializer=tf.constant(0), trainable=False)
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)

            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.LT = tf.get_variable('LT',
                    initializer=tf.constant(0.0, shape=[vocab_size, embedding_size]),
                    trainable=False)

                self.embedding_placeholder = tf.placeholder(
                    tf.float32, [self.vocab_size, self.embedding_size])
                self.embedding_init = self.LT.assign(self.embedding_placeholder)

            with tf.variable_scope('sent'):
                self.sent_out_size = tf.convert_to_tensor(
                    sent_kmax * sent_nb_filter * len(sent_filter_sizes))
                fc_shape = [self.sent_out_size.eval(), sent_embed_size]
                self._create_sharable_weights(sent_filter_sizes, embedding_size,
                                              sent_nb_filter, fc_shape)

            with tf.variable_scope('doc'):
                self.doc_out_size = tf.convert_to_tensor(
                    doc_kmax * doc_nb_filter * len(doc_filter_sizes))
                if sent_embed_size is None:
                    sent_size = self.sent_out_size.eval()
                else:
                    sent_size = sent_embed_size
                fc_shape = [self.doc_out_size.eval(), doc_embed_size]
                self._create_sharable_weights(doc_filter_sizes, sent_size,
                                              doc_nb_filter, fc_shape)

        print('sent_out_size %s, doc_out_size %s' % 
            (self.sent_out_size.eval(), self.doc_out_size.eval()))
        print('sent_embed_size %s, doc_embed_size %s' % 
            (self.sent_embed_size, doc_embed_size))
            
    def inference(self, X):
        """ This is the forward calculation from batch X to doc embeddins """
        
        embedded_words = tf.nn.embedding_lookup(self.LT, X)
        embedded_words_expanded = tf.expand_dims(embedded_words, -1)
        
        with tf.variable_scope('sent'):

            def convolv_on_sents(embeds):
                add_fc = self.sent_embed_size is not None
                return self._convolv_on_embeddings(
                    embeds, self.sent_filter_sizes, self.sent_nb_filter,
                    self.sent_kmax, add_fc)

            # iter over each document
            self.sent_embed = tf.map_fn(
                convolv_on_sents,
                embedded_words_expanded,
                parallel_iterations=10,
                name='iter_over_docs')
            # sent_embed shape is [batch, n_sent, sent_sent_kmax*sent_nb_filter*len(sent_filter_sizes), 1]

        with tf.variable_scope('doc'):
            # finally, convolv on documents
            add_fc = self.doc_embed_size is not None
            self.doc_embed = self._convolv_on_embeddings(
                self.sent_embed, self.doc_filter_sizes, self.doc_nb_filter,
                self.doc_kmax, add_fc)
            # doc_embed shape is [batch, doc_kmax*doc_nb_filter*len(doc_filter_sizes), 1]
            
        doc_embed_normalized = tf.nn.l2_normalize(
            self.doc_embed, dim=1, name='doc_embed_normalized')

        return doc_embed_normalized

    def loss(self, X):
        with tf.name_scope("loss"):
            doc_embed_normalized = self.inference(X)
            if self.doc_embed_size is not None:
                doc_size = tf.convert_to_tensor(self.doc_embed_size)
            else:
                doc_size = self.doc_out_size
            self.anchor, self.positive, self.negative = tf.unstack(
                tf.reshape(doc_embed_normalized, [-1, 3, doc_size]),
                3, 1)
            _loss = triplet_loss(self.anchor, self.positive, self.negative)
        return _loss

    def optimize(self, X):
        with tf.name_scope("optimize"):
            self.loss_op = self.loss(X)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.gradients = self.optimizer.compute_gradients(self.loss_op)
                apply_gradient_op = self.optimizer.apply_gradients(
                    self.gradients, global_step=self.global_step)
        return apply_gradient_op

    def _convolv_on_embeddings(self, embeds, filter_sizes, nb_filter, kmax, add_fc):
        """
        Create a convolution + k-max pool layer for each filter size, then concat and vectorize.
        embeds shape is [batch, (n_words or n_sents), embedding_size, 1]
        """
        pooled_outputs = []
        for fsize in filter_sizes:
            with tf.name_scope("conv-%s" % fsize):
                with tf.variable_scope(
                        "conv_weights_fsize-%s" % fsize, reuse=True):
                    weights_init = tf.get_variable('W')
                    bias_init = tf.get_variable('b')
                conv = tf.nn.conv2d(
                    embeds,
                    weights_init,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                h = tf.nn.relu(tf.nn.bias_add(conv, bias_init), name="relu")
#                 h shape is [batch, n_words - fsize + 1, 1, nb_filter]
#             tf.summary.histogram("relu", h)

            with tf.name_scope('%s-maxpool-fsize-%s' % (kmax, fsize)):
                # k-maxpooling over the outputs
                trans = tf.transpose(h, perm=[0, 2, 3, 1])
                values, indices = tf.nn.top_k(trans, k=kmax, sorted=False)
                pooled = tf.transpose(values, perm=[0, 3, 1, 2])
                # pooled shape is [batch, kmax, 1, nb_filter]
                pooled_outputs.append(pooled)

        with tf.name_scope('concat_and_vectorize'):
            # Combine all the pooled features
            h_pool = tf.concat(pooled_outputs, 3)
            # h_pool shape is [batch, kmax, 1, nb_filter*len(filter_sizes)]

            # Vectorize filters for each sent to get sent embeddings
            trans = tf.transpose(h_pool, perm=[0, 2, 3, 1])
            batch = tf.shape(embeds)[0]
            layer = tf.reshape(trans, [batch, -1, 1])
            # layer shape is [batch, kmax*nb_filter*len(filter_sizes), 1]

        # with tf.variable_scope('batch_norm', reuse=True):
        #     layer = tf.nn.batch_normalization(layer, 
        #         center=True, scale=True, is_training=self.phase, scope='bn')

        if add_fc:
            with tf.variable_scope('fully_connected', reuse=True):
                layer = tf.matmul(tf.squeeze(layer), tf.get_variable('fc_W')) + \
                    tf.get_variable('fc_b')
            layer = tf.nn.relu(layer, name="relu")
            layer = tf.expand_dims(layer, 2)

        return layer

    def _create_sharable_weights(self, filter_sizes, embedding_size,
                                 nb_filter, fc_shape):
        """ Create sharable weights for each type of convolution """
        with tf.name_scope('sharable_weights'):
            for fsize in filter_sizes:
                with tf.variable_scope("conv_weights_fsize-%s" % fsize):
                    filter_shape = [fsize, embedding_size, 1, nb_filter]
                    initializer = tf.contrib.layers.xavier_initializer_conv2d(
                        uniform=True)
                    weights_init = tf.get_variable(
                        'W', filter_shape, initializer=initializer)
                    bias_init = tf.get_variable(
                        'b', initializer=tf.constant(0.1, shape=[nb_filter]))

            with tf.variable_scope('fully_connected'):
                if fc_shape[1] is not None:
                    initializer = tf.contrib.layers.xavier_initializer(uniform=True)
                    weights_init = tf.get_variable(
                        'fc_W', fc_shape, initializer=initializer)
                    bias_init = tf.get_variable(
                        'fc_b', shape=[fc_shape[1]], initializer=tf.zeros_initializer)

            # with tf.variable_scope('batch_norm'):
            #     1


    def init_lookup_table(self, word_embeddings):
        # Assign word embeddings to variable W
        self.sess.run(
            self.embedding_init,
            feed_dict={self.embedding_placeholder: word_embeddings})

    def init_summary(self, add_histograms=False):
        self.saver = tf.train.Saver()
        # Create summaries to visualize weights
        tf.summary.scalar("loss", self.loss_op)
        if add_histograms:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name.replace(':', '_'), var)
            # Summarize all gradients
            for grad, var in self.gradients:
                tf.summary.histogram(
                    var.name.replace(':', '_') + '/gradient', grad)
        self.merged_summary_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(
            join(DATA_FOLDER, 'summary', 'train',
                 str(datetime.datetime.now())), self.sess.graph)
    
    def add_summary(self, summary, current_step):
        self.train_writer.add_summary(summary, current_step)
        
    def save(self, global_step):
        self.saver.save(
            self.sess,
            join(DATA_FOLDER, 'models', '%s' % str(datetime.datetime.now())),
            global_step=global_step)

        
def triplet_loss(anchor_embed,
                 positive_embed,
                 negative_embed,
                 margin=0.2):
    """
    input: Three L2 normalized tensors of shape [None, dim], compute on a batch
    output: float
    """
    with tf.variable_scope('triplet_loss'):
        d_pos = tf.reduce_sum(tf.square(anchor_embed - positive_embed), 1)
        d_neg = tf.reduce_sum(tf.square(anchor_embed - negative_embed), 1)

        loss = tf.maximum(0., margin + d_pos - d_neg)
        loss = tf.reduce_mean(loss)

    return loss
    