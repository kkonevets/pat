import tensorflow as tf
from common import *
import random


def parse_csv(text, doc_shape):
    with tf.name_scope('parse_csv'):
        strings = tf.string_split([text], delimiter='\n')
        raw_nums = tf.string_split(strings.values)
        nums = tf.string_to_number(raw_nums.values, tf.int32)
        dense = tf.sparse_to_dense(
            raw_nums.indices, raw_nums.dense_shape, nums, default_value=0)
        shape = tf.shape(dense)
        dense = tf.pad(dense, [[0,doc_shape[0]-shape[0]], [0,doc_shape[1]-shape[1]]], 'CONSTANT')
        dense.set_shape(doc_shape)
    return dense


def read_input_tuple(filename_queue, doc_shape):
    with tf.name_scope('read_input_tuple'):
        fnames = filename_queue.dequeue()
        example = []
        for fn in tf.unstack(fnames):
            record_string = tf.read_file(fn)
            arr = parse_csv(record_string, doc_shape)
            example.append(arr)
        example.append(fnames)
    return example


def input_pipeline(triples, doc_shape, batch_size, num_epochs=1, num_threads=cpu_count, shuffle=True, seed=0):
    filename_queue = tf.train.input_producer(
        triples, num_epochs=num_epochs, capacity=32, shuffle=shuffle, seed=seed)
    example = read_input_tuple(filename_queue, doc_shape)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    anchor, positive, negative, fnames = tf.train.batch(
        example,
        batch_size=batch_size,
        capacity=capacity,
        # allow_smaller_final_batch=True,
        num_threads=num_threads)
    
    X = tf.reshape(
    tf.transpose([anchor, positive, negative],
                 [1, 0, 2, 3]), [-1, doc_shape[0], doc_shape[1]], name='X')
    return X, fnames


def nameit(_id, with_path=True):
    if with_path:
        return join(DATA_FOLDER, 'corpus/%s.txt' % _id)
    else:
        return _id


def random_triples(sims, ids, num_epochs=1, with_path=True, seed=0):
    """
    Get random triples, select negatives at random in each epoch.
    Output: [anchor, positive, negative]
    """
    random.seed(0)
    ixs = list(range(len(ids)))
    for ep in range(num_epochs):
        random.shuffle(ixs)
        it = iter(ixs)
        for k, v in tqdm(sims.items()):
            exclude = [nameit(i, with_path) for i in [k] + v]
            for vi in v:
                ix = next(it)
                _neg = ids[ix]
                while _neg in exclude:
                    ix = next(it)
                    _neg = ids[ix]
                if not with_path:
                    _neg = basename(_neg).split('.')[0]
                yield [nameit(k, with_path), nameit(vi, with_path), _neg]


def gen_batches(embeds, triples, batch_size):
    """ Divide input data into batches.

    :param data: input data
    :param batch_size: size of each batch

    :return: data divided into batches
    """
    triples = np.array(triples)

    for i in range(0, triples.shape[0], batch_size):
        yield embeds.loc[triples[i:i+batch_size].reshape(-1), :]