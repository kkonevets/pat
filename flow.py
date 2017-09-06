import tensorflow as tf
from common import *


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
#         dynamic_pad=True,
        #         allow_smaller_final_batch=True,
        num_threads=num_threads)
    
    X = tf.reshape(
    tf.transpose([anchor, positive, negative],
                 [1, 0, 2, 3]), [-1, doc_shape[0], doc_shape[1]], name='X')
    return X, fnames