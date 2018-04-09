""" starter code for word2vec skip-gram model with NCE loss
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 04
"""


# No eager execution, variable sharing only with "name_scope"
# name_scope tells TF which nodes are similar to which nodes and should be grouped together

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

import utils
import word2vec_utils

# Model hyperparameters
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128            # dimension of the word embedding vectors
SKIP_WINDOW = 1             # the context window
NUM_SAMPLED = 64            # number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 3000        # number of tokens to visualize

# Why this is not a class but rather an independent method?
def word2vec(dataset):
    """ Build the graph for word2vec model and train it """
    # Step 1: get input, output from the dataset
    with tf.name_scope('data'): # group data related nodes together
        iterator = dataset.make_initializable_iterator() # dataset defined in the def main
        center_words, target_words = iterator.get_next()

    """ Step 2 + 3: define weights and embedding lookup.
    In word2vec, it's actually the weights that we care about 
    """
    with tf.name_scope('embed'): # group embedding nodes together
        embed_matrix = tf.get_variable('embed_matrix', 
                                        shape=[VOCAB_SIZE, EMBED_SIZE],
                                        initializer=tf.random_uniform_initializer())
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embedding')

    # Step 4: construct variables for NCE loss and define loss function
    with tf.name_scope('loss'): # group loss nodes together
        nce_weight = tf.get_variable('nce_weight', shape=[VOCAB_SIZE, EMBED_SIZE],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / (EMBED_SIZE ** 0.5)))
        nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))

        # define loss function to be NCE loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                            biases=nce_bias, 
                                            labels=target_words, 
                                            inputs=embed, 
                                            num_sampled=NUM_SAMPLED, 
                                            num_classes=VOCAB_SIZE), name='loss')

    # Step 5: define optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    
    utils.safe_mkdir('checkpoints') # Create a directory if there isn't one already. 

    with tf.Session() as sess:
        # step 1-2: initialize all variables and iterator
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())

        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('graphs/word2vec_simple', sess.graph)
        
        # execute inference model on the training data and compute loss, adjust the params using optimizer
        for index in range(NUM_TRAIN_STEPS):
            try:
                loss_batch, _ = sess.run([loss, optimizer])
                total_loss += loss_batch
                if (index + 1) % SKIP_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP)) # print avg loss
                    total_loss = 0.0
            except tf.errors.OutOfRangeError: # if tf.errors raises OOR error, reinitialize the iterator
                sess.run(iterator.initializer)
        writer.close()

def gen():
    yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE, 
                                        BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)
    """ yield from returns a sequence instead of a single output """

def main():
  """import data using tf.data and from_generator"""
    dataset = tf.data.Dataset.from_generator(gen, 
                                (tf.int32, tf.int32), 
                                (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    word2vec(dataset)

if __name__ == '__main__':
    main()
