""" Examples to demonstrate variable sharing
CS 20: 'TensorFlow for Deep Learning Research'
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 05
"""

## name_scope recall
### TensorBoard doesnt know which nodes are similar to which nodes and should be grouped together.
### we tell TF which nodes are similar and to be grouped together

## variable_scope
###  variable_scope also create "namespace" and it is to facilitate "variable sharing"

# ????? Bunlarda neden tf.Session yok? Sadece build ettik, execute etmedik diye. Loss/optimizer yok. 


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

x1 = tf.truncated_normal([200, 100], name='x1')
x2 = tf.truncated_normal([200, 100], name='x2')

def two_hidden_layers(x):
    assert x.shape.as_list() == [200, 100]
    w1 = tf.Variable(tf.random_normal([100, 50]), name='h1_weights')
    b1 = tf.Variable(tf.zeros([50]), name='h1_biases')
    h1 = tf.matmul(x, w1) + b1
    assert h1.shape.as_list() == [200, 50]  
    w2 = tf.Variable(tf.random_normal([50, 10]), name='h2_weights')
    b2 = tf.Variable(tf.zeros([10]), name='2_biases')
    logits = tf.matmul(h1, w2) + b2
    return logits

# logits1 = two_hidden_layers(x1)
# logits2 = two_hidden_layers(x2)

# Each time you call two network (h1, h2), TF creates a different set of variables, while in fact,
    # you want network to share the same variables for all inputs: x1, x3, x3...

# =============================================================================
# We need some changes!
#    we first need to use tf.get_variable(). When we create a variable with tf.get_variable()
    # it first checks whether that variable exists.
# =============================================================================

def two_hidden_layers_2(x):
    assert x.shape.as_list() == [200, 100]
    w1 = tf.get_variable('h1_weights', [100, 50], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable('h1_biases', [50], initializer=tf.constant_initializer(0.0))
    h1 = tf.matmul(x, w1) + b1
    assert h1.shape.as_list() == [200, 50]  
    w2 = tf.get_variable('h2_weights', [50, 10], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable('h2_biases', [10], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(h1, w2) + b2
    return logits

# This will return ValueError: Variable h1_weights already exists, disallowed.
    # To avoid this, we need to put all variables we want to use in a VarScope, and set that VarScope to be reusable.


# logits1 = two_hidden_layers_2(x1)
# logits2 = two_hidden_layers_2(x2)

# with tf.variable_scope('two_layers') as scope:
#     logits1 = two_hidden_layers_2(x1)
#     scope.reuse_variables()                        # here we set VarScope to be reusable while calling the function.
#     logits2 = two_hidden_layers_2(x2)

# with tf.variable_scope('two_layers') as scope:
#     logits1 = two_hidden_layers_2(x1)
#     scope.reuse_variables()
#     logits2 = two_hidden_layers_2(x2)

def fully_connected(x, output_dim, scope): # more elegantly, set VarScope to be reusable when building the model.
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope: # scope name comes from the calling function
        w = tf.get_variable('weights', [x.shape[1], output_dim], initializer=tf.random_normal_initializer())
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b

def two_hidden_layers(x):
    h1 = fully_connected(x, 50, 'h1')
    h2 = fully_connected(h1, 10, 'h2')

with tf.variable_scope('two_layers') as scope:
    logits1 = two_hidden_layers(x1)
    # scope.reuse_variables()
    logits2 = two_hidden_layers(x2)

writer = tf.summary.FileWriter('./graphs/cool_variables', tf.get_default_graph())
writer.close()