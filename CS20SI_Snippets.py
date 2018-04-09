# CS20SI Snippets #

# This could speed up your CPU. You may get an error if not executed.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
#%%
# Build the graph
a = tf.constant(2, name = "a")
b = tf.constant(3, name = "b")
x = tf.add(a, b)

# Use writer to save summary after graph before sess
writer = tf.summary.FileWriter("./graphs", tf.get_default_graph())

with tf.Session() as sess:
  # writer = tf.summary.FileWriter("./graphs", sess.graph) - above better
  print(sess.run(x))
  
  
writer.close() # close writer when you are done

# TENSORBOARD
# On terminal 
# $ python [your_program].py
# $ tensorboard --logdir="./graphs" --port 6006 (default)

tf.zeros([2,3], tf.int32) # faster than tf.constant with 0s

# input_tensor is [[0,1], [2,3], [4,5]]
# tf.zeros_like(input_tensor) / or ones

tf.fill([2,3], 8) # 2,3 tensor full of 8s

# Sequences
tf.lin_space(10., 13., 4) # [10. 11. 12. 13.] start stop num_steps (not step size!) 
tf.range(3, 18, 3) # [3 6 9 12 15] # start stop step_size 
tf.range(5) # 
#%%
# TENSOR OBJECTS ARE NOT ITERABLE! Not the same as numPy
# for _ in tf.range(4): # TypeError
  
# Randomly generated constants
tf.random_normal
tf.truncated_normal # used more, it doesn't create outliers (tail)
tf.random_uniform
tf.random_shuffle
tf.random_crop
tf.multinomial
tf.random_gamma
  
#tf.set_random_seed(seed)
#%%

# Wizard of Div
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
with tf.Session() as sess:
	print(sess.run(tf.div(b, a)))           #  ⇒ [[0 0] [1 1]]
	print(sess.run(tf.divide(b, a)))        #  ⇒ [[0. 0.5] [1. 1.5]]
	print(sess.run(tf.truediv(b, a)))       #  ⇒ [[0. 0.5] [1. 1.5]]
	print(sess.run(tf.floordiv(b, a)))      #  ⇒ [[0 0] [1 1]]
	#print(sess.run(tf.realdiv(b, a)))       #  ⇒ # Error: only works for real values
	print(sess.run(tf.truncatediv(b, a)))   #  ⇒ [[0 0] [1 1]]
	print(sess.run(tf.floor_div(b, a)))     #  ⇒ [[0 0] [1 1]]


# TF Data Types
t_0 = 19 			         			# scalars are treated like 0-d tensors
tf.zeros_like(t_0)                  			# ==> 0
tf.ones_like(t_0)                    			# ==> 1

t_1 = [b"apple", b"peach", b"grape"] 	# 1-d arrays are treated like 1-d tensors
tf.zeros_like(t_1)                   			# ==> [b'' b'' b'']
tf.ones_like(t_1)                    			# ==> TypeError: Expected string, got 1 of type 'int' instead.

t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]]         		# 2-d arrays are treated like 2-d tensors

tf.zeros_like(t_2)                   			# ==> 3x3 tensor, all elements are False
tf.ones_like(t_2)                    			# ==> 3x3 tensor, all elements are True

# Single values will be converted to 0-d tensors (or scalars), 
# lists of values will be converted to 1-d tensors (vectors), 
# lists of lists of values will be converted to 2-d tensors (matrices), and so on.

#%%

# WHAT IS WRONG WITH CONSTANTS?
# They are stored in the graph which makes the graph big and their loading expensive.
# So ONLY USE constanst for primitive types (?)
# USE VARIABLES or readers for more data that requires more memory

# create variables with tf.Variable
s = tf.Variable(2, name="scalar")
m = tf.Variable([[0,1], [2,3]], name="matrix")
W = tf.Variable(tf.zeros([784,10])) # we use [] for stating the matrix dims (np: ())

# create variables with tf.get_variable
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0,1], [2,3]]))
W = tf.get_variable("big_matrix", shape=([784,19]), initializer=tf.zeros_initializer()) # zero initialization of big matrix

# tf.constant is an op (operation) but tf.Variable is a class with many ops
x = tf.Variable()

x.initializer # init op
x.value() # read op as an obj
x.eval() # evaluate op .. use for print
x.assign(...) # write op
x.assign_add(...) # and more... 

# you have to initialize your variables
# the easiest way
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer()) # initializer is an op, not stored in the graph. you need to execute it within the context of a session. 
  sess.run(tf.variables_initializer([a, b])) # initializes only the vars a, b
  sess.run(W.initializer) # init only W

#%%

W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
  sess.run(W.initializer)
  print(W) # does not works because W is an object. call its value
  print(W.value()) # does not works it needs to be evaluated first
  print(W.eval()) # here you get your variable

W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
  sess.run(W.initializer)
  print(W.eval()) # prints 10 why? because assign is an op which needs to be executed is a session.

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
  sess.run(W.initializer)
  sess.run(assign_op) 
  print(W.eval()) # prints 100

my_var = tf.Variable(10)
with tf.Session() as sess:
  sess.run(my_var.initializer)
  # increment by 10 
  sess.run(my_var.assign_add(10)) # >> 20
  # decrement by 2 
  sess.run(my_var.assign_sub(2)) # >> 18
  print(my_var.eval())

#%%

# Each session maintains its own copy of vars
W = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10))) 		# >> 20
print(sess2.run(W.assign_sub(2))) 		# >> 8

print(sess1.run(W.assign_add(100))) 		# >> 120
print(sess2.run(W.assign_sub(50))) 		# >> -42

sess1.close()
sess2.close()

#%%

# CONTROL DEPENDENCIES
# tf.Graph.control_dependencies(control_inputs) defines which ops should run first

# your graph g has 5 ops: a, b, c, d, e
g = tf.get_default_graph()
with g.control_dependencies([a, b, c]):
  # d and e will only run after a, b, c have executed. 
  d = ...
  e = ...
  
#%%

# PLACEHOLDER & FEED_DICT 
# Assemble the graph first without knowing the values needed for compt
# Analogy: f(x, y) = 2*x + y without knowing x and y but they are "placeholders" for the actual values.
  
# tf.placeholder(dtype, shape=None, name=None)
# create a placeholder for a vector of 3 elements, type tf.float32
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)

# use placeholder as you would a constant or a variable
c = a + b # short for tf.add(a,b)

with tf.Session() as sess:
  print(sess.run(c)) # InvalidArgumentError: You must feed a value for placeholder tensor. # what is a? it is a placeholder for what?

# Supplement the values to placeholders using a dictionary "feed_dict" 
with tf.Session() as sess:
  print(sess.run(c, feed_dict={a: [1, 2, 3]})) # feed [1, 2, 3] to a >> [6. 7. 8.]
  
# tf.placeholder shape=None means that tensor of any shape will be accepted as value for placeholder
# shape=None is easy to construct graphs, but it makes harder to debug
  
# Multiple values to feed in: use for loop
with tf.Session() as sess:
  for a_value in list_of_values_for_a:
    print(sess.run(c, {a: a_value}))

# You can feed_dict any feedable tensor. Placeholder is just a way to indicate that something must be fed.
tf.get_default_graph().is_feedable(tensor=a) # check if it is feedable. # No need to sess as it is about the type, not eval. 

# Feeding values to TF ops
# create operations, tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
  # compute the value of b given a is 15
  sess.run(b, feed_dict={a: 15}) 				# >> 45
  print(b.eval())   # prints 21 why? because it tries to eval b, it needs a (not a placeholder), gets it from above.
  print(sess.run(b, feed_dict={a: 15})) # prints 45 as a is fed with 15. 

#%%

# LAZY LOADING
# Compare the graphs (1) and (2). Which one is "Normal" which one is "Lazy? Why?

# (1) 
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y) 		# create the node before executing the graph

writer = tf.summary.FileWriter('./graphs/normal_loading', tf.get_default_graph())
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for _ in range(10):
    sess.run(z)
writer.close()
  

# (2)
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')

writer = tf.summary.FileWriter('./graphs/normal_loading', tf.get_default_graph())
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for _ in range(10):
    sess.run(tf.add(x, y)) # someone decides to be clever to save one line of code
writer.close()

# (1) is Normal and (2) is Lazy. Because (2) defers to create until it's needed and 
# ...creates a new node at each time (x10 "Add" node). Imagine if thousands/millions times.
# Your graph gets bloated. Slow to load and Expensive to pass around. 
# One of the most common TF non-bug bugs.

# Solution
# 1. Separate definition of ops from computing/running ops 
# 2. Use Python property* (decorator) to ensure function is also loaded once the first time it is called*

#%%
## Lecture 3 
### Review

# TensorFlow separates -definition of computations- from their -executions-
# Phase 1 - Assemble a graph
# Phase 2 - Use a session to exec ops in the graph

# Lecture 4 - Eager Execution

#%%

import time

def measure(x):
  # The very first time a GPU is used by TensorFlow, it is initialized.
  # So exclude the first run from timing.
  tf.matmul(x, x)

  start = time.time()
  for i in range(10):
    tf.matmul(x, x)
  end = time.time()

  return "Took %s seconds to multiply a %s matrix by itself 10 times" % (end - start, x.shape)

# Run on CPU:
with tf.device("/cpu:0"):
  print("CPU: %s" % measure(tf.random_normal([1000, 1000])))

# If a GPU is available, run on GPU:
if tfe.num_gpus() > 0:
  with tf.device("/gpu:0"):
    print("GPU: %s" % measure(tf.random_normal([1000, 1000])))

#%% Variable Sharing

"variable_scope'ta as scope var, name_scope'ta as scope yok"
    
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
    

# Graph Collections
"""As you create a model, you might put your variables to different parts of the graph. 
Sometimes, you’d want an easy way to access them. tf.get_collection lets you access a certain collection of variables, 
with key being the name of the collection, scope is the scope of the variables."""


tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_scope')  # collects all global variables in my_scope
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="my_scope")  # collects all trainable variables -


"""You can have collections of ops that aren’t variables. 
And yes, you can create your own collections with tf.add_to_collection(name, value). 
For example, you can create a collection of initializers and  add all init ops to that. """

"""For example, the tf.train.Optimizer subclasses default to optimizing the variables collected 
under tf.GraphKeys.TRAINABLE_VARIABLES if none is specified, 
but it is also possible to pass an explicit list of variables. 
"""

# Randomization
# Example 1: session keeps track of the random state
c = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(c)) # >> 3.574932
    print(sess.run(c)) # >> -5.9731865

# Example 2: each new session will start the random state all over again.
c = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(c)) # >> 3.574932

with tf.Session() as sess:
    print(sess.run(c)) # >> 3.574932

# Example 3: with operation level random seed, each op keeps its own seed.
c = tf.random_uniform([], -10, 10, seed=2)
d = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(c)) # >> 3.574932
    print(sess.run(d)) # >> 3.574932

# Example 4: graph level random seed
tf.set_random_seed(2)
c = tf.random_uniform([], -10, 10)
d = tf.random_uniform([], -10, 10)

with tf.Session() as sess:
    print(sess.run(c)) # >> 9.123926
    print(sess.run(d)) # >> -4.5340395


# Manage Experiments
# tf.train.Saver()
"""For example, if we want to save the variables of the graph after every 1000 training steps, we do the following"""

# define model

# create a saver object
saver = tf.train.Saver()

"""tf.train.Saver.save(
    sess,
    save_path,
    global_step=None,
    latest_filename=None,
    meta_graph_suffix='meta',
    write_meta_graph=True,
    write_state=True
)
"""

# launch a session to execute the computation
with tf.Session() as sess:
    # actual training loop
    for step in range(training_steps):
        sess.run([optimizer])
        if (step + 1) % 1000 == 0:
            saver.save(sess, 'checkpoint_directory/model_name', global_step=global_step)


"""In TensorFlow lingo, the step at which you save your graph’s variables is called a checkpoint. 
Since we will be creating many checkpoints, it’s helpful to append the number of training steps 
our model has gone through a variable called global_step. It’s a variable you’d see in many TensorFlow programs. 
We first need to create it, initialize it to 0 and set it to be not trainable, 
since we don’t want TensorFlow to optimize it."""
global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_step")

"""We need to pass global_step as a parameter to the optimizer 
so it knows to increment global_step by one with each training step."""
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)

"""To restore the variables, we use tf.train.Saver.restore(sess, save_path). 
For example, to restore the checkpoint at the 10,000th step."""
saver.restore(sess, 'checkpoints/skip-gram-10000')

"""Check if a checkpoint and its path is valid using:"""
ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
if ckpt and ckpt.model_checkpoint_path:
     saver.restore(sess, ckpt.model_checkpoint_path)

# Example tf.train.Saver()

saver = tf.train.Saver()  # define the saver

initial_step = 0  # ??
utils.safe_mkdir('checkpoints') # create dir if not exist

with tf.Session() as sess: # create session
  sess.run(self.iterator.initializer) # initialize the iterator for pumping the data
  sess.run(tf.global_variables_initializer())  # initialize all the variables

  # if a checkpoint exists, restore from the latest checkpoint
  ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)   # create ckpt if both exists and valid

  writer = tf.summary.FileWriter('graphs/word2vec' + str(self.lr), sess.graph)

  for index in range(num_train_steps):
    try:
      sess.run(self.optimizer)
      # save the model every 1000 steps
      if (index + 1) % 1000 == 0:
        saver.save(sess, 'checkpoints/skip-gram', index)
    except tf.errors.OutOfRangeError:   # raise error
      sess.run(self.iterator.initializer)

  writer.close()


"""Note that savers only save variables, not the entire graph, so we still have to create the graph ourselves, 
and then load in variables. The checkpoints specify the way to map from variable names to tensors.

What people usually do is not just save the parameters from the last iteration, 
but also save the parameters that give the best result so far 
so that you can evaluate your model on the best parameters so far.
"""


# tf.summary

"""We’ve been using matplotlib to visualize our losses and accuracy, which is unnecessary 
because TensorBoard provides us with a great set of tools to visualize our summary statistics during our training. 
Some popular statistics to visualize is loss, average loss, accuracy. 
You can visualize them as scalar plots, histograms, or even images. 

So we have a new name_scope in our graph to hold all the summary ops.
"""

def _create_summary(self):
  with tf.name_scope("summaries"):
    tf.summary.scalar("loss", self.loss)
    tf.summary.scalar("accuracy", self.accuracy)
    tf.summary.histogram("historgram loss", self.loss)
    # because we have several summaries, we want to merge them into one top summary for easy management
    self.summary_op = tf.summary.merge_all()  # it is an op!

# because it is an op, you have to execute it with sess.run()
""" To create an Operation, you call its constructor in Python,
which takes in whatever Tensor parameters needed for its calculation, 
known as inputs, as well as any additional information needed to properly create the Op, known as attributes. 
The Python constructor returns a handle to the Operation’s output (zero or more Tensor objects), 
and it is this output which can be passed on to other Operations or Session.run"""

loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)

"""Now you’ve obtained the summary, you need to write the summary to file 
using the same FileWriter object we created to visualize our graph."""

writer.add_summary(summary, global_step=step)

"""You can visualize the statistics as images using tf.summary.image"""
tf.summary.image(name, tensor, max_outputs=3, collections=None)



## tf.gradients()
"""TensorFlow can take gradients for us, but it cant give us intuition about what functions to use. 
It doesnt tell us if a function will suffer from exploding or vanishing gradients. 
We still need to know about gradients to get an understanding of why a model works while another doesnt.
"""

"""tf.gradients(ys, xs, grad_ys=None, name='gradients', 
                colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)"""

# example gradients
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.Variable(2.0)
y = 2.0 * (x ** 3)

grad_y = tf.gradients(ys=y, xs=x)
with tf.Session() as sess:
    sess.run(x.initializer)
    print(sess.run(grad_y))


# multiple gradient example
x = tf.Variable(2.0)
y = 2.0 * (x ** 3)
z = 3.0 + y ** 2

grad_z = tf.gradients(z, [x, y])
with tf.Session() as sess:
    sess.run(x.initializer)
    print(sess.run(grad_z)) # >> [768.0, 32.0]
# 768 is the gradient of z with respect to x, 32 with respect to y
















