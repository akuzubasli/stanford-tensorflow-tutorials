""" Using convolutional net on MNIST dataset of handwritten digits
MNIST dataset: http://yann.lecun.com/exdb/mnist/
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 07
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 

import tensorflow as tf

import utils

def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    '''
    A method that does convolution + relu on inputs
    '''
    #############################
    ########## TO DO ############
    #############################
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope: # open a scope, with a name scope_name which is reusable
        in_channel = inputs.shape[-1]
        kernel = tf.get_variable("kernel", shape=[k_size, k_size, in_channel, filters],
                                 initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable("bias", shape=[filters], initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(input=inputs,
                              filter=kernel, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(conv + bias, name=scope_name)

def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    '''A method that does max pooling on inputs'''
    #############################
    ########## TO DO ############
    #############################
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)
    return pool

def fully_connected(inputs, out_dim, scope_name='fc'):
    '''
    A fully connected linear layer on inputs
    '''
    #############################
    ########## TO DO ############
    #############################
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights", shape=[inputs.shape[1], out_dim],
                                 initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable("bias", shape=[out_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(inputs, weights) + bias

class ConvNet(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                trainable=False, name='global_step')
        self.num_classes = 10
        self.skip_step = 20
        self.num_test = 10000
        self.training = True

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_mnist_dataset(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                                   train_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            # reshape the image to make it work with tf.nn.conv2d

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)    # initializer for train_data

    def inference(self):
        '''
        Build the model according to the description we've shown in class
        '''
        #############################
        ########## TO DO ############
        #############################
        conv1 = conv_relu(inputs=self.img, filters=32, k_size=5, stride=1, padding="SAME", scope_name="conv1")
        pool1 = maxpool(inputs=conv1, ksize=2, stride=2, padding="SAME")
        conv2 = conv_relu(pool1, filters=64, k_size=5, stride=1, padding="SAME", scope_name="conv2")
        pool2 = maxpool(inputs=conv2, ksize=2, stride=2, padding="SAME", scope_name="pool2")
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc1_relu = tf.nn.relu(fully_connected(pool2, 1024, scope_name="fc1_relu"))
        drop_fc1 = tf.nn.dropout(fc1_relu, keep_prob=self.keep_prob, name="drop_fc1")
        #drop_fc1 = tf.layers.dropout(fc1_relu, rate=self.keep_prob, training=self.training, name="layers_dropout")
        self.logits = fully_connected(drop_fc1, self.num_classes, scope_name="logits")

    def loss(self):
        '''
        define loss function
        use softmax cross entropy with logits as the loss function
        tf.nn.softmax_cross_entropy_with_logits
        softmax is applied internally
        don't forget to compute mean cross all sample in a batch
        '''
        #############################
        ########## TO DO ############
        #############################
        with tf.name_scope("loss"):
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name="loss")
    
    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        Don't forget to use global step
        '''
        #############################
        ########## TO DO ############
        #############################
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss,
                                                                          global_step=self.gstep)

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        Remember to track both training loss and test accuracy
        '''
        #############################
        ########## TO DO ############
        #############################
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.histogram("histogram_loss", self.loss)
            # because we have several summaries, we want to merge them into one top summary for easy management
            self.summary_op = tf.summary.merge_all()
        
    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/convnet_starter/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.num_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_starter')
        writer = tf.summary.FileWriter('./graphs/convnet_starter', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_starter/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=15)