import time
import math

import tensorflow as tf
from tqdm import tqdm

from ecg.models import Model

class Classifier(Model):

    # --------------------------------------------------------------------------
    def __init__(self, input_dim, n_classes, do_train, scope):

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.do_train = do_train

        with tf.variable_scope(scope):
            self.create_graph()
        if do_train:
            self.cost = self.create_cost_graph(self.targets, self.logits)
            self.create_summary(self.targets, self.logits)
            self.train = self.create_optimizer_graph(self.cost)
            self.train_writer, self.test_writer = self.create_summary_writers()
            self.merged = tf.summary.merge_all()

        self.sess = self.create_session()
        self.sess.run(tf.global_variables_initializer())
        self.stored_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        self.saver = tf.train.Saver(self.stored_vars, max_to_keep=1000)


    # --------------------------------------------------------------------------
    def create_graph(self):
        print('Creat graph')

        self.inputs,\
        self.targets,\
        self.weight_decay,\
        self.learn_rate,\
        self.is_training = self.input_graph() # inputs shape is # b*n_f x h1 x c1

        self.logits = self.dense_block(inputs=self.inputs, structure=[512, 128,
            self.n_classes])
        self.pred = tf.argmax(self.logits, axis=1)

        print('Done!')


    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')
        inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim],
            name='inputs')

        targets = tf.placeholder(tf.float32, shape=[None, self.n_classes],
            name='targets')

        weight_decay = tf.placeholder(tf.float32, name='weight_decay')

        learn_rate = tf.placeholder(tf.float32, name='learn_rate')

        is_training = tf.placeholder(tf.bool, name='is_training')

        return inputs, targets, weight_decay, learn_rate, is_training

    # --------------------------------------------------------------------------
    def dense_block(self, inputs, structure):
        print('\tdense_block')
        for layer in structure[:-1]:
            inputs = tf.layers.dense(inputs=inputs, units=layer, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            inputs = tf.contrib.layers.batch_norm(inputs=inputs, scale=True,
                updates_collections=None, is_training=self.is_training)
            inputs = tf.nn.relu(inputs)
        out = tf.layers.dense(inputs=inputs, units=structure[-1], activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        return out


    # --------------------------------------------------------------------------
    def create_cost_graph(self, targets, logits):
        print('create_cost_graph')
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=targets,
            logits=logits))
        self.L2_loss = self.weight_decay*sum([tf.reduce_mean(tf.square(var))
            for var in tf.trainable_variables()])
        return self.cross_entropy + self.L2_loss


    # --------------------------------------------------------------------------
    def create_summary(self, targets, logits):
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        tf.summary.scalar('L2 loss', self.L2_loss)
        pred = tf.reduce_max(logits, axis=1)
        pred = tf.cast(tf.equal(logits, tf.expand_dims(pred, 1)), tf.float32)
        for i in range(self.n_classes):
            y = targets[:,i]
            y_ = pred[:,i]
            tp = tf.reduce_sum(y*y_)
            tn = tf.reduce_sum((1-y)*(1-y_))
            fp = tf.reduce_sum((1-y)*y_)
            fn = tf.reduce_sum(y*(1-y_))
            pr = tp/(tp+fp+1e-5)
            re = tp/(tp+fn+1e-5)
            f1 = 2*pr*re/(pr+re+1e-5)
            with tf.name_scope('Class_{}'.format(i)):
                tf.summary.scalar('Class {} precision'.format(i), pr)
                tf.summary.scalar('Class {} recall'.format(i), re)
                tf.summary.scalar('Class {} f1 score'.format(i), f1)


    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, cost):
        print('create_optimizer_graph')
        with tf.variable_scope('optimizer_graph'):
            optimizer = tf.train.AdamOptimizer(self.learn_rate)
            train = optimizer.minimize(cost)
        return train


    # --------------------------------------------------------------------------
    def train_(self, data_loader, batch_size, weight_decay,  learn_rate_start,
        learn_rate_end, n_iter, save_model_every_n_iter, path_to_model):
        """
        Args:
            noise_range: list, first item - std_start, second item - std_end
        """
        print('\n\t----==== Training ====----')
            
        start_time = time.time()
        for current_iter in tqdm(range(n_iter)):
            learn_rate = self.scaled_exp_decay(learn_rate_start, learn_rate_end,
                n_iter, current_iter)

            #evaluate
            batch = data_loader.test.next_batch(batch_size)
            feedDict = {self.inputs : batch[0],
                        self.targets : batch[1],
                        self.weight_decay : weight_decay,
                        self.learn_rate : learn_rate,
                        self.is_training : False}
            _, summary = self.sess.run([self.cost, self.merged], feed_dict=feedDict)
            self.test_writer.add_summary(summary, current_iter)


            #train
            batch = data_loader.train.next_batch(batch_size)
            feedDict[self.inputs] = batch[0]
            feedDict[self.targets] = batch[1]
            feedDict[self.is_training] = True
            _, summary = self.sess.run([self.train, self.merged], feed_dict=feedDict)
            self.train_writer.add_summary(summary, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        print('\nTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))


################################################################################
# TESTING
if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    ae = Classifier(input_dim=784, n_classes=10, do_train=True, scope='classifier')
    # try:
    #     ae.load_model(path='models/', sess=ae.sess)
    # except FileNotFoundError:
    #     pass
    ae.train_(data_loader=mnist, batch_size=256, weight_decay=1e-2,
        learn_rate_start=1e-2, learn_rate_end=1e-4, n_iter=1000,
        save_model_every_n_iter=10000, path_to_model='models/cl')




