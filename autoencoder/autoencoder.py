import time
import math

import tensorflow as tf
from tqdm import tqdm

from model_abstract.model_abstract import Model

class AE(Model):

    # --------------------------------------------------------------------------
    def __init__(self, input_dim, z_dim, do_train, scope):

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.do_train = do_train
        self.scope = scope

        with tf.variable_scope(scope):
            self.create_graph()
        if do_train:
            self.cost = self.create_cost_graph(self.inputs, self.recover)
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
        self.weight_decay,\
        self.learn_rate,\
        self.noise_value,\
        self.is_training = self.input_graph() # inputs shape is # b*n_f x h1 x c1

        self.z = self.encoder(inputs=self.inputs, structure=[800, 400, self.z_dim])

        z_noised = self.z + tf.random_normal(shape=tf.shape(self.z),
            stddev=self.noise_value)
        z_noised = tf.reshape(z_noised,[-1, self.z_dim])
        z_noised = self.z

        self.recover = self.decoder(inputs=z_noised, structure=[400,800, 1000,
            self.input_dim])

        print('Done!')


    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')
        inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim],
            name='inputs')

        weight_decay = tf.placeholder(tf.float32, name='weight_decay')

        learn_rate = tf.placeholder(tf.float32, name='learn_rate')

        noise_value = tf.placeholder(tf.float32, name='noise_value')

        is_training = tf.placeholder(tf.bool, name='is_training')

        return inputs, weight_decay, learn_rate, noise_value, is_training

    # --------------------------------------------------------------------------
    def encoder(self, inputs, structure):
        print('\tencoder')
        for layer in structure[:-1]:
            inputs = tf.layers.dense(inputs=inputs, units=layer, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            inputs = tf.contrib.layers.batch_norm(inputs=inputs, scale=True,
                updates_collections=None, is_training=self.is_training)
            inputs = tf.nn.relu(inputs)
        out = tf.layers.dense(inputs=inputs, units=structure[-1], activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        out = out/tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(out), 1)), 1) #normalization to 1
        return out


    # --------------------------------------------------------------------------
    def decoder(self, inputs, structure):
        print('\tdecoder')
        for layer in structure[:-1]:
            inputs = tf.layers.dense(inputs=inputs, units=layer, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            inputs = tf.contrib.layers.batch_norm(inputs=inputs, scale=True,
                updates_collections=None, is_training=self.is_training)
            inputs = tf.nn.elu(inputs)
        out = tf.layers.dense(inputs=inputs, units=structure[-1], activation=tf.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        return out


    # --------------------------------------------------------------------------
    def create_cost_graph(self, original, recovered):
        print('create_cost_graph')
        self.mse = tf.reduce_mean(tf.square(original - recovered))
        self.L2_loss = self.weight_decay*sum([tf.reduce_mean(tf.square(var))
            for var in tf.trainable_variables()])
        tf.summary.scalar('MSE', self.mse)
        tf.summary.scalar('L2 loss', self.L2_loss)
        tf.summary.scalar('noise_value', self.noise_value)
        tf.summary.scalar('learn_rate', self.learn_rate)
        images = tf.reshape(recovered, [-1, 28, 28, 1])
        tf.summary.image('recovered img', images, max_outputs=12)
        return self.mse + self.L2_loss


    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, cost):
        print('create_optimizer_graph')
        with tf.variable_scope('optimizer_graph'+self.scope):
            optimizer = tf.train.AdamOptimizer(self.learn_rate)
            train = optimizer.minimize(cost)
        return train


    # --------------------------------------------------------------------------
    def train_(self, data_loader, batch_size, weight_decay,  learn_rate_start,
        learn_rate_end, n_iter, noise_range, save_model_every_n_iter, path_to_model):
        #depricated
        """
        Args:
            noise_range: list, first item - std_start, second item - std_end
        """
        print('\n\t----==== Training ====----')
            
        start_time = time.time()
        for current_iter in tqdm(range(n_iter)):
            learn_rate = self.scaled_exp_decay(learn_rate_start, learn_rate_end,
                n_iter, current_iter)
            noise_value = self.scaled_exp_decay(noise_range[0], noise_range[1],
                n_iter, current_iter)

            #evaluate
            batch = data_loader.test.next_batch(batch_size)
            feedDict = {self.inputs : batch[0],
                        self.weight_decay : weight_decay,
                        self.learn_rate : learn_rate,
                        self.noise_value : noise_value,
                        self.is_training : False}
            _, summary = self.sess.run([self.cost, self.merged], feed_dict=feedDict)
            self.test_writer.add_summary(summary, current_iter)


            #train
            batch = data_loader.train.next_batch(batch_size)
            feedDict[self.inputs] = batch[0]
            feedDict[self.is_training] = True
            _, summary = self.sess.run([self.train, self.merged], feed_dict=feedDict)
            self.train_writer.add_summary(summary, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        print('\nTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))


    # --------------------------------------------------------------------------
    def train_step(self, inputs, weight_decay, learn_rate, noise_value, is_training):
        feedDict = {self.inputs : inputs,
            self.weight_decay : weight_decay,
            self.learn_rate : learn_rate,
            self.noise_value : noise_value,
            self.is_training : is_training}
        self.sess.run(self.train, feed_dict=feedDict)


    # --------------------------------------------------------------------------
    def save_summaries(self, inputs, weight_decay, learn_rate, noise_value, is_training, it):
        feedDict = {self.inputs : inputs,
            self.weight_decay : weight_decay,
            self.learn_rate : learn_rate,
            self.noise_value : noise_value,
            self.is_training : is_training}
        summary = self.sess.run(self.merged, feed_dict=feedDict)
        self.train_writer.add_summary(summary, it)




def test_autoencoder():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    ae = AE(input_dim=784, z_dim=100, do_train=True, scope='autoencoder')
    # try:
    #     ae.load_model(path='models/', sess=ae.sess)
    # except FileNotFoundError:
    #     pass
    ae.train_(data_loader=mnist, batch_size=256, weight_decay=1e-2,
        learn_rate_start=1e-2, learn_rate_end=1e-4, n_iter=50000, noise_range=[0.4, 1e-3],
        save_model_every_n_iter=10000, path_to_model='models/ae')
    
################################################################################
# TESTING
if __name__ == '__main__':
    test_autoencoder()




