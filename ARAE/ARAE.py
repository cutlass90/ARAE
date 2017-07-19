import time
from tqdm import tqdm

import tensorflow as tf
import numpy as np

from WGAN.WGAN import WGAN, test_WGAN
from autoencoder.autoencoder import AE
from classifier.classifier import test_classifier
from model_abstract.model_abstract import Model

class ARAE(Model):

    # --------------------------------------------------------------------------
    def __init__(self, input_dim, z_dim, c_dim, do_train):

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.do_train = do_train

        self.ae = AE(input_dim=self.input_dim, z_dim=self.c_dim, do_train=True,
            scope='autoencoder')
        self.wgan = WGAN(do_train=True, input_dim=self.c_dim, z_dim=self.z_dim,
            scope='WGAN', inputs=self.ae.z)
        self.sess = self.create_session()
        self.wgan.sess = self.sess
        self.ae.sess = self.sess
        self.ae.sess.run(tf.global_variables_initializer())
        self.wgan.sess.run(tf.global_variables_initializer())


    # --------------------------------------------------------------------------
    def train_model(self, data_loader, batch_size, weight_decay, n_iter, noise_range,
        learn_rate_ae, keep_prob, learn_rate_gen, learn_rate_disc, n_critic):
        """
        Args:
            noise_range: list, first item - std_start, second item - std_end
        """
        print('\n\t----==== Training ====----')
        start_time = time.time()
        for current_iter in tqdm(range(n_iter)):
            noise_value = self.scaled_exp_decay(noise_range[0], noise_range[1],
                n_iter, current_iter)

            self.train_autoencoder(data_loader, batch_size, weight_decay,
                n_iter, noise_value, learn_rate_ae, current_iter)

            self.train_wgan(data_loader, batch_size, keep_prob, weight_decay,
                learn_rate_gen, learn_rate_disc, n_critic, current_iter)


    # --------------------------------------------------------------------------
    def train_autoencoder(self, data_loader, batch_size, weight_decay, n_iter,
        noise_value, learn_rate, current_iter):
            #evaluate
            batch = data_loader.test.next_batch(batch_size)
            feedDict = {self.ae.inputs : batch[0],
                        self.ae.weight_decay : weight_decay,
                        self.ae.learn_rate : learn_rate,
                        self.ae.noise_value : noise_value,
                        self.ae.is_training : False}
            _, summary = self.sess.run([self.ae.cost, self.ae.merged], feed_dict=feedDict)
            self.ae.test_writer.add_summary(summary, current_iter)

            #train
            batch = data_loader.train.next_batch(batch_size)
            feedDict[self.ae.inputs] = batch[0]
            feedDict[self.ae.is_training] = True
            _, summary = self.sess.run([self.ae.train, self.ae.merged], feed_dict=feedDict)
            self.ae.train_writer.add_summary(summary, current_iter)


    # --------------------------------------------------------------------------
    def train_wgan(self, data_loader, batch_size, keep_prob, weight_decay,
        learn_rate_gen, learn_rate_disc, n_critic, current_iter):
        batch = data_loader.train.next_batch(batch_size)
        feedDict = {self.wgan.z : np.random.normal(size=[batch_size, self.wgan.z_dim]),
                    self.wgan.keep_prob : keep_prob,
                    self.wgan.weight_decay : weight_decay,
                    self.wgan.learn_rate : learn_rate_disc,
                    self.wgan.is_training : True,
                    self.ae.inputs : batch[0],
                    self.ae.is_training : False}

        self.sess.run(self.wgan.train_disc, feed_dict=feedDict)
        self.sess.run(self.wgan.clip_weights)

        if current_iter%n_critic == 0:
            feedDict[self.wgan.learn_rate] = learn_rate_gen
            self.sess.run(self.wgan.train_gen, feed_dict=feedDict)

        disc_s = self.sess.run(self.wgan.disc_merge, feed_dict=feedDict)
        gen_s = self.sess.run(self.wgan.gen_merge, feed_dict=feedDict)
        self.wgan.train_writer.add_summary(disc_s, current_iter)
        self.wgan.train_writer.add_summary(gen_s, current_iter)



def test_ARAE():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    arae = ARAE(input_dim=784, z_dim=32, c_dim=100, do_train=True)
    arae.train_model(data_loader=mnist, batch_size=256, weight_decay=0.01,
        n_iter=100000, noise_range=[0.4, 1e-3], learn_rate_ae=5e-4, keep_prob=1,
        learn_rate_gen=5e-5, learn_rate_disc=5e-4, n_critic=10)
    
