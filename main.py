import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from WGAN.WGAN import WGAN
from autoencoder.autoencoder import AE
from classifier.classifier import test_classifier

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.Graph().as_default() as g_1:
    ae = AE(input_dim=784, z_dim=100, do_train=True, scope='autoencoder')
    wgan = WGAN(do_train=True, input_dim=100, z_dim=32, scope='WGAN')


    sess = tf.Session()
    wgan.sess = sess
    ae.sess = sess
    ae.sess.run(tf.global_variables_initializer())
    wgan.sess.run(tf.global_variables_initializer())


    batch = mnist.train.next_batch(128)

    feedDict = {ae.inputs : batch[0],
                ae.weight_decay : 1e-2,
                ae.learn_rate : 0.0001,
                ae.noise_value : 0.0001,
                ae.is_training : True}
    sess.run(ae.train, feed_dict=feedDict)


    wgan.inputs = ae.z
    print('wgan.inputs', wgan.inputs)
    a = sess.run(wgan.inputs, {ae.inputs : batch[0], ae.is_training : False})
    print(a.shape, a)

    feedDict = {wgan.z : np.random.normal(size=[128, wgan.z_dim]),
                wgan.keep_prob : 1,
                wgan.weight_decay : 1e-2,
                wgan.learn_rate : 0.0001,
                wgan.is_training : True,
                ae.inputs : batch[0],
                ae.is_training : False}
    sess.run(wgan.train_disc, feed_dict=feedDict)
    sess.run(wgan.train_gen, feed_dict=feedDict)





# test_WGAN()
# test_autoencoder()
# test_classifier()