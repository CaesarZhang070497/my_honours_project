from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from src import data_processing
import gc
import matplotlib.pyplot as plt

'''
num_hidden_1=155,
num_hidden_2=81,
num_hidden_3=27,
num_hidden_4=9,
num_hidden_5=3,

'''

class autoencoder(object):# 155 27 128 101 74
    def __init__(self,num_epochs=5000, display_step=100, learning_rate=0.1, batch_size=100,
                 denoising=False, new_poll_weight=0.002,masking=0, num_layers=3, num_hidden_1=155,
                 num_hidden_2=81,num_hidden_3=27,num_hidden_4=9,num_hidden_5=3,continue_from_saved=False, time_decay=1):

        self.data_provider = data_processing.data_provider('/afs/inf.ed.ac.uk/user/s16/s1688201/PycharmProjects/complex_collaberative_filtering/src/this_that.json')
        self.data_provider.parse()
        # Interactions are fed as binary but using decimal helps with adding 2 interactions together
        self.interaction_dict = {'skips': 16,
            'owns': 8,
            'tracks': 4,
            'comment': 2,
            'vote': 1
        }

        self.polls = self.data_provider.polls  # [:50]
        self.num_engagements = len(self.interaction_dict)

        self.num_input = len(self.polls)
        tf.set_random_seed(1)

        self.users = self.data_provider.users#[:50]

        self.test_polls = self.data_provider.polls#[500:]
        self.num_epochs = num_epochs # initial training eppchs given training data
        self.display_step = display_step # display training loss every x epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.denoising = denoising # whether to add noise to the input vectors - might help with accidental interactions
        self.new_poll_weight = new_poll_weight # How much weight new polls are given in the output layer (gives new polls some initial traction)
        self.masking = masking # TODO: Add masking to data to make synthetic users with less interactions and see if it helps
        self.num_layers = num_layers
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.num_hidden_3 = num_hidden_3
        self.num_hidden_4 = num_hidden_4
        self.num_hidden_5 = num_hidden_5
        self.continue_from_saved = continue_from_saved
        self.train= self.data_provider.train
        self.test = self.data_provider.test
        self.validation = self.data_provider.validation
        self.test_users = []
        self.time_decay = time_decay
        self.X = tf.placeholder("float", [None, None])
        self.Y = tf.placeholder("float", [None, None])
        self.saver = None

        self.compressed_train = []
        self.compressed_test = []

        self.weights2 = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'encoder_h3': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_3])),
            'encoder_h4': tf.Variable(tf.random_normal([self.num_hidden_3, self.num_hidden_4])),

            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_4, self.num_hidden_3])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_3, self.num_hidden_2])),
            'decoder_h3': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h4': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input])),

        }
        self.biases2 = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([self.num_hidden_3])),
            'encoder_b4': tf.Variable(tf.random_normal([self.num_hidden_4])),

            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_3])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b3': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b4': tf.Variable(tf.random_normal([self.num_input])),

        }
        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op)
        self.setup_graph()

        self.train_and_predict()

    def encoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights2['encoder_h1']),self.biases2['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights2['encoder_h2']),self.biases2['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights2['encoder_h3']), self.biases2['encoder_b3']))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, self.weights2['encoder_h4']), self.biases2['encoder_b4']))
        return layer_4

    def decoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights2['decoder_h1']), self.biases2['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights2['decoder_h2']),self.biases2['decoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights2['decoder_h3']), self.biases2['decoder_b3']))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, self.weights2['decoder_h4']), self.biases2['decoder_b4']))
        return layer_4

    def setup_graph(self):

        # Prediction
        self.y_pred = self.decoder_op

        self.y_true = self.Y
        self.loss = self._loss()
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def _loss(self):
        return tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))

    def train_and_predict(self, save=True):
        gc.collect()

        f = open("guru99.txt", "w+")

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            train_loss = []
            validation_loss = []
            t0 = time.time()

            for i in range(0,self.num_epochs):
                self.train = np.asarray(self.train)
                batch_x = self.train[np.random.choice(self.train.shape[0], self.batch_size, replace=True), :]
                batch_y = np.copy(batch_x)

                _, l = sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y})
                train_loss.append(l)


                # print("the minibatch loss for epoch %d is %f"% (i,l));

                f.write("%f\r\n" % l)

                l = sess.run(self.loss, feed_dict={self.X: self.validation, self.Y: self.validation})
                validation_loss.append(l)


            train_encoder_result = sess.run(self.encoder_op, feed_dict={self.X: self.train})
            test_encoder_result = sess.run(self.encoder_op, feed_dict={self.X: self.test})

            print()

            self.compressed_train = np.copy(train_encoder_result)
            self.compressed_test = np.copy(test_encoder_result)
            # print(test_encoder_result)

#            xs = np.arange(1,self.num_epochs+1,1)
#            ys = train_loss
#            fig, ax = plt.subplots(figsize=(12, 8))
#            ax.plot(xs, ys,color = 'b')
#            xs = np.arange(1, self.num_epochs+1, 1)
#            ys = validation_loss
#            plt.plot(xs, ys,color = 'r')
#            plt.show()

            print('autoencoder finished')
        sess.close()

#a = autoencoder()
