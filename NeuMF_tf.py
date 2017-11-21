'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np

import tensorflow as tf
import theano
import theano.tensor as T
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
# import GMF, MLP
import argparse

BATCH_SIZE = 100
LAYERS = [64,32,16,8]

def scaled_initial_variable(shape, name):
  """
  weight_variable generates a weight variable of a given shape.

  This is a helper function to create the weight variable.  The dimension of
  the weight variable is passed in, and a random array of weights with a stdev
  of .1 is generated.
  """

  scaled_initializer = tf.contrib.layers.variance_scaling_initializer(
    factor=0.01,
    mode='FAN_IN',
    uniform=False,
    seed=None,
    dtype=tf.float32
  )
  initial = tf.get_variable(
    name,
    shape=shape,
    initializer=scaled_initializer
  )
  return initial

def glorot_weight_variable(shape, name):
  glorot_initializer = tf.contrib.layers.xavier_initializer()
  initial = tf.get_variable(
    name,
    shape=shape,
    initializer=glorot_initializer
  )
  return initial

def lecun_weight_variable(shape, name):
  lecun_initializer = tf.contrib.keras.initializers.lecun_uniform()
  initial = tf.get_variable(
    name,
    shape=shape,
    initializer=lecun_initializer,
  )
  return initial

def bias_variable(shape):
  """
  bias_variable generates a bias variable of a given shape.

  This is a helper function to generate the biases.  Just like the weights,
  a vector the size of the input shape is created with random variables with
  stdev of 0.1
  """
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def init_normal(shape):
    return tf.random_normal(shape, stdev=0.01)


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--data_dir', nargs='?', default='',
                        help='')
    return parser.parse_args()

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
  args = parse_args()
  num_epochs = args.epochs
  batch_size = args.batch_size
  layers = eval(args.layers)
  num_negatives = args.num_neg
  learning_rate = args.lr
  verbose = args.verbose

  topK = 10
  evaluation_threads = 1#mp.cpu_count()
  print("NeuMF arguments: %s " %(args))

  # Loading data
  t1 = time()
  dataset = Dataset(args.path + args.dataset)

  # # train matrix user, item, rating
  train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
  num_users, num_items = train.shape
  print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
        %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))

  # Input data.
  user_inputs = tf.placeholder(tf.int32, shape=(None))
  item_inputs = tf.placeholder(tf.int32, shape=(None))
  train_labels = tf.placeholder(tf.int32, shape=[None, 1])
  y = tf.placeholder(tf.float32, [None, 1])

  embedding_size = LAYERS[0]
  user_embedding_size = embedding_size / 2
  item_embedding_size = embedding_size / 2

  user_embeddings = scaled_initial_variable(
    [num_users, user_embedding_size], 'user_embeddings'
  )
  item_embeddings = scaled_initial_variable(
    [num_items, item_embedding_size], 'item_embeddings'
  )

  user_embed = tf.nn.embedding_lookup(user_embeddings, user_inputs)
  item_embed = tf.nn.embedding_lookup(item_embeddings, item_inputs)

  input_embed = tf.concat([user_embed, item_embed], axis=1)
  activation = input_embed

  for idx in range(1, len(LAYERS)):
    W = glorot_weight_variable([LAYERS[idx - 1], LAYERS[idx]], 'W_layer_{}'.format(idx))
    b = bias_variable([LAYERS[idx]]) #, 'b_layer_{}'.format(idx))
    activation = tf.nn.relu(tf.matmul(activation, W) + b)

  pred_W = lecun_weight_variable([LAYERS[-1], 1], 'prediction_weights')
  pred_b = bias_variable([1])# , 'prediction_bias')
  y_pred = tf.sigmoid(tf.matmul(activation, pred_W) + pred_b)

  with tf.name_scope('loss'):
    with tf.name_scope('log_loss1'):
      loss = tf.losses.log_loss(
        labels=y,
        predictions=y_pred,
      )
      total_loss = tf.reduce_mean(loss)

  with tf.name_scope('adam_optimizer'):
    # our goal is to minimize the cross entropy
    train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

  with tf.name_scope('accuracy'):
    # we will know if our prediction is correct if the highest output vector on
    # the final layer is the same as what the label set suggests
    with tf.name_scope('correct'):
      correct_prediction = tf.equal(y, tf.round(y_pred))

      # We will cast our prediction to a float from a boolean so we can compute
      # the accuracy
      correct_prediction = tf.cast(correct_prediction, tf.float32)
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(correct_prediction)
      tf.summary.scalar('accuracy', accuracy)

    # create a location to create and store the graph info
    merged = tf.summary.merge_all()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Merge all summary info here, so we can write it out.
    train_writer = tf.summary.FileWriter(args.data_dir + '/tmp/tf/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(args.data_dir + '/tmp/tf/test')
    # initialize all of the random variables
    sess.run(tf.global_variables_initializer())

    # loop over the epocs
    for i in range(num_epochs):
      # get the next 50 examples for the batch
      t1 = time()
      user_input, item_input, labels = get_train_instances(train, num_negatives)
      labels = np.expand_dims(labels, axis=1)

      # gather performance metrics every #verbose epocs
      if i % verbose == 0:
        # evaluate the test set accuracy.  We use a keep_prob of 1 because
        # we don't want to drop any of the neurons during evaluation
        summary, train_acc = sess.run(
          [merged, accuracy],
          feed_dict={
            user_inputs: user_input,
            item_inputs: item_input,
            y: labels,
          }
        )
        t2 = time()
        (hits, ndcgs) = evaluate_model(sess, user_inputs, item_inputs, y_pred, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f [%.1f s]' 
              % (i,  t2-t1, hr, ndcg, time()-t2))
        # output the training accuracy
        print('step %d, accuracy %g' % (i, train_acc))
        test_writer.add_summary(summary, i)
      # run feedforward and backpropagation on the batch.  We use a keep
      # probability of .5, meaning we remove a neurons influence on the output
      # randomly half the time
      num_examples = len(user_input) + num_negatives

      summary, _ = sess.run(
        [merged, train_step],
        feed_dict={
          user_inputs: user_input,
          item_inputs: item_input,
          y: labels,
        }
      )
      train_writer.add_summary(summary, i)

#         if epoch %verbose == 0:
#             if hr > best_hr:
#                 best_hr, best_ndcg, best_iter = hr, ndcg, epoch
#                 if args.out > 0:
#                     model.save_weights(model_out_file, overwrite=True)
# 
#     print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
#     if args.out > 0:
#         print("The best NeuMF model is saved to %s" %(model_out_file))
