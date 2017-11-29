'''
Created on Aug 9, 2016

Keras Implementation of Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from Dataset import CDataset as Dataset
from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse
from sklearn.model_selection import StratifiedKFold
from embedding_helper import save_embedding

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-xm',
                        help='Choose a dataset.')
    parser.add_argument('--save_name', nargs='?', default='SET_SAVE_NAME_',
                        help='Name of file to save embeddings to')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def init_normal(shape, name=None):
    return initializers.VarianceScaling(scale=0.01)

def get_model(num_users, num_items, latent_dim, regs=[0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_regularizer = l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_regularizer = l2(regs[1]), input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))

    # Element-wise product of user and item embeddings 
    predict_vector = merge([user_latent, item_latent], mode = 'mul')

    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='relu', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)

    model = Model(input=[user_input, item_input], output=prediction)

    return model

if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    save_name = args.save_name
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("GMF arguments: %s" %(args))
    model_out_file = 'Pretrain/%s_GMF_%d_%d.h5' %(args.dataset, num_factors, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset,)
    train_x, train_y, test_x, test_y = dataset.train_x, dataset.train_y, dataset.test_x, dataset.test_y
    num_users = dataset.num_users
    num_items = dataset.num_items
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train_x.shape[0], len(test_y)))

    # Build model
    model = get_model(num_users, num_items, num_factors, regs)

    if learner.lower() == "adagrad":
        model.compile(
            optimizer=Adagrad(lr=learning_rate), loss='mse',
            metrics=['mae']
        )
    elif learner.lower() == "rmsprop":
        model.compile(
            optimizer=Adagrad(lr=learning_rate), loss='mse',
            metrics=['mae']
        )
    elif learner.lower() == "adam":
        model.compile(
            optimizer=Adagrad(lr=learning_rate), loss='mse',
            metrics=['mae']
        )

    t1 = time()
    test_user = test_x.as_matrix()[:, 0]
    test_item = test_x.as_matrix()[:, 1]
    test_label = test_y.as_matrix()

    initial_evaluation = model.evaluate([test_user, test_item], test_label)
    # (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    # hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    #mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights())+np.linalg.norm(model.get_layer('item_embedding').get_weights())
    #p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
    # print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
    print('evaluation: {}'.format(initial_evaluation))
    best_loss = initial_evaluation

    # Train model
    # best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input = train_x['user'].as_matrix()
        item_input = train_x['item'].as_matrix()
        labels = train_y.as_matrix()

        # Training
        hist = model.fit(
            [user_input, item_input], #input
            labels, # labels
            batch_size=batch_size,
            epochs=1,
            verbose=0,
            shuffle=True
        )
        t2 = time()

        # Evaluation
        if epoch %verbose == 0:
            loss = model.evaluate([test_user, test_item], test_label)
            # (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            # hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]

            print('Iteration %d [%.1f s]\tmse = %.4f [%.1f s]\tmae = %.4f'
                  % (epoch,  t2-t1, loss[0], time()-t2, loss[1]))
            if epoch %10 == 0:
                # only get train loss every once in a while
                train_loss = model.evaluate([user_input, item_input], labels)
                print('Train Iteration %d [%.1f s]\tmse = %.4f [%.1f s]\tmae = %.4f'
                      % (epoch,  t2-t1, train_loss[0], time()-t2, train_loss[1]))

            if loss < best_loss:
                best_loss = loss
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)
                    save_embedding(model, dataset.user_idx_map, 2, '{}_user_mf_test.pickle'.format(save_name))
                    save_embedding(model, dataset.item_idx_map, 3, '{}_item_mf_test.pickle'.format(save_name))
            save_embedding(model, dataset.user_idx_map, 2, '{}_user_mf_train.pickle'.format(save_name))
            save_embedding(model, dataset.item_idx_map, 3, '{}_item_mf_train.pickle'.format(save_name))

        print("The best GMF model is saved to %s" %(model_out_file))
