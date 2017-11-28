'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, StratifiedShuffleSplit, train_test_split
)


class BaseDataset(object):
    def __init__(self, path, genre_path=None, tag_path=None):
        '''
        Base class for handling movie lens data.  Handles loading the data,
        transforming the columns to generic names, designating embedding vector
        input sizes (through the num_* instance variables), and stores a mapping
        so that the original instances can be referenced from the new embedding.

        Also handles retrieving, cleaning and storing Tag and Genre information.

        Requires that the 'get_labels' instance method be overridden.  The
        'get_labels' method is used to retrieve label information for training
        and testing.
        '''
        self.data = self.load_rating_file_as_df(path + ".train.rating")
        if genre_path:
            self.genre_df = pd.read_csv(genre_path)
            self.genre_idx_map, self.genre_df['genres'] = self.get_idx_mapping(self.genre_df, 'genres')
            self.add_genres()
            self.num_genres = len(self.genre_idx_map)
        if tag_path:
            self.tag_df = pd.read_csv(tag_path)
            self.tag_idx_map, self.tag_df['tagId'] = self.get_idx_mapping(self.tag_df, 'tagId')
            self.add_tags()
            self.num_tags = len(self.tag_idx_map)
        self.user_idx_map, self.data[0]['user'] = self.get_idx_mapping(self.data[0], 'user')
        self.item_idx_map, self.data[0]['item'] = self.get_idx_mapping(self.data[0], 'item')
        self.train_x, self.test_x, self.train_y, self.test_y = self.split_test_data()
        self.num_users = len(self.data[0]['user'].unique())
        self.num_items = len(self.data[0]['item'].unique())

    def get_labels(df, label_column_name):
        raise NotImplemented()

    def load_rating_file_as_df(self, filename):
        '''
        Read .rating file and return a dataframe.
        '''
        ratings_df = pd.read_csv(
            filename,
        )
        ratings_df[['user', 'item']] = ratings_df[['userId', 'movieId']]
        return [ratings_df[['user', 'item']], self.get_labels(ratings_df, 'rating')]

    def get_idx_mapping(self, df, field_name):
        keys = df[field_name].unique()
        series_idx = [idx for idx in range(len(keys))]
        mapping = {key: idx for key, idx in zip(keys, series_idx)}
        inv_map = {v: k for k, v in mapping.items()}
        return inv_map, df[field_name].map(mapping)

    def split_test_data(self):
        train_x, test_x, train_y, test_y = train_test_split(
            self.data[0],
            self.data[1],
            test_size=0.2,
        )
        return train_x, test_x, train_y, test_y

    def add_genres(self):
        self.genre_df.rename({'movieId': 'item'}, inplace=True, axis=1)
        self.genre_df.drop(['title'], axis=1, inplace=True)
        self.data[0] = self.data[0].merge(self.genre_df, how='left', on='item')

    def add_tags(self, tag_path):
        tag_df = pd.read_csv(tag_path)
        tag_df.rename({'movieId': 'item'}, inplace=True, axis=1)
        tag_df = tag_df.sort_values(['item', 'relevance'], ascending=False)
        tag_df = tag_df.groupby('item').head(10)
        self.data[0] = self.data[0].merge(tag_df, how='left', on='item')
        tag_map = self.get_idx_mapping(self.data[0]['tagId'])
        self.data[0]['tagId'] = self.data[0]['tagId'].map(genre_map)

class PDataset(BaseDataset):
    '''
    Designates the output vector as a series of classes.  For movielens, there
    will be 10 ouput classes from 0.5 to 5 in 0.5 increments.
    '''
    def get_labels(self, df, label_column_name):
        return pd.get_dummies(df[label_column_name])

class CDataset(BaseDataset):
    '''
    Designates the output vector as a continuous variable.  For movie lens this
    is float from 0.5 to 5.  All values are in 0.5 increments.
    '''
    def get_labels(self, df, label_column_name):
        return df[label_column_name]

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat
