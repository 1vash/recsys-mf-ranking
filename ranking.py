import os

import numpy as np
import random
import pandas as pd

from typing import Optional, Any, List

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Set the random seeds for reproducibility
seed_value = 45

np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy


not_train_features = ['userId', 'movieId', 'title', 'timestamp', 'rating', 'relevant_content']


# TODO: Make RankingModel inheritable from interfaces

class RankingModel(object):
    """
    A binary classification model that predicts whether a movie is relevant or not based on its genres and id.

    Attributes:
        X_train (pd.DataFrame): The training features.
        X_test (pd.DataFrame): The testing features.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The testing labels.

        model (Sequential): The Keras Sequential model.
        mlb (MultiLabelBinarizer): Encoding object
        enc (OneHotEncoder): Encoding object
    """

    def __init__(self):
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self.y_pred: Optional[pd.Series] = None
        self.y_pred_proba: Optional[pd.Series] = None

        self.data: Optional[pd.DataFrame] = None

        self.model: Optional[Any] = None

        # trick to not save scalers/encoders and reuse them for testing/inference
        self.mlb: Optional[Any] = None
        self.enc: Optional[Any] = None

    def load_data(self,
                  ratings_data_path: str ='./data/ml-25m/ratings.csv',
                  movie_data_path: str ='./data/ml-25m/movies.csv'):
        """
        Loads the movie and ratings data from CSV files and merges them into one DataFrame.

        Args:
            ratings_data_path (str): The path to the ratings.csv file.
            movie_data_path (str): The path to the movies.csv file
        """

        # Load data as dataframes
        movies_data = pd.read_csv(movie_data_path, usecols=['movieId', 'title', 'genres'])
        ratings_data = pd.read_csv(ratings_data_path, nrows=10000)
        self.data = pd.merge(movies_data, ratings_data, on='movieId')

    def change_dtypes(self):
        self.data = self.data.astype({
            'userId': 'int32',
            'movieId': 'int32',
            'rating': 'float16',
            'relevant_content': 'int8'})

        one_hot_cols = self.mlb.classes_.tolist()
        self.data[one_hot_cols] = self.data[one_hot_cols].astype('int8')

    def get_multilabel_binarizer_genre_features(self):
        # split the genre content from pipe delimited to a list in a new column
        self.data['genres_list'] = self.data['genres'].str.split('|')
        # assign a new series to the genres_list column that contains a list of categories for each movie
        list2series = pd.Series(self.data.genres_list)
        # convert labels list to multi label binarizer (as onehot but for multiple labels for each row)
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(list2series)
        # use mlb to create a new dataframe of the genres from the list for each row from the original data
        return pd.DataFrame(self.mlb.transform(list2series), columns=self.mlb.classes_, index=list2series.index)


    def get_one_hot_features(self, categories):
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(self.data[categories])
        transformed_data = self.enc.transform(self.data[categories]).toarray()
        return pd.DataFrame(transformed_data, columns=self.enc.get_feature_names_out())

    def preprocessing(self):
        df_genres = self.get_multilabel_binarizer_genre_features()

        # IMPORTANT: Users and Items are represented as one-hot encoding features for demo purposes.
        # Use embeddings for memory efficient and more accurate solution!
        ohe_df = self.get_one_hot_features(categories=['userId', 'movieId'])

        # merge the one_hot_genres with df dataframe and drop the 'genres' specific columns
        self.data = pd.concat([self.data, ohe_df, df_genres], axis=1)
        self.data = self.data.drop(['genres', 'genres_list'], axis=1)

    def generate_target(self, threshold=3.5):
        self.data['relevant_content'] = np.where(self.data['rating'] >= threshold, 1, 0)

    def get_train_test(self):
        # Validation
        # Split data into train and test sets using random shuffling (caring about "user's historical preferences")
        # for caring about "user's current preferences" which changing over time use time based splitting

        X = self.data.drop(not_train_features, axis=1).to_numpy()
        y = self.data['relevant_content'].to_numpy()

        # As we do not tune hyperparameters, validation dataset is skipped
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

        print('Train/Test Shapes:', self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)

    def get_model(self):
        model = Sequential([
            Dense(128, input_shape=(self.X_train.shape[1],), activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu', kernel_regularizer='l2'),
            Dropout(0.5),
            Dense(1, activation='sigmoid', kernel_regularizer='l2')
        ])
        model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def train(self):
        self.model = self.get_model()
        self.model.summary()
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                       epochs=3, batch_size=64)

    def take_candidates(self, user_id, item_candidates_idxs):
        item_candidates_mask = self.data.movieId.isin(item_candidates_idxs)
        user_mask = self.data.userId == user_id
        taken_candidates = self.data[user_mask & item_candidates_mask].copy()
        return taken_candidates

    @staticmethod
    def preprocess_candidates(data: pd.DataFrame):
        taken_candidates = data.drop(not_train_features, axis=1)
        return taken_candidates.to_numpy()

    def predict(self, data, binary_threshold=0.5):
        y_pred_proba = self.model.predict(data)
        y_pred = (y_pred_proba > binary_threshold).astype(int).flatten()
        return y_pred, y_pred_proba

    def evaluate(self):
        """
        Calculate predictions and calculate metrics
        """
        self.y_pred, self.y_pred_proba = self.predict(self.X_test)

        print('\nMETRICS:')
        print('Accuracy:', accuracy_score(self.y_test, self.y_pred))
        print('Precision:', precision_score(self.y_test, self.y_pred))
        print('Recall:', recall_score(self.y_test, self.y_pred))
        print('F1 Score:', f1_score(self.y_test, self.y_pred, average='binary'))
        print('\n')

    def save_model(self, path: str = './models/ranking_model.h5'):
        self.model.save(path)