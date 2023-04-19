import pandas as pd
import numpy as np

from typing import Optional

np.random.seed(42)

class CandidateModel(object):
    def __init__(self):
        self.movies_data: Optional[pd.DataFrame] = None
        self.ratings_data: Optional[pd.DataFrame] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.candidate_df: Optional[pd.DataFrame] = None

    def load_data(self, ratings_data_path='./data/ml-25m/ratings.csv', movie_data_path='./data/ml-25m/movies.csv'):
        # Load data as dataframes
        self.movies_data = pd.read_csv(movie_data_path, usecols=['movieId', 'title'])
        self.ratings_data = pd.read_csv(ratings_data_path, nrows=10000)

    def get_recommendations_from_mf(self, user_id, top_k_movies=10):
        recommended_movies = self.candidate_df[self.candidate_df['userId'] == user_id].\
            sort_values('rating', ascending=False)
        return recommended_movies.iloc[:top_k_movies].reset_index(drop=True)

    def set_continuous_indexes(self):
        # Convert the movie ids and user ids to a continuous index
        # is needed to not create dataframe where only 1 item exists with item_id=256 and 255 zeros created for him
        unique_users = self.ratings_data['userId'].unique()
        unique_movies = self.ratings_data['movieId'].unique()
        user_index = {user: idx for idx, user in enumerate(unique_users)}
        movie_index = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.ratings_data['user_index'] = self.ratings_data['userId'].apply(lambda x: user_index[x])
        self.ratings_data['movie_index'] = self.ratings_data['movieId'].apply(lambda x: movie_index[x])

    def create_user_item_matrix(self):
        """
        User-Item Matrix is created in dense way for studying purposes. User Sparse matrix for efficiency.
        """
        # pivot matrix, so NaNs appear for each not-rated user-item pairs
        self.user_item_matrix = self.ratings_data.pivot_table(index='user_index', columns='movie_index', values='rating')
        # replace NaN to 0 to converge linalg.svd
        self.user_item_matrix[np.isnan(self.user_item_matrix)] = 0

    def perform_matrix_factorization(self):
        """
        Perform Singular Value Decomposition (SVD) on the user-item matrix to generate embeddings.
        Decomposition assumes finding user and items vectors latent factors(embeddings) which may reconstruct the matrix
        with minimum loss

        "U" - unitary matrix that represents the left singular vectors of matrix A (represents Users)
        "S" - diagonal matrix that contains the singular values of matrix A
        "V^T - unitary matrix that represents the right singular vectors of A (represents Items)"

        How SVD works explain good here:
            https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/

        Performing a matrix factorization into only two matrices, without involving a diagonal
        matrix is called "economy sized SVD"

        Even though diagonal matrix that is obtained in the full SVD is crucial because it captures the relative
        importance of the singular vectors. Specifically, the singular values on the diagonal of the matrix S provide
        a measure of the "strength" of each singular vector in the decomposition.

        :return: U, S, V^T
        """
        return np.linalg.svd(self.user_item_matrix)

    def reconstruct_user_item_matrix(self, U, S, Vt, k=50):

        # Reconstruct the user-item matrix using the SVD components
        # choose the number of latent factors "k", decrease dimensionality, latent factors are also called embeddings
        # For better accuracy take maximum number of latent factors, which are equal to total number of candidates

        print('# of total latent factors:', len(S))
        print('Chosen # of latent factors:', k)

        # Fill the zeros in user_item matrix using Matrix Decomposition
        S_diag = np.diag(S[:k])
        U_k = U[:, :k]
        Vt_k = Vt[:k, :]
        self.user_item_matrix = U_k @ S_diag @ Vt_k

        # Convert the filled user-item matrix to a dataframe and melt it to a long format
        self.user_item_matrix = pd.DataFrame(self.user_item_matrix, columns=self.ratings_data['movieId'].unique())
        self.user_item_matrix['userId'] = self.ratings_data['userId'].unique()
        # do not forget to user initial userId & movieId indexes
        self.user_item_matrix = pd.melt(self.user_item_matrix, id_vars='userId', var_name='movieId', value_name='rating')

    def get_candidate_df(self, threshold=3.5):
        # Merge the melted user-item matrix with the movies dataframe
        self.candidate_df = self.user_item_matrix.merge(self.movies_data[['movieId', 'title']], on='movieId', how='left')
        # generate target of relevant content, by filter by the ratings threshold
        self.candidate_df['relevant_content'] = np.where(self.candidate_df['rating'] >= threshold, 1, 0)
