import numpy as np

from candidate_model import CandidateModel
from ranking import RankingModel
from filtering import Filtering

USER_ID = 1


if __name__ == '__main__':

    # CandidateModel Stages
    cnd_model: CandidateModel = CandidateModel()
    cnd_model.load_data()
    cnd_model.set_continuous_indexes()
    cnd_model.create_user_item_matrix()
    cnd_model.perform_matrix_factorization()
    cnd_model.reconstruct_user_item_matrix()
    cnd_model.get_candidate_df(threshold=4)

    # decreased # of candidates to top_k_movies instead of "all_users" * "all_items" space
    # remember: ratings are set by matrix decomposition and not include user & item feature as well as embeddings
    item_candidates = cnd_model.get_recommendations_from_mf(user_id=USER_ID)
    item_candidates_idxs = item_candidates.movieId.to_list()

    print(f'\nCandidates for user {USER_ID}:\n',
          item_candidates[['movieId', 'rating', 'title']].head())

    # ranking model is trained on real ratings, not user-item-matrix, so it's easy to separate workload.
    # This is why in big-tech there are different teams working under Ranking and Candidate Models

    # RankingModel Stages
    ranking_model = RankingModel()
    ranking_model.load_data()
    ranking_model.preprocessing()
    ranking_model.generate_target(threshold=4)
    ranking_model.change_dtypes()
    ranking_model.get_train_test()
    ranking_model.train()
    ranking_model.evaluate()

    # Inference on Candidates taken from CandidateModel
    taken_candidates = ranking_model.take_candidates(USER_ID, item_candidates_idxs)
    prpc_taken_candidates = ranking_model.preprocess_candidates(taken_candidates)
    taken_candidates['is_relevant'], taken_candidates['relevant_score'] = ranking_model.predict(prpc_taken_candidates)
    taken_candidates = taken_candidates.reset_index()

    # Just imagine, person set do not provide me this item as it's totally disgusting
    taken_candidates['do_not_show_anymore'] = np.random.choice([0, 1], size=len(taken_candidates))

    print(f'\nRanking Candidates with scores for user {USER_ID}:\n',
          taken_candidates[['movieId', 'rating', 'is_relevant', 'relevant_score', 'do_not_show_anymore']].head(10))

    # Filtering Stages
    taken_candidates = Filtering(taken_candidates).filter_users_not_likes()

    print(f'\nRanking Candidates with scores for user {USER_ID} after filtering:\n',
          taken_candidates[['movieId', 'rating', 'is_relevant', 'relevant_score', 'do_not_show_anymore']].head(10))
