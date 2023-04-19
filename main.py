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
    U, S, Vt = cnd_model.perform_matrix_factorization()
    cnd_model.reconstruct_user_item_matrix(U, S, Vt, k=74)
    cnd_model.get_candidate_df(threshold=4)

    # decreased # of candidates to top_k_movies instead of "All_Users" * "All_Items" space
    # to specified number of candidates
    cols_to_show = ['movieId', 'rating', 'title']

    item_candidates = cnd_model.get_recommendations_from_mf(user_id=USER_ID)[cols_to_show]
    item_candidates_idxs = item_candidates.movieId.to_list()

    print(f'\nCandidates for user {USER_ID}:\n',
          item_candidates[cols_to_show].head())

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

    cols_to_show = ['movieId', 'rating', 'is_relevant', 'relevant_score']
    taken_candidates = taken_candidates[cols_to_show].sort_values('relevant_score', ascending=False).reset_index(drop=True)
    # Just imagine, person set do not provide me this item as it's totally disgusting
    taken_candidates['not_interested'] = np.random.choice([0, 1], size=len(taken_candidates))
    print(f'\nRanking Candidates with scores for user {USER_ID}:\n', taken_candidates.head(10))

    # Filtering Stages
    taken_candidates = Filtering(taken_candidates).filter_users_not_likes()
    print(f'\nRanking Candidates with scores for user {USER_ID} after filtering:\n', taken_candidates.head(10))
