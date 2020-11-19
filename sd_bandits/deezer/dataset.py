from obp.dataset import BaseRealBanditDataset
import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm import tqdm


class DeezerDataset(BaseRealBanditDataset):
    def __init__(self, user_features: str, playlist_features: str, seed: int = 1):
        (
            self.user_features,
            self.user_segments,
            self.playlist_features,
        ) = self.load_raw_data(user_features, playlist_features)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def load_raw_data(self, user_features: str, playlist_features: str):
        user_df = pd.read_csv(user_features)
        user_features = np.concatenate(
            [np.array(user_df.drop(["segment"], axis=1)), np.ones((len(user_df), 1))],
            axis=1,
        )
        user_segments = user_df["segment"]

        playlists_df = pd.read_csv(playlist_features)
        playlist_features = np.array(playlists_df)

        return user_features, user_segments, playlist_features

    def pre_process(self):
        pass

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int = 100,
        users_per_round: int = 20000,
        replace_within_round: bool = True,
        len_list: int = 12,
    ):

        (
            actions,
            rewards,
            positions,
            context,
            action_context,
            pscore,
        ) = self._simulate_random_policy(
            n_rounds=n_rounds,
            users_per_round=users_per_round,
            replace_within_round=replace_within_round,
            len_list=len_list,
        )

        return {
            "action": actions,
            "reward": rewards,
            "position": positions,
            "context": context,
            "action_context": action_context,
            "pscore": pscore,
        }

    def _simulate_random_policy(
        self,
        n_rounds: int,
        users_per_round: int,
        replace_within_round: bool,
        len_list: int,
    ):
        user_indices = self.rng.choice(
            range(len(self.user_features)),
            size=users_per_round,
            replace=replace_within_round,
        )
        all_scores = []
        all_item_indices = []

        for user_idx in tqdm(user_indices):
            item_indices = self.rng.choice(
                range(len(self.playlist_features)), size=len_list, replace=False
            )
            all_item_indices.append(item_indices)
            user = self.user_features[user_idx, :]
            items = self.playlist_features[item_indices, :]
            raw_scores = items @ user
            all_scores.append(raw_scores)

        all_probabilities = expit(np.array(all_scores))
        all_item_indices = np.array(all_item_indices)

        rewards_uncascaded = self.rng.binomial(n=1, p=all_probabilities)

        relevant_user_features = self.user_features[user_indices, :]

        actions = []
        rewards = []
        positions = []
        context = []

        # apply the cascade effect so that the first playlist interacted with is
        # the only playlist interacted with
        for row in tqdm(range(rewards_uncascaded.shape[0])):
            rewarded = False
            for position in range(rewards_uncascaded.shape[1]):
                if rewards_uncascaded[row, position] == 1 and not rewarded:
                    rewarded = True
                    rewards.append(1)
                else:
                    rewards.append(0)
                positions.append(position)
                actions.append(all_item_indices[row, position])
                context.append(relevant_user_features[row])

        actions = np.array(actions)
        rewards = np.array(rewards)
        positions = np.array(positions)
        context = np.array(context)
        action_context = self.playlist_features[actions, :]
        pscore = np.array([1 / self.playlist_features.shape[0] for i in actions])

        return (actions, rewards, positions, context, action_context, pscore)
