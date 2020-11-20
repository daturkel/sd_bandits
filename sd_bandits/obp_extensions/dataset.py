from obp.dataset import BaseRealBanditDataset
import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm import tqdm


class DeezerDataset(BaseRealBanditDataset):
    def __init__(
        self,
        user_features: str,
        playlist_features: str,
        len_list: int = 12,
        len_init: int = 3,
    ):
        (
            self.user_features,
            self.user_segments,
            self.playlist_features,
        ) = self.load_raw_data(user_features, playlist_features)

        self.dim_context = self.user_features.shape[1]
        self.dim_action_context = self.playlist_features.shape[1]
        self.n_actions = self.playlist_features.shape[0]
        self.len_list = len_list
        self.len_init = len_init

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
        n_batches: int = 100,
        users_per_batch: int = 20000,
        replace_within_batch: bool = True,
        seed: int = 1,
        cascade: bool = True,
    ):
        rng = np.random.default_rng(seed)
        user_indices = []
        for i in range(n_batches):
            user_indices_batch = rng.choice(
                range(len(self.user_features)),
                size=users_per_batch,
                replace=replace_within_batch,
            )
            user_indices.append(user_indices_batch)
        user_indices = np.concatenate(user_indices)
        all_scores = []
        all_item_indices = []

        for user_idx in tqdm(user_indices, desc="Calculating user scores:"):
            item_indices = rng.choice(
                range(len(self.playlist_features)), size=self.len_list, replace=False
            )
            all_item_indices.append(item_indices)
            user = self.user_features[user_idx, :]
            items = self.playlist_features[item_indices, :]
            raw_scores = items @ user
            all_scores.append(raw_scores)

        all_probabilities = expit(np.array(all_scores))
        all_item_indices = np.array(all_item_indices)

        rewards_uncascaded = rng.binomial(n=1, p=all_probabilities)

        relevant_user_features = self.user_features[user_indices, :]

        actions = []
        rewards = []
        positions = []
        context = []

        # apply the cascade effect so that the first playlist interacted with is
        # the only playlist interacted with
        for row in tqdm(
            range(rewards_uncascaded.shape[0]), desc="Generating feedback:"
        ):
            # Loosely based on Deezer's simulation code
            # https://github.com/deezer/carousel_bandits/blob/master/environment.py#L77

            no_rewards = rewards_uncascaded[row, :].sum() == 0

            # in cascade mode, check only the first len_init spots if there are no rewards
            # otherwise check the whole list
            if cascade:
                positions_to_check = self.len_init if no_rewards else self.len_list
            # if not cascade mode, always check the whole list
            else:
                positions_to_check = self.len_list

            for position in range(positions_to_check):
                # if we're in cascade mode AND we've made it past len_init AND there are
                # no rewards left, stop recording rewards
                if cascade:
                    remaining_rewards = rewards_uncascaded[row, position:].sum()
                    if (position >= self.len_init) and (remaining_rewards == 0):
                        break

                reward = rewards_uncascaded[row, position]
                rewards.append(reward)
                positions.append(position)
                actions.append(all_item_indices[row, position])
                context.append(relevant_user_features[row])

        actions = np.array(actions)
        rewards = np.array(rewards)
        positions = np.array(positions)
        context = np.array(context)
        action_context = self.playlist_features[actions, :]
        pscore = np.array([1 / self.playlist_features.shape[0] for i in actions])
        n_rounds = len(actions)

        return {
            "action": actions,
            "reward": rewards,
            "position": positions,
            "context": context,
            "action_context": action_context,
            "pscore": pscore,
            "n_rounds": n_rounds,
            "n_actions": self.n_actions,
        }
