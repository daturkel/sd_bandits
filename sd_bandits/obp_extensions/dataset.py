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
        """Dataset container for Deezer's carousel bandit data.

        Deezer's dataset doesn't contain logged clicks but instead has user and item (playlist)
        features that can be used to generate clicks based on "ground-truth" click probabilities.

        This class allows for the generation of those clicks.

        Relevant links:
        - repo: https://github.com/deezer/carousel_bandits
        - paper: https://arxiv.org/abs/2009.06546
        - data: https://zenodo.org/record/4048678#.X22w4pMza3J

        Parameters
        ----------
        user_features : str
            Path to the user features csv.
        playlist_features : str
            Path to the playlist features csv.
        len_list : int
            How many playlists can be shown to a user (Deezer uses 12).
        len_init : int
            How many playlists is the user guaranteed to see (Deezer uses 3).
        """
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
        """Load data from Deezer data CSVs.

        Parameters
        ----------
        user_features : str
            Path to the user features csv.
        playlist_features : str
            Path to the playlist features csv.
        """
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
        """Generate feedback based on the "ground truth" Deezer user/item features.

        Sample users (with or without replacement) for each batch, then simulate uniformly random
        policy, where users click actions based on the "ground-truth" click probabilities calculated
        from the user/playlist feature dot-product.

        If cascade is enabled, we don't observe any rewards after max(len_init, last_click_index).

        Parameters
        ----------
        n_batches : int
            How many batches of sampled users to simulate.
        users_per_batch : int
            How many users to sample each batch.
        replace_within_batch : bool
            Whether to replace users when sampling for a batch (Deezer uses True).
        seed : int
            Random seed.
        cascade : bool
            Whether to simulate cascade logic, where rewards after max(len_init, last_click_index)
            are not observed.
        """
        rng = np.random.default_rng(seed)
        user_indices = []
        for i in range(n_batches):
            user_indices_batch = rng.choice(
                range(len(self.user_features)),
                size=users_per_batch,
                replace=replace_within_batch,
            )
            user_indices += list(user_indices_batch)
        all_scores = []
        all_item_indices = []

        for user_idx in tqdm(user_indices, desc="Calculating user scores:"):
            item_indices = rng.choice(
                range(len(self.playlist_features)), size=self.len_list, replace=False
            )
            all_item_indices.append(item_indices)
            user = self.user_features[user_idx, :]
            items = self.playlist_features[item_indices, :]
            # calculate raw user-playlist affinities
            raw_scores = items @ user
            all_scores.append(raw_scores)

        # convert user/playlist affinities into click probabilities
        all_probabilities = expit(np.array(all_scores))
        all_item_indices = np.array(all_item_indices)

        rewards_uncascaded = rng.binomial(n=1, p=all_probabilities)

        relevant_user_features = self.user_features[user_indices, :]

        actions = []
        rewards = []
        positions = []
        context = []

        for row in tqdm(
            range(rewards_uncascaded.shape[0]), desc="Generating feedback:"
        ):
            # Loosely based on Deezer's simulation code, https://github.com/deezer/carousel_bandits/blob/master/environment.py#L77
            # However, the code deviates from the paper in the cascade behavior: paper says that if
            # l_init = 3 and only the 2nd item was clicked, rewards are 0 1 0 X X X ..., but code
            # instead does 0 1 X X X X ... (stopping after 2nd item, rather than after l_init'th item).
            #
            # Guillaume from Deezer confirmed the simplification but noted that it did not make an
            # appreciable difference in the experiments. We choose to simulate the logic of the *paper*
            # (0 1 0 X X X ...).

            # If we're not in cascade mode, we always see all rewards
            if not cascade:
                positions_to_check = self.len_list
            # Otherwise, we see max(l_init, last_click_index + 1) rewards
            else:
                click_indices = np.where(rewards_uncascaded[row, :])[0]
                try:
                    last_click_index = click_indices[-1]
                    positions_to_check = max(self.len_init, last_click_index + 1)
                except IndexError:
                    # no rewards at all
                    positions_to_check = self.len_init

            for position in range(positions_to_check):
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
