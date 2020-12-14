from typing import Optional, Tuple, Union

from obp.dataset import BaseRealBanditDataset
from obp.policy import BaseContextFreePolicy, BaseContextualPolicy
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

    def load_raw_data(
        self, user_features: str, playlist_features: str
    ) -> Tuple[np.array]:
        """Load data from Deezer data CSVs.

        Parameters
        ----------
        user_features : str
            Path to the user features csv.
        playlist_features : str
            Path to the playlist features csv.

        Returns
        -------
        Tuple[np.array]
            A tuple of three arrays: the user features, user segments and playlist features.
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

    def _get_positions_to_observe(
        self, rewards_uncascaded: np.array, cascade: bool, cascade_at: str
    ) -> int:
        """Determine which positions to observe the rewards of.

        Loosely based on Deezer's simulation code, https://github.com/deezer/carousel_bandits/blob/master/environment.py#L77
        However, the code deviates from the paper in the cascade behavior: paper says that if
        l_init = 3 and only the 2nd item was clicked, rewards are 0 1 0 X X X ..., but code
        instead does 0 1 X X X X ... (stopping after 2nd item, rather than after l_init'th item).

        Guillaume from Deezer confirmed the simplification but noted that it did not make an
        appreciable difference in the experiments. We choose to simulate the logic of the *paper*
        (0 1 0 X X X ...).

        Deezer's code also cascades at the _first_ clicked item, so we default to that behavior,
        but allow cascading at the _last_ clicked item too.

        Parameters
        ----------
        rewards_uncascaded : np.array
            An array of the full list of rewards for all positions.
        cascade : bool
            Whether or not to simulate the cascade.
        cascade_at : str
            One of "first" or "last" indicating if we want to stop observing at the first or
            last clicked item.

        Returns
        -------
        int :
            the number of positions whose rewards we'll observe

        """
        # If we're not in cascade mode, we always see all rewards
        if not cascade:
            positions_to_observe = self.len_list
        # Otherwise, we see max(l_init, click_index_of_interest + 1) rewards
        else:
            # the indices of every position that was clicked on
            click_indices = np.where(rewards_uncascaded)[0]
            index_of_interest = 0 if cascade_at == "first" else -1
            try:
                click_index_of_interest = click_indices[index_of_interest]
                positions_to_observe = max(self.len_init, click_index_of_interest + 1)
            except IndexError:
                # no rewards at all, default to len_init
                positions_to_observe = self.len_init
        return positions_to_observe

    def obtain_batch_bandit_feedback(
        self,
        n_batches: int = 100,
        users_per_batch: int = 20000,
        seed: int = 1,
        cascade: bool = True,
        cascade_at: str = "first",
        policy: Optional[Union[BaseContextFreePolicy, BaseContextualPolicy]] = None,
        lightweight: bool = False,
    ) -> dict:
        """Generate feedback based on the "ground truth" Deezer user/item features.

        If a policy is provided, we use that policy to generate actions for each round and update
        the policy every batch. Otherwise, actions are always selected uniformly at random.

        If cascade is enabled, we don't observe any rewards after max(len_init, last_click_index).

        Parameters
        ----------
        n_batches : int
            How many batches of sampled users to simulate.
        users_per_batch : int
            How many users to sample each batch.
        seed : int
            Random seed.
        cascade : bool
            Whether to simulate cascade logic, where rewards after max(len_init, last_click_index)
            are not observed.
        cascade_at : str
            One of "first" or "last" indicating if we want to stop observing at the first or
            last clicked item.
        policy : Optional[Union[BaseContextFreePolicy, BaseContextualPolicy]]
            Optionally, a policy that will be used for online simulation.
        lightweight : bool (default = False)
            Only return the feedback relevant for reward analysis: segments, batches, reward,
            n_rounds, n_actions, and policy (if applicable).

        Returns
        -------
        dict
            A feedback dict containing:
                action : np.array (n_rounds,)
                    Array of actions presented.
                reward : np.array (n_rounds,)
                    Array of rewards returned.
                position : np.array (n_rounds,)
                    Array of the position of each action.
                context : np.array (n_rounds, 97)
                    Array of the user context for each round.
                action_context : np.array (n_rounds, 97)
                    Array of the action context for each round.
                pscore : np.array (n_rounds,)
                    Array of the probability of choosing this action (always = 1 / n_actions).
                n_rounds : int
                    Number of users per batch * number of batches * number of items observed for that
                    user's session.
                n_actions : int
                    Number of available items (always 862).
                users : np.array (n_rounds * users_per_round,)
                    The users that were generated for this simulation.
                batches : np.array (n_rounds,)
                    The batch number corresponding to each round.
                segments : np.array (n_rounds,)
                    The segment according to each round.

        """
        if cascade_at not in ["first", "last"]:
            raise ValueError("cascade_at must be one of ('first','last')")
        if policy is None:
            return self._obtain_batch_bandit_feedback_random(
                n_batches, users_per_batch, seed, cascade, cascade_at, lightweight
            )
        else:
            return self._obtain_batch_bandit_feedback_on_policy(
                policy,
                n_batches,
                users_per_batch,
                seed,
                cascade,
                cascade_at,
                lightweight,
            )

    def _obtain_batch_bandit_feedback_random(
        self,
        n_batches: int,
        users_per_batch: int,
        seed: int,
        cascade: bool,
        cascade_at: str,
        lightweight: bool,
    ) -> dict:
        """Generate feedback based on the "ground truth" Deezer user/item features.

        Sample users (with replacement) for each batch, then simulate uniformly random
        policy, where users click actions based on the "ground-truth" click probabilities calculated
        from the user/playlist feature dot-product.

        If cascade is enabled, we don't observe any rewards after max(len_init, last_click_index).

        Parameters
        ----------
        n_batches : int
            How many batches of sampled users to simulate.
        users_per_batch : int
            How many users to sample each batch.
        seed : int
            Random seed.
        cascade : bool
            Whether to simulate cascade logic, where rewards after max(len_init, last_click_index)
            are not observed.
        cascade_at : str
            One of "first" or "last" indicating if we want to stop observing at the first or
            last clicked item.
        lightweight : bool
            Only return the feedback relevant for reward analysis: segments, batches, reward,
            n_rounds, and n_actions.

        Returns
        -------
        dict
            A feedback dict containing:
                action : np.array (n_rounds,)
                    Array of actions presented.
                reward : np.array (n_rounds,)
                    Array of rewards returned.
                position : np.array (n_rounds,)
                    Array of the position of each action.
                context : np.array (n_rounds, 97)
                    Array of the user context for each round.
                action_context : np.array (n_rounds, 97)
                    Array of the action context for each round.
                pscore : np.array (n_rounds,)
                    Array of the probability of choosing this action (always = 1 / n_actions).
                n_rounds : int
                    Number of users per batch * number of batches * number of items observed for that
                    user's session.
                n_actions : int
                    Number of available items (always 862).
                users : np.array (n_rounds * users_per_round,)
                    The users that were generated for this simulation.
                batches : np.array (n_rounds,)
                    The batch number corresponding to each round.
                segments : np.array (n_rounds,)
                    The segment according to each round.

        """
        rng = np.random.default_rng(seed)
        user_indices = rng.choice(
            range(len(self.user_features)),
            size=users_per_batch * n_batches,
            replace=True,
        )
        all_scores = []
        all_item_indices = []

        for user_idx in tqdm(user_indices, desc="Calculating click probabilities"):
            item_indices = rng.choice(
                range(len(self.playlist_features)), size=self.len_list, replace=False
            )
            all_item_indices.append(item_indices)
            this_user_features = self.user_features[user_idx, :]
            these_items_features = self.playlist_features[item_indices, :]
            # calculate raw user-playlist affinities
            raw_scores = these_items_features @ this_user_features
            all_scores.append(raw_scores)

        # convert user/playlist affinities into click probabilities
        all_probabilities = expit(np.array(all_scores))
        all_item_indices = np.array(all_item_indices)

        rewards_uncascaded = rng.binomial(n=1, p=all_probabilities)

        relevant_user_features = self.user_features[user_indices, :]

        relevant_user_segments = self.user_segments[user_indices]

        actions = []
        rewards = []
        positions = []
        context = []
        segments = []
        batches = []

        for row in tqdm(range(rewards_uncascaded.shape[0]), desc="Generating feedback"):
            positions_to_observe = self._get_positions_to_observe(
                rewards_uncascaded[row], cascade, cascade_at
            )

            for position in range(positions_to_observe):
                reward = rewards_uncascaded[row, position]
                rewards.append(reward)
                positions.append(position)
                actions.append(all_item_indices[row, position])
                context.append(relevant_user_features[row])
                segments.append(relevant_user_segments.iloc[row])
                batches.append(row // users_per_batch)

        rewards = np.array(rewards)
        segments = np.array(segments)
        n_rounds = len(actions)
        batches = np.array(batches)
        if not lightweight:
            actions = np.array(actions)
            positions = np.array(positions)
            context = np.array(context)
            action_context = self.playlist_features[actions, :]
            pscore = np.array([1 / self.playlist_features.shape[0] for i in actions])
            return {
                "action": actions,
                "reward": rewards,
                "position": positions,
                "context": context,
                "action_context": action_context,
                "pscore": pscore,
                "n_rounds": n_rounds,
                "n_actions": self.n_actions,
                "users": user_indices,
                "segments": segments,
                "batches": batches,
            }
        else:
            return {
                "reward": rewards,
                "n_rounds": n_rounds,
                "n_actions": self.n_actions,
                "segments": segments,
                "batches": batches,
            }

    def _obtain_batch_bandit_feedback_on_policy(
        self,
        policy: BaseContextFreePolicy,
        n_batches: int,
        users_per_batch: int,
        seed: int,
        cascade: bool,
        cascade_at: str,
        lightweight: bool,
    ) -> dict:
        """Generate feedback based on the "ground truth" Deezer user/item features.

        Sample users (with replacement) for each batch, then simulate uniformly random
        policy, where users click actions based on the "ground-truth" click probabilities calculated
        from the user/playlist feature dot-product.

        If cascade is enabled, we don't observe any rewards after max(len_init, last_click_index).

        Parameters
        ----------
        policy : Union[BaseContextFreePolicy, BaseContextualPolicy]
            Policy that will be used for online simulation.
        n_batches : int
            How many batches of sampled users to simulate.
        users_per_batch : int
            How many users to sample each batch.
        seed : int
            Random seed.
        cascade : bool
            Whether to simulate cascade logic, where rewards after max(len_init, last_click_index)
            are not observed.
        cascade_at : str
            One of "first" or "last" indicating if we want to stop observing at the first or
            last clicked item.
        lightweight : bool
            Only return the feedback relevant for reward analysis: segments, batches, reward,
            n_rounds, n_actions, and policy.

        Returns
        -------
        dict
            A feedback dict containing:
                action : np.array (n_rounds,)
                    Array of actions presented.
                reward : np.array (n_rounds,)
                    Array of rewards returned.
                position : np.array (n_rounds,)
                    Array of the position of each action.
                context : np.array (n_rounds, 97)
                    Array of the user context for each round.
                action_context : np.array (n_rounds, 97)
                    Array of the action context for each round.
                pscore: NoneType
                    N/A.
                n_rounds : int
                    Number of users per batch * number of batches * number of items observed for that
                    user's session.
                n_actions : int
                    Number of available items (always 862).
                users : np.array (n_rounds * users_per_round,)
                    The users that were generated for this simulation.
                batches : np.array (n_rounds,)
                    The batch number corresponding to each round.
                segments : np.array (n_rounds,)
                    The segment according to each round.

        """
        policy.batch_size = np.inf
        rng = np.random.default_rng(seed)
        user_indices = rng.choice(
            range(len(self.user_features)),
            size=users_per_batch * n_batches,
            replace=True,
        )

        all_probabilities = []
        all_item_indices = []

        actions = []
        rewards = []
        positions = []
        context = []
        segments = []
        selected_actions = []
        batches = []

        for i, user_idx in tqdm(
            enumerate(user_indices),
            desc="Simulating online learning",
            total=len(user_indices),
        ):
            if policy.policy_type == "contextfree":
                item_indices = policy.select_action()
            elif policy.policy_type == "contextual":
                item_indices = policy.select_action(self.user_features[user_idx])
            elif policy.policy_type == "segmented":
                item_indices = policy.select_action(self.user_segments[user_idx])
            all_item_indices.append(item_indices)
            this_user_features = self.user_features[user_idx, :]
            these_items_features = self.playlist_features[item_indices, :]
            # calculate raw user-playlist affinities
            raw_scores = these_items_features @ this_user_features
            probabilities = expit(raw_scores)

            # observe all rewards
            rewards_uncascaded = rng.binomial(n=1, p=probabilities)
            positions_to_observe = self._get_positions_to_observe(
                rewards_uncascaded, cascade, cascade_at
            )

            for position in range(positions_to_observe):
                is_last_position = position == positions_to_observe - 1
                is_end_of_batch = (i + 1) % users_per_batch == 0
                if is_last_position and is_end_of_batch:
                    # sneakily trigger update
                    policy.batch_size = policy.n_trial + 1
                reward = rewards_uncascaded[position]
                action = item_indices[position]
                if policy.policy_type == "contextfree":
                    policy.update_params(action=action, reward=reward)
                elif policy.policy_type == "contextual":
                    policy.update_params(
                        action=action,
                        reward=reward,
                        context=self.user_features[user_idx].reshape(
                            1, self.user_features.shape[1]
                        ),
                    )
                elif policy.policy_type == "segmented":
                    policy.update_params(
                        action=action,
                        reward=reward,
                        segment=self.user_segments[user_idx],
                    )
                rewards.append(reward)
                positions.append(position)
                actions.append(action)
                context.append(self.user_features[user_idx])
                segments.append(self.user_segments[user_idx])
                selected_actions.append(all_item_indices[i])
                batches.append(i // users_per_batch)

        rewards = np.array(rewards)
        segments = np.array(segments, dtype=int)
        n_rounds = len(actions)
        batches = np.array(batches, dtype=int)
        if not lightweight:
            actions = np.array(actions, dtype=int)
            positions = np.array(positions, dtype=int)
            context = np.array(context)
            action_context = self.playlist_features[actions, :]
            selected_actions = np.array(selected_actions, dtype=int)
            return {
                "action": actions,
                "reward": rewards,
                "position": positions,
                "context": context,
                "action_context": action_context,
                "pscore": None,
                "n_rounds": n_rounds,
                "n_actions": self.n_actions,
                "policy": policy,
                "selected_actions": selected_actions,
                "users": user_indices,
                "segments": segments,
                "batches": batches,
            }
        else:
            return {
                "reward": rewards,
                "n_rounds": n_rounds,
                "n_actions": self.n_actions,
                "policy": policy,
                "segments": segments,
                "batches": batches,
            }
