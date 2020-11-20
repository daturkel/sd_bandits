import numpy as np
from obp.policy import BaseContextFreePolicy
from dataclasses import dataclass

@dataclass
class ExploreThenCommit(BaseContextFreePolicy):
    """Explore Then Commit policy.
    Explores every action until each action has been chosen n times,
    then commits to the top len_list actions

    Parameters
    ----------
    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.
    
    min_n: int, default=1
        Minimum number of times to explore each action before commiting
        to one

    random_state: int, default=None
        Controls the random seed in sampling actions.

    policy_name: str, default=f'egreedy_{epsilon}'.
        Name of bandit policy.
        
        
        
    CITE: Used Deezer's ETC implementation. Not an official citation but want to
    put that in here
    """

    min_n: int = 1
    policy_name: str = f"etc{min_n}"
    
    def __post_init__(self) -> None:
        """Initialize Class."""
        assert self.min_n >= 1 and isinstance(
            self.min_n, int
        ), f"min_n must be an integer larger than 1, but {self.min_n} is given"
        super().__post_init__()

    def select_action(self) -> np.ndarray:
        """Select a list of actions.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.

        """
        
        predicted_rewards = self.reward_counts
        random_score = self.random_.random(predicted_rewards.shape) # to break ties
        # get the number of times each action needs to be pulled to get out of the 'explore stage'
        actions_left = np.maximum(np.zeros_like(self.action_counts), self.min_n - self.action_counts)
        # rank the actions based on actions_left, predicted_rewards, and finally, random_score to break ties
        ranked_actions = np.lexsort((random_score, -predicted_rewards, -actions_left ))
        return ranked_actions[: self.len_list]


    def update_params(self, action: int, reward: float) -> None:
        """Update policy parameters.
        Same as obp.policy.EpsilonGreedy

        Parameters
        ----------
        action: int
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.

        """
        self.n_trial += 1
        self.action_counts_temp[action] += 1
        n, old_reward = self.action_counts_temp[action], self.reward_counts_temp[action]
        self.reward_counts_temp[action] = (old_reward * (n - 1) / n) + (reward / n)
        if self.n_trial % self.batch_size == 0:
            self.action_counts = np.copy(self.action_counts_temp)
            self.reward_counts = np.copy(self.reward_counts_temp)
