import numpy as np
from obp.policy import BaseContextFreePolicy
from dataclasses import dataclass
from copy import deepcopy

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

    policy_name: str, default=f'etc_{min_n}'.
        Name of bandit policy.
        
        
        
    CITE: Used Deezer's ETC implementation. Not an official citation but want to
    put that in here
    """

    min_n: int = 1
    policy_name: str = ""
    
    def __post_init__(self) -> None:
        """Initialize Class."""
        assert self.min_n >= 1 and isinstance(
            self.min_n, int
        ), f"min_n must be an integer larger than 1, but {self.min_n} is given"
        if self.policy_name == "":
            self.policy_name = f"etc_{self.min_n}"
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


@dataclass
class SegmentPolicy():
    """Segment Policy.
    Takes any context-free policy and turns it into a segmented policy, given
    the number of different segments in the dataset

    Parameters
    ----------
    base_policy: BaseContextFreePolicy
        A policy object that is a subclass of BaseContextFreePolicy
        Right now, one of Random, EpsilonGreedy, BernoulliTS, or 
        ExploreThenCommit
        
    n_segments: int
        Number of segments, assumes segments are valued 0 to n_segments-1

    policy_name: str, default=f"{self.policy.policy_name}-seg"
        Name of bandit policy.
        
    """
    
    base_policy: BaseContextFreePolicy
    n_segments: int
    policy_name: str = ""
    
    def __post_init__(self) -> None:
        """Initialize Class."""
        assert isinstance(
            self.base_policy, BaseContextFreePolicy
        ), "Supplied policy must be an instance of BaseContextFreePolicy"
        assert isinstance(self.n_segments, int), f"n_segments must be an integer, you supplied a {type(self.n_segments)}"
        
        # then instantiate all of base policies attributes
        self.n_actions = self.base_policy.n_actions
        self.len_list = self.base_policy.len_list
        self.batch_size = self.base_policy.batch_size
        self.random_state = self.base_policy.random_state
        self.policy_type = "segmented"
        self.n_trial = 0
        
        # give name if no name given
        if self.policy_name == "":
            self.policy_name = f'{self.base_policy.policy_name}_seg'
        
        # create dictionary of policy objects
        segment_policy_pairs = [(i, deepcopy(self.base_policy)) for i in range(self.n_segments)]
        self.segment_policies = dict(segment_policy_pairs)
        self.update_dict = dict([(i, []) for i in range(self.n_segments)])
        
    @property
    def action_counts(self):
        """Returns action counts"""
        action_counts = np.zeros(self.n_actions)
        for i in range(self.n_segments):
            action_counts += self.segment_policies[i].action_counts
        return action_counts
    
    @property
    def reward_counts(self):
        """Returns reward counts"""
        reward_counts = np.zeros(self.n_actions)
        for i in range(self.n_segments):
            reward_counts += self.segment_policies[i].action_counts * self.segment_policies[i].reward_counts
        reward_counts = np.divide(reward_counts, self.action_counts, where=self.action_counts != 0)
        return reward_counts
        
        
    def select_action(self, segment: int) -> np.ndarray:
        """Select a list of actions.
        
        Parameters
        segment: int
            The user's segment
        
        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.

        """
        
        return self.segment_policies[segment].select_action()


    def update_params(self, action: int, reward: float, segment: int) -> None:
        """Update policy parameters.
        Same as obp.policy.EpsilonGreedy

        Parameters
        ----------
        action: int
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.
        
        segment: int
            The user's segment

        """
        # first update n_trial for the policy
        self.n_trial += 1
        
        # then add action and reward to the update dictionary
        self.update_dict[segment].append((action, reward))

        # if the policy should update its parameters, call _update_all_params
        if self.n_trial % self.batch_size == 0:
            self._update_all_params()
                    
    def _update_all_params(self) -> None:
        """Updates all segment policy parameters.
        Only triggered when n_trials mod batch_size == 0
        """
        for seg_, action_reward_list in self.update_dict.items():
            # setting each segment policy's batch_size to the number of times
            # it has to update. so it will trigger a full at the end of the 
            # inner for-loop
            self.segment_policies[seg_].batch_size =  len(action_reward_list)
            self.segment_policies[seg_].n_trial = 0
            for (action_, reward_) in action_reward_list:
                self.segment_policies[seg_].update_params(action_, reward_)
                
            # then reset the update dict
            self.update_dict[seg_] = []