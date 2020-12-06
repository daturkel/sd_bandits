# Copied from ZOZO Technologies, Inc. Open Bandit Pipeline
# Modifications made to allow for segment-based policies
# Original link: https://github.com/st-tech/zr-obp/blob/master/obp/simulator/simulator.py
# Original copyright:
# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Bandit Simulator."""
from tqdm import tqdm

import numpy as np

from obp.utils import check_bandit_feedback_inputs, convert_to_action_dist
from obp.types import BanditFeedback, BanditPolicy


def run_bandit_simulation(
    bandit_feedback: BanditFeedback, policy: BanditPolicy
) -> np.ndarray:
    """Run an online bandit algorithm on the given logged bandit feedback data.

    Parameters
    ----------
    bandit_feedback: BanditFeedback
        Logged bandit feedback data used in offline bandit simulation.

    policy: BanditPolicy
        Online bandit policy evaluated in offline bandit simulation (i.e., evaluation policy).

    Returns
    --------
    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities (can be deterministic).

    """
    for key_ in ["action", "position", "reward", "pscore", "context"]:
        if key_ not in bandit_feedback:
            raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")
    check_bandit_feedback_inputs(
        context=bandit_feedback["context"],
        action=bandit_feedback["action"],
        reward=bandit_feedback["reward"],
        position=bandit_feedback["position"],
        pscore=bandit_feedback["pscore"],
    )

    policy_ = policy
    
    # if policy is a segment policy, check if feedback has segments
    # then make sure the size is correct
    # else add a 'segment' feature that is all zeros to feedback
    if "segment" not in bandit_feedback:
        if policy_.policy_type == "segmented":
            raise RuntimeError("Missing segment key in 'bandit_feedback'.")
        else:
            bandit_feedback["segment"] = np.zeros_like(bandit_feedback["action"])

    assert bandit_feedback["segment"].shape[0] == bandit_feedback["action"].shape[0], "feedback must be the same size as action and reward"
        
            
    selected_actions_list = list()
    dim_context = bandit_feedback["context"].shape[1]

    for action_, reward_, position_, context_, segment_ in tqdm(
        zip(
            bandit_feedback["action"],
            bandit_feedback["reward"],
            bandit_feedback["position"],
            bandit_feedback["context"],
            bandit_feedback["segment"]
        ),
        total=bandit_feedback["n_rounds"],
    ):
        # select a list of actions
        if policy_.policy_type == "contextfree":
            selected_actions = policy_.select_action()
        elif policy_.policy_type == "contextual":
            selected_actions = policy_.select_action(context_.reshape(1, dim_context))
        elif policy_.policy_type == "segmented":
            selected_actions = policy_.select_action(segment=segment_)
        action_match_ = action_ == selected_actions[position_]
        # update parameters of a bandit policy
        # only when selected actions&positions are equal to logged actions&positions
        if action_match_:
            if policy_.policy_type == "contextfree":
                policy_.update_params(action=action_, reward=reward_)
            elif policy_.policy_type == "contextual":
                policy_.update_params(
                    action=action_,
                    reward=reward_,
                    context=context_.reshape(1, dim_context),
                )
            elif policy_.policy_type == "segmented":
                policy_.update_params(
                    action=action_,
                    reward=reward_,
                    segment=segment_,
                )
        selected_actions_list.append(selected_actions)

    action_dist = convert_to_action_dist(
        n_actions=bandit_feedback["action"].max() + 1,
        selected_actions=np.array(selected_actions_list),
    )
    return action_dist
