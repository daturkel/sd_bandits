import sys
import os
import numpy as np
import argparse
from pathlib import Path
import yaml
from obp.simulator.simulator import run_bandit_simulation
from obp.ope.regression_model import RegressionModel
from sklearn.linear_model import Ridge

# loading the custom deezer datset
# assumes your sd_bandits directory is in your root directory,
# change root_dir if not
root_dir = os.path.expanduser("~/sd_bandits")
sys.path.append(root_dir)
import sd_bandits.utils as utils

regression_model_args = [
    "context",
    "action",
    "reward",
    "pscore",
    "position",
    "action_dist",
]

estimator_args_dict = {
    "DirectMethod": ["position", "action_dist", "estimated_rewards_by_reg_model"],
    "DoublyRobust": [
        "reward",
        "action",
        "position",
        "pscore",
        "action_dist",
        "estimated_rewards_by_reg_model",
    ],
    "DoublyRobustWithShrinkage": [
        "reward",
        "action",
        "position",
        "pscore",
        "action_dist",
        "estimated_rewards_by_reg_model",
    ],
    "InverseProbabilityWeighting": [
        "reward",
        "action",
        "position",
        "pscore",
        "action_dist",
    ],
    "ReplayMethod": ["reward", "action", "position"],
    "SelfNormalizedDoublyRobust": [
        "reward",
        "action",
        "position",
        "pscore",
        "action_dist",
        "estimated_rewards_by_reg_model",
    ],
    "SelfNormalizedInverseProbabilityWeighting": [
        "reward",
        "action",
        "position",
        "pscore",
        "action_dist",
    ],
    "SwitchDoublyRobust": [
        "reward",
        "action",
        "position",
        "pscore",
        "action_dist",
        "estimated_rewards_by_reg_model",
    ],
    "SwitchInverseProbabilityWeighting": [
        "reward",
        "action",
        "position",
        "pscore",
        "action_dist",
        "estimated_rewards_by_reg_model",
    ],
}


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--experiment-dir",
        dest="experiment_dir",
        type=str,
        help="The directory containing the experiment design",
    )

    parser.add_argument(
        "--rng-seed",
        dest="rng_seed",
        type=int,
        default=11192020,
        help="RNG seed for replication",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    params = process_arguments(sys.argv[1:])

    dataset_spec_path = os.path.join(params.experiment_dir, "dataset_spec.yaml")
    dataset = utils.load_obj_from_spec(dataset_spec_path)

    # grab logged bandit feedback from dataset
    feedback = dataset.obtain_batch_bandit_feedback()

    # build policy
    policy_spec_path = os.path.join(params.experiment_dir, "policy_spec.yaml")
    policy = utils.load_obj_from_spec(policy_spec_path)

    # compute action probabilities
    actions = run_bandit_simulation(bandit_feedback=feedback, policy=policy)

    # evaluate policy
    estimator_spec_path = os.path.join(params.experiment_dir, "estimator_spec.yaml")
    estimator, estimator_dict = utils.load_obj_from_spec(
        estimator_spec_path, return_dict=True
    )

    required_args = estimator_args_dict[estimator_dict["key"]]

    if "estimated_rewards_by_reg_model" in required_args:
        estimated_rewards_model = RegressionModel(
            Ridge(),
            n_actions=feedback["n_actions"],
            action_context=feedback["action_context"],
            len_list=dataset.len_list,
        )
        reg_dict = {}
        for key in regression_model_args:
            if key == "action_dist":
                reg_dict[key] = actions
            else:
                reg_dict[key] = feedback[key]

        estimated_rewards = estimated_rewards_model.fit_predict(**reg_dict)

    est_args = {}
    for key in required_args:
        if key == "action_dist":
            est_args[key] = actions
        elif key == "estimated_rewards_by_reg_model":
            est_args[key] = estimated_rewards
        else:
            est_args[key] = feedback[key]

    # TODO update this for all estimators
    interval_results = estimator.estimate_interval(**est_args)

    ground_truth_mean = feedback["reward"].mean()
    print("Ground truth mean value: {}".format(ground_truth_mean))
    print("Policy mean value: {}".format(interval_results["mean"]))

    with open(os.path.join(params.experiment_dir, "results.yaml"), "w") as file:
        yaml.dump(interval_results, file)
