from collections import defaultdict
import logging
from time import perf_counter
from typing import List, Optional, Union

import numpy as np
from obp.dataset.base import BaseRealBanditDataset
from obp.dataset.real import OpenBanditDataset
from obp.ope.estimators import BaseOffPolicyEstimator
from obp.ope import RegressionModel
from obp.policy.base import BaseContextFreePolicy, BaseContextualPolicy
from obp.simulator.simulator import run_bandit_simulation
from obp.utils import estimate_confidence_interval_by_bootstrap
import sklearn

from sd_bandits.obp_extensions.dataset import DeezerDataset
from sd_bandits.utils import estimator_args_dict, log_performance


class Experiment:
    def __init__(
        self,
        dataset: BaseRealBanditDataset,
        policies: List[Union[BaseContextFreePolicy, BaseContextualPolicy]],
    ):
        """Base class for to encapsulate experiment logic.

        Parameters
        ----------
        dataset : BaseRealBanditDataset
            The dataset that the experiment will run on.
        policies : List[Union[BaseContextFreePolicy, BaseContextualPolicy]]
            List of policies to be run on this dataset.
        """
        self.dataset = dataset
        self.policies = policies

        self.policy_feedback = defaultdict(dict)
        self.reward_summary = defaultdict(dict)

    def run_experiment(self):
        logging.info("Running experiment")
        tic = perf_counter()
        result = self._run_experiment()
        toc = perf_counter()
        logging.info(f"Experiment finished in {round(toc - tic, 2)} seconds")
        return result

    def _run_experiment(self):
        raise NotImplementedError

    @property
    def output(self) -> dict:
        """
        Return a subset of the experiment contents for pickling.
        """
        policy_feedback = {}
        for policy_name in self.policy_feedback:
            this_policy_feedback = {}
            for key, value in self.policy_feedback[policy_name].items():
                if key in ["segments", "batches"]:
                    this_policy_feedback[key] = value.astype("int8")
                elif key in ["reward", "n_rounds", "n_actions"]:
                    this_policy_feedback[key] = value
            policy_feedback[policy_name] = this_policy_feedback

        return {
            "policy_feedback": policy_feedback,
            "policies": self.policies,
            "reward_summary": self.reward_summary,
        }


class OBDExperiment(Experiment):
    def __init__(
        self,
        dataset: OpenBanditDataset,
        policies: List[Union[BaseContextFreePolicy, BaseContextualPolicy]],
        estimators: List[BaseOffPolicyEstimator],
        regression_base_model: Optional[sklearn.base.BaseEstimator] = None,
    ):
        """Class to encapsulate OBD experiments, which perform offline learning and estimation.

        Parameters
        ----------
        dataset : BaseRealBanditDataset
            The dataset that the experiment will run on.
        policies : List[Union[BaseContextFreePolicy, BaseContextualPolicy]]
            List of policies to be run on this dataset.
        estimator : BaseOffPolicyEstimator
            Off-policy estimator to evaluate the policies with.
        regression_base_model : Optional[sklearn.base.BaseEstimator]
            If needed, a regression base model that will be used by the OPE.
        """
        super().__init__(dataset, policies)
        self.estimators = estimators
        self.estimator_required_args = [
            estimator_args_dict[type(estimator).__name__]
            for estimator in self.estimators
        ]
        self.regression_base_model = regression_base_model

        self.regression_model: Optional[RegressionModel] = None
        self.estimated_rewards_by_reg_model: Optional[np.array] = None

    @log_performance
    def obtain_feedback(self):
        """
        Get logged feedback from baseline dataset.

        """
        logging.info("Obtaining logged feedback")
        self.policy_feedback["logged"] = self.dataset.obtain_batch_bandit_feedback()

    @log_performance
    def fit_regression(self):
        """
        If a regression_base_model was provided at instantiation, fit that base
        model to the logged feedback so it can be used for estimators which
        require it.

        """
        logging.info("Fitting regression model")
        self.regression_model = RegressionModel(
            self.regression_base_model,
            n_actions=self.dataset.n_actions,
            len_list=self.dataset.len_list,
            action_context=self.dataset.action_context,
        )
        self.estimated_rewards_by_reg_model = self.regression_model.fit_predict(
            context=self.policy_feedback["logged"]["context"],
            action=self.policy_feedback["logged"]["action"],
            reward=self.policy_feedback["logged"]["reward"],
            position=self.policy_feedback["logged"]["position"],
            pscore=self.policy_feedback["logged"]["pscore"],
            n_folds=3,
            random_state=0,
        )

    @log_performance
    def run_simulations(self):
        """
        For each provided policy, simulate learning it off of logged feedback.

        """
        logging.info("Running simulations")
        for i, (policy, _) in enumerate(self.policies):
            logging.info(
                f"[{i + 1} of {len(self.policies)}] Running simulation for {policy.policy_name}"
            )
            self.policy_feedback[policy.policy_name]["action"] = run_bandit_simulation(
                self.policy_feedback["logged"], policy
            )

    @log_performance
    def estimate_rewards(self):
        """
        For logged baseline and for simulated policies, calculate a confidence
        interval of expected rewards under that policy. For the simulated policies,
        use provided offline policy estimator.

        """
        logging.info("Estimating rewards")
        for e, estimator in enumerate(self.estimators):
            logging.info(
                f"[{e + 1} of {len(self.estimators)}] Estimator {type(estimator).__name__}"
            )
            for i, (policy_name, feedback) in enumerate(self.policy_feedback.items()):
                if policy_name == "logged":
                    logging.info(
                        f"  [{i + 1} of {len(self.policy_feedback)}] Estimating reward confidence interval for {policy_name}"
                    )
                    reward = feedback["reward"]
                    self.reward_summary[
                        policy_name
                    ] = estimate_confidence_interval_by_bootstrap(
                        reward,
                        n_bootstrap_samples=100,
                        random_state=10,
                    )
                else:
                    logging.info(
                        f"  [{i + 1} of {len(self.policy_feedback)}] Estimating rewards and reward confidence interval for {policy_name}"
                    )
                    est_args = {}
                    for key in self.estimator_required_args[e]:
                        # use the action dist when available (e.g. for baseline), else use actions (e.g. for estimated policies)
                        if key == "action_dist":
                            est_args[key] = feedback["action"]
                        elif key == "estimated_rewards_by_reg_model":
                            est_args[key] = self.estimated_rewards_by_reg_model
                        else:
                            est_args[key] = self.policy_feedback["logged"][key]
                    reward = estimator._estimate_round_rewards(**est_args)
                    if "reward" not in self.policy_feedback[policy_name]:
                        self.policy_feedback[policy_name]["reward"] = {}
                    self.policy_feedback[policy_name]["reward"][
                        type(estimator).__name__
                    ] = reward

                    self.reward_summary[policy_name][
                        type(estimator).__name__
                    ] = estimate_confidence_interval_by_bootstrap(
                        reward,
                        n_bootstrap_samples=100,
                        random_state=10,
                    )

    def _run_experiment(self):
        """
        Hidden function which is called by user-facing run_experiment() method. Runs
        all relevant functions in order.

        """
        self.obtain_feedback()
        if self.regression_base_model is not None:
            self.fit_regression()
        self.run_simulations()
        self.estimate_rewards()


class DeezerExperiment(Experiment):
    def __init__(
        self,
        dataset: DeezerDataset,
        policies: List[Union[BaseContextFreePolicy, BaseContextualPolicy]],
    ):
        """Class to encapsulate Deezer experiments.

        Parameters
        ----------
        dataset : DeezerDataset
            The dataset that the experiment will run on.
        policies : List[Tuple[Union[BaseContextFreePolicy, BaseContextualPolicy],dict]]
            List of policies to be run on this dataset. Each policy is a tuple of the policy
            object and an additional dictionary of options that will be passed to
            obtain_batch_bandit_feedback.
        policy_dict : dict (optional)
            Dictionary containing contents of policy_spec.yaml. Each policy's online_parameters section will be passed to obtain_batch_bandit_feedback.

        """
        super().__init__(dataset, policies)

    @log_performance
    def learn_and_obtain_policy_feedback(self):
        """
        For each provided policy, use Deezer's provided click probabilities to learn
        a policy online and collected feedback as we learn.

        """
        logging.info("Learning and obtaining policy feedback")
        for i, (policy, policy_meta) in enumerate(self.policies):
            logging.info(
                f"[{i + 1} of {len(self.policies)}] Learning and obtaining {policy.policy_name} feedback"
            )
            if policy_meta:
                self.policy_feedback[
                    policy.policy_name
                ] = self.dataset.obtain_batch_bandit_feedback(
                    policy=policy,
                    lightweight=True,
                    **policy_meta,
                )
            else:
                self.policy_feedback[
                    policy.policy_name
                ] = self.dataset.obtain_batch_bandit_feedback(
                    policy=policy, lightweight=True
                )

    @log_performance
    def get_policy_rewards(self):
        """
        For random baseline and for each policy, calculate a confidence interval of the
        expected reward.
        """
        logging.info(
            "Estimating reward confidence interval for random baseline feedback"
        )
        for i, (policy, _) in enumerate(self.policies):
            logging.info(
                f"[{i + 1} of {len(self.policies)}] Estimating reward confindence interval for {policy.policy_name} feedback"
            )
            self.reward_summary[
                policy.policy_name
            ] = estimate_confidence_interval_by_bootstrap(
                self.policy_feedback[policy.policy_name]["reward"],
                n_bootstrap_samples=100,
                random_state=i,
            )

    def _run_experiment(self):
        """
        Hidden function which is called by user-facing run_experiment() method. Runs
        all relevant functions in order.

        """
        self.learn_and_obtain_policy_feedback()
        self.get_policy_rewards()
