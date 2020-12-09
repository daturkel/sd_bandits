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

        self.actions = {}
        self.rewards = {}

    def run_experiment(self):
        tic = perf_counter()
        result = self._run_experiment()
        toc = perf_counter()
        logging.info(f"Experiment finished in {round(toc - tic, 2)} seconds")
        return result

    def _run_experiment(self):
        raise NotImplementedError


class OBDExperiment(Experiment):
    def __init__(
        self,
        dataset: OpenBanditDataset,
        policies: List[Union[BaseContextFreePolicy, BaseContextualPolicy]],
        estimator: BaseOffPolicyEstimator,
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
        self.estimator = estimator
        self.estimator_required_args = estimator_args_dict[
            type(self.estimator).__name__
        ]
        self.regression_base_model = regression_base_model

        self.logged_feedback: Optional[dict] = None
        self.regression_model: Optional[RegressionModel] = None
        self.estimated_rewards: Optional[np.array] = None

    @log_performance
    def obtain_feedback(self):
        logging.info("Obtaining logged feedback")
        self.logged_feedback = self.dataset.obtain_batch_bandit_feedback()

    @log_performance
    def fit_regression(self):
        logging.info("Fitting regression model")
        self.regression_model = RegressionModel(
            self.regression_base_model,
            n_actions=self.dataset.n_actions,
            len_list=self.dataset.len_list,
            action_context=self.dataset.action_context,
        )
        self.estimated_rewards = self.regression_model.fit_predict(
            context=self.logged_feedback["context"],
            action=self.logged_feedback["action"],
            reward=self.logged_feedback["reward"],
            position=self.logged_feedback["position"],
            pscore=self.logged_feedback["pscore"],
            n_folds=3,
            random_state=0,
        )

    @log_performance
    def run_simulations(self):
        logging.info("Running simulations")
        for i, policy in enumerate(self.policies):
            logging.info(
                f"[{i + 1} of {len(self.policies)}] Running simulation for {policy.policy_name}"
            )
            self.actions[policy.policy_name] = run_bandit_simulation(
                self.logged_feedback, policy
            )

    @log_performance
    def estimate_rewards(self):
        logging.info("Estimating rewards")
        logging.info("Estimating reward confidence interval for logged feedback")
        self.rewards["logged"] = estimate_confidence_interval_by_bootstrap(
            self.logged_feedback["reward"],
            n_bootstrap_samples=100,
            random_state=0,
        )

        for i, (policy_name, actions) in enumerate(self.actions.items()):
            est_args = {}
            for key in self.estimator_required_args:
                if key == "action_dist":
                    est_args[key] = self.actions[policy_name]
                elif key == "estimated_rewards_by_reg_model":
                    est_args[key] = self.estimated_rewards
                else:
                    est_args[key] = self.logged_feedback[key]

            logging.info(
                f"[{i + 1} of {len(self.actions)}] Estimating reward confidence interval for {policy_name}"
            )
            self.rewards[policy_name] = self.estimator.estimate_interval(**est_args)

    def _run_experiment(self):
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
        super().__init__(dataset, policies)
        self.random_feedback: Optional[dict] = None
        self.policy_feedback = dict()

    @log_performance
    def obtain_random_feedback(self):
        logging.info("Obtaining random baseline feedback")
        self.dataset: DeezerDataset
        self.random_feedback = self.dataset.obtain_batch_bandit_feedback()

    @log_performance
    def learn_and_obtain_policy_feedback(self):
        logging.info("Learning and obtaining policy feedback")
        for i, policy in enumerate(self.policies):
            logging.info(
                f"[{i + 1} of {len(self.policies)}] Learning and obtaining {policy.policy_name} feedback"
            )
            self.policy_feedback[
                policy.policy_name
            ] = self.dataset.obtain_batch_bandit_feedback(policy=policy)

    @log_performance
    def get_policy_rewards(self):
        logging.info(
            "Estimating reward confidence interval for random baseline feedback"
        )
        self.rewards["random"] = estimate_confidence_interval_by_bootstrap(
            self.random_feedback["reward"], n_bootstrap_samples=100, random_state=0
        )
        for i, policy in enumerate(self.policies):
            logging.info(
                f"[{i + 1} of {len(self.policies)}] Estimating reward confindence interval for {policy.policy_name} feedback"
            )
            self.rewards[
                policy_policy.name
            ] = estimate_confidence_interval_by_bootstrap(
                self.policy_feedback[policy.policy_name],
                n_bootstrap_samples=100,
                random_state=i,
            )

    def _run_experiment(self):
        self.obtain_random_feedback()
        self.learn_and_obtain_policy_feedback()
        self.get_policy_rewards()
