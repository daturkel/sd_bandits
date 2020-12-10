import argparse
import logging
import os
from pathlib import Path
import pickle
import sys
import yaml

import numpy as np
from obp.dataset.real import OpenBanditDataset
from sklearn.linear_model import LinearRegression

from sd_bandits.experiment import DeezerExperiment, OBDExperiment
from sd_bandits.obp_extensions.dataset import DeezerDataset
import sd_bandits.utils as utils


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--experiment-dir",
        dest="experiment_dir",
        type=str,
        help="The directory containing the experiment design",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
        datefmt="%-I:%M:%S",
    )

    params = process_arguments(sys.argv[1:])

    logging.info("Building dataset")
    dataset_spec_path = os.path.join(params.experiment_dir, "dataset_spec.yaml")
    dataset, _ = utils.load_obj_from_spec(dataset_spec_path, "dataset")

    logging.info("Building policies")
    policy_spec_path = os.path.join(params.experiment_dir, "policy_spec.yaml")
    policies = utils.load_obj_from_spec(
        policy_spec_path,
        "policy",
        extra_parameters={"n_actions": dataset.n_actions, "len_list": dataset.len_list},
    )

    if os.path.exists(os.path.join(params.experiment_dir, "estimator_spec.yaml")):
        logging.info("Building estimators")
        estimator_spec_path = os.path.join(params.experiment_dir, "estimator_spec.yaml")
        estimator, _ = utils.load_obj_from_spec(estimator_spec_path, "estimator")

    if isinstance(dataset, DeezerDataset):
        experiment = DeezerExperiment(dataset, policies)
    else:
        experiment = OBDExperiment(dataset, policies, estimator, LinearRegression())

    experiment.run_experiment()

    logging.info(
        f"Writing output to results {os.path.join(params.experiment_dir,'results.pickle')}"
    )
    with open(os.path.join(params.experiment_dir, "results.pickle"), "wb") as file:
        pickle.dump(experiment.output, file)
    logging.info("Bye!")
