import sys
import os
import numpy as np
import argparse
from pathlib import Path
from obp.simulator.simulator import run_bandit_simulation

#loading the custom deezer datset
#assumes your sd_bandits directory is in your root directory,
#change root_dir if not
root_dir = os.path.expanduser('~/sd_bandits')
sys.path.append(root_dir)
import sd_bandits.utils as utils



def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--experiment-dir',
                        dest='experiment_dir', type=str, ,
                        help='The directory containing the experiment design')
    
    parser.add_argument('--rng-seed',
                        dest='rng_seed', type=int, default=11192020,
                        help='RNG seed for replication')
    
        
    return parser.parse_args(args)

if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])

    dataset_spec_path = os.path.join(params.experiment_dir, 'dataset_spec.yaml')
    dataset = load_obj_from_spec(datset_spec_path)
    
    #grab logged bandit feedback from dataset
    feedback = dataset.obtain_batch_bandit_feedback()
    
    #build policy
    policy_spec_path = os.path.join(params.experiment_dir, 'policy_spec.yaml')
    policy = load_obj_from_spec(policy_spec_path)
    
    #compute action probabilities
    actions = run_bandit_simulation(bandit_feedback=feedback,
                                    policy=policy)

    #evaluate policy
    estimator_spec_path = os.path.join(params.experiment_dir, 'estimator_spec.yaml')
    estimator = load_obj_from_spec(estimator_spec_path)
    
    
    #TODO update this for all estimators
    interval_results = estimator.estimate_interval(reward=feedback["reward"],
                                          action=feedback["action"],
                                          position=feedback["position"],
                                          action_dist=actions)
    
    ground_truth_mean = feedback["reward"].mean()
    print('Ground truth mean value: {}'.format(ground_truth_mean))
    print('{} policy mean value: {}'.format(params.policy, results['mean']))
    
    #TODO Dump results to file