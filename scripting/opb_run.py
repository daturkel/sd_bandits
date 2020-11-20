import sys
import os
import numpy as np
import argparse
from pathlib import Path

#loading the custom deezer datset
#assumes your sd_bandits directory is in your root directory,
#change root_dir if not
root_dir = os.path.join('~', 'sd_bandits/')
sys.path.append(root_dir)

from obp.dataset import OpenBanditDataset
#from sd_bandits.deezer.dataset import DeezerDataset
from obp.policy import BernoulliTS, EpsilonGreedy, Random, LinEpsilonGreedy, LinTS, LinUCB, LogisticEpsilonGreedy, LogisticTS, LogisticUCB

from obp.ope.estimators import DirectMethod, DoublyRobust, DoublyRobustWithShrinkage, InverseProbabilityWeighting, ReplayMethod, SelfNormalizedDoublyRobust, SelfNormalizedInverseProbabilityWeighting, SwitchDoublyRobust, SwitchInverseProbabilityWeighting

from obp.simulator.simulator import run_bandit_simulation

#for loading policies, to be changed to include custom policies
policy_dict = {'BernoulliTS':BernoulliTS,
               'EpsilonGreedy':EpsilonGreedy,
               'Random':Random,
               'LinEpsilonGreedy':LinEpsilonGreedy,
               'LinTS':LinTS,
               'LinUCB':LinUCB,
               'LogisticEpsilonGreedy':LogisticEpsilonGreedy,
               'LogisticTS':LogisticTS,
               'LogisticUCB':LogisticUCB}

estimator_dict = {'DirectMethod':DirectMethod,
                  'DoublyRobust':DoublyRobust,
                  'DoublyRobustWithShrinkage':DoublyRobustWithShrinkage,
                  'InverseProbabilityWeighting':InverseProbabilityWeighting,
                  'ReplayMethod':ReplayMethod,
                  'SelfNormalizedDoublyRobust':SelfNormalizedDoublyRobust,
                  'SelfNormalizedInverseProbabilityWeighting':SelfNormalizedInverseProbabilityWeighting,
                  'SwitchDoublyRobust':SwitchDoublyRobust,
                  'SwitchInverseProbabilityWeighting':SwitchInverseProbabilityWeighting}


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--dataset-name',
                        dest='dataset_name', type=str, default='obd',
                        help='The dataset (either "obd" or "deezer")')
    
    parser.add_argument('--dataset-path',
                        dest='dataset_path', type=str,
                        default=os.path.join(root_dir, 'data/obd'),
                        help='Path pointing to the dataset')
    
    parser.add_argument('--policy',
                        dest='policy', type=str,
                        default='BernoulliTS',
                        help='Policy to use for actions')
    
    parser.add_argument('--estimator',
                        dest='estimator', type=str,
                        default='ReplayMethod',
                        help='Estimator to use for evaluation')
    
    parser.add_argument('--rng-seed',
                        dest='rng_seed', type=int, default=11192020,
                        help='RNG seed for replication')
    
    parser.add_argument('--campaign',
                        dest='campaign', type=str,
                        default='all',
                        help='Campaign of OBD (can be all, men, women)')
    
    parser.add_argument('--obd-policy',
                        dest='obd_policy', type=str,
                        default='random',
                        help='Policy used to generate OBD (random or bts)')
    
    parser.add_argument('--user-features',
                        dest='user_features', type=str,
                        default='user_features_small.csv',
                        help='Deezer user features file')

    parser.add_argument('--playlist-features',
                        dest='playlist_features', type=str,
                        default='playlist_features.csv',
                        help='Deezer playlist features file')
        
    return parser.parse_args(args)

if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])

    #load the dataset (obd or deezer)
    print("Loading dataset")
    if params.dataset_name=='obd':
        dataset = OpenBanditDataset(data_path=Path(params.dataset_path),
                                    campaign=params.campaign,
                                    behavior_policy=params.obd_policy)
    elif params.dataset_name=='deezer':
        user_features = os.path.join(params.dataset_path, params.user_features)
        playlist_features = os.path.join(params.dataset_path, params.playlist_features)
        dataset = DeezerDataset(user_features=user_features,
                                playlist_feature=playlist_features)
    else:
        print('Invalid dataset name "{}"'.format(params.dataset_name))
    
    #grab logged bandit feedback from dataset
    feedback = dataset.obtain_batch_bandit_feedback()
    
    #build policy
    print("Constructing Policy")
    policy_constructor = policy_dict[params.policy]
    
    #TODO add modularity for kwargs from argparse for hyperparameter adjustments
    policy = policy_constructor(n_actions=dataset.n_actions,
                                len_list = dataset.len_list)
    
    #compute action probabilities
    actions = run_bandit_simulation(bandit_feedback=feedback,
                                    policy=policy)

    #evaluate policy
    #TODO add modularity for kwargs for argparse for estimator adjustments
    print("Evaluating results")
    estimator = estimator_dict[params.estimator]()
    results = estimator.estimate_interval(reward=feedback["reward"],
                                          action=feedback["action"],
                                          position=feedback["position"],
                                          action_dist=actions)
    
    ground_truth_mean = feedback["reward"].mean()
    print('Ground truth mean value: {}'.format(ground_truth_mean))
    print('{} policy mean value: {}'.format(params.policy, results['mean']))
    
    #TODO Dump results to file