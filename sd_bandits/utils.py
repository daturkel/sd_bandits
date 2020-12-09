import os
from datetime import datetime
import yaml
import sys
from pathlib import Path

root_dir = os.path.expanduser('~/sd_bandits')
sys.path.append(root_dir)

from obp.policy import BernoulliTS, EpsilonGreedy, Random, LinEpsilonGreedy,\
                       LinTS, LinUCB, LogisticEpsilonGreedy, LogisticTS, LogisticUCB

from obp.ope.estimators import DirectMethod, DoublyRobust, DoublyRobustWithShrinkage,\
                               InverseProbabilityWeighting, ReplayMethod,\
                               SelfNormalizedDoublyRobust, SelfNormalizedInverseProbabilityWeighting,\
                               SwitchDoublyRobust, SwitchInverseProbabilityWeighting

from obp.dataset import OpenBanditDataset
from sd_bandits.obp_extensions.dataset import DeezerDataset

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

dataset_dict = {'obp': OpenBanditDataset,
                'deezer': DeezerDataset}


def build_obj_spec(obj_key, parameter_dict, experiment_name=None, obj_type='policy', output='./policy_yamls/'):
    '''
    Constructs a yaml output file specifying the type of 
    policy and the policy paramters
    
    Parameters
    ------------
    obj_key: str
        The policy/estimator/dataset name
    parameter_dict: dict
        The dict containing parameters for the
        obp object {kwarg: value}
    experiment_name: str
        The associated experiment name, if not
        specified, will be automatically generated
        via timestamp
    obj_type: str
        The type of OBP object, should be 'policy'
        or 'estimator', throw error otherwise
    output: str
        Path to directory to store
    
    Returns
    ------------
    obj_dict: dict
        The constructor dict for the object
    '''
    #Set name via timestamp if not specified
    if obj_type not in ['policy','estimator', 'dataset']:
        print('Invalid type: {}'.format(obj_type))
        return None
    
    if experiment_name==None:
        now = datetime.now()
        current_time = now.strftime("%H%M%S")
        experiment_name = 'experiment_{}'.format(current_time)
    
    #Build dict structure
    obj_dict = {}
    obj_dict['name'] = experiment_name
    obj_dict['type'] = obj_type
    obj_dict['key'] = obj_key
    obj_dict['parameters'] = parameter_dict
    
    #Set output folder
    output_folder = os.path.join(output, experiment_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(os.path.join(output_folder, '{}_spec.yaml'.format(obj_type)), 'w') as file:
        yaml.dump(obj_dict, file)
        
    return obj_dict


def load_obj_from_spec(obj_dict_path, return_dict=False):
    '''
    Loads policy/estimator from spec dict
    
    Parameters
    ------------
    obj_dict_path: str
        Path to configuration dict from build_obj_spec()
    return_dict: bool
        If true, returns the config dict
    Returns
    ------------
    obj: obp.policy/obp.estimator/dataset
        The policy/estimator loaded from the spec dict
    config_dict: dict (optional)
        The spec dict
    '''
    with open(obj_dict_path, 'r') as file:
        obj_dict = yaml.load(file, Loader=yaml.FullLoader)
        
    obj_name = obj_dict['name']
    obj_type = obj_dict['type']
    obj_key = obj_dict['key']
    parameter_dict = obj_dict['parameters']
    
    if obj_type=='policy':
        obj = policy_dict[obj_key](**parameter_dict)
    elif obj_type=='estimator':
        obj = estimator_dict[obj_key](**parameter_dict)
    elif obj_type=='dataset':
        parameter_dict['data_path'] = Path(parameter_dict['data_path'])
        obj = dataset_dict[obj_key](**parameter_dict)
    if return_dict:
        return obj, obj_dict
    else:
        return obj
    

script_shell='''#!/bin/bash
#SBATCH --job-name=obp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --output="{}.out"
#SBATCH --export=NONE

module purge
module load anaconda3/4.3.1

source activate sd_bandits_env

cd ~/sd_bandits/scripting/

python opb_run.py --experiment-dir '{}'
'''

def build_experiment(experiment_name, policy, estimator, dataset, policy_params,
                     estimator_params, dataset_params, output_folder='./policy_yamls/',
                     slurm_output='./outputs/'):
    '''
    Builds full experiment spec folder w/ policy, estimator, and dataset, as well as a
    slurm script
    
    Parameters
    ------------
    experiment_name: str
        Name of the experiment dir
    policy: str
        Which obp policy to use
    estimator: str
        Which obp estimator to use
    dataset: str
        Which dataset to use ('obp' or 'deezer')
    policy_params: dict
        Dict for any parameters to construct the policy
    estimator_params:
        Dict for any parameters to construct the policy
    dataset_params:
        Dict for any parameters to contruct the ataset object
    output_folder: str
        Directory that will contain experiment directory
    Returns
    ------------
    None
    '''
    
    policy_dict = build_obj_spec(policy, policy_params, experiment_name=experiment_name,\
                                 obj_type='policy', output=output_folder)
    estimator_dict = build_obj_spec(estimator, estimator_params, experiment_name=experiment_name,\
                                    obj_type='estimator', output=output_folder)
    dataset_dict = build_obj_spec(dataset, dataset_params, experiment_name=experiment_name,\
                                  obj_type='dataset', output=output_folder)
    
    experiment_dir = os.path.join(output_folder, experiment_name)
    slurm_output = os.path.join(slurm_output, experiment_name+'.out')
    slurm_script = script_shell.format(slurm_output, experiment_dir)
    
    with open(os.path.join(experiment_dir, 'script.sbatch'),'w') as file:
        file.write(slurm_script)
        
