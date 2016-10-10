""" Hyperparameters for MJC peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_linwp import CostLinWP
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, ACTIVATIONS
from gps.gui.config import generate_experiment_info
import copy
SENSOR_DIMS = {
    JOINT_ANGLES: 30,
    JOINT_VELOCITIES: 30,
    ACTION: 20,
    ACTIVATIONS: 0
}

PR2_GAINS = np.ones(20)

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_example/'


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])
x0_init = np.zeros((60,))
x0_init[1] = -1                                         #wrist
x0_init[[19, 20, 21, 22, 23]] = [.5, 1.2, 0, -.4, -.4]  #thumb base
x0_init[[4, 8, 12, 17]] = [1, 1.1, 1.2, 1]              #finger PIP
x0_init[24:27] = [0.004, 0.45, 0.28]                    #object 
x0_init[[3, 7, 11, 16]] = [.25, .24, .1, 0]             #finger DIP

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/Adroit/Adroit_hand(twirl_jntMotorAct)131.xml',
    'x0': x0_init,
    'dt': 0.05,
    'substeps': 25,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [[]],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'smooth_noise': True,
    'smooth_noise_sigma': 1.0,
    'smooth_noise_renormalize': True
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 25,
    'kl_step': 2.0,
    'max_step_mult': 5.0,
    'min_step_mult': 0.05,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  100*np.ones(20),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': .06,
    'stiffness': 0.0,
    'stiffness_vel': 0.0,
    'final_weight': 10.0, 
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 1e-2 / PR2_GAINS,
}


wp = np.zeros((30,))
wp[0:24]    = 1e-2
wp[0:2]     = 2.0
wp[24:27]   = 1.0
wp[27:30]   = [10, 5, 5]

tgt = copy.copy(x0_init[0:30])
tgt[27:30] = [0, 0, -np.pi/2];

state_cost = {
    'type': CostState,
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'data_types': {
        JOINT_ANGLES: {
            'target_state': tgt,
            'wp': wp,
        },
    },
}


algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, state_cost],
    'weights': [1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 10,
        'min_samples_per_cluster': 40,
        'max_samples': 50,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 5,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)