from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_statel2 import CostStatel2
from gps.algorithm.cost.cost_linwp import CostLinWP
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd, init_demo
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, ACTIVATIONS, SENSORDATA
from gps.gui.config import generate_experiment_info
from gps.utility.data_logger import DataLogger
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf

import struct
import IPython
import copy
SENSOR_DIMS = {
    JOINT_ANGLES: 30,
    JOINT_VELOCITIES: 30,
    ACTION: 40,
    ACTIVATIONS: 40,
}

PR2_GAINS = np.ones(40)

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/Adroit_jointtrack/'

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 2,
    'train_conditions': [0],
    'test_conditions': [1],
}

# Demos ======================================================
init_rod = [np.array([0 ,0.5 ,0 ,0 ,0 , 00.95]),
            np.array([0 ,0.5 ,0 ,0 ,0 , 00.75]),
            np.array([0 ,0.5 ,0 ,0 ,0 , 00.35]),
            np.array([0 ,0.5 ,0 ,0 ,0 , 00.00]),
            np.array([0 ,0.5 ,0 ,0 ,0 , -0.35]),
            np.array([0 ,0.5 ,0 ,0 ,0 , -0.70]),
            np.array([0 ,0.5 ,0 ,0 ,0 , -0.95]),
            np.array([0 ,0.5 ,0 ,0 ,0 , -1.30]),
            np.array([0 ,0.5 ,0 ,0 ,0 , -1.65]),
            np.array([0 ,0.5 ,0 ,0 ,0 , -2.00])]

key_id = 1
#matfilenames = ['teleOpPickUp_noGround_180Coverage/key0(7006).mat']
#matfilenames = ['teleOpPickUp_noGround_180Coverage/key0(37370).mat']
#matfilenames = ['teleOpPickUp_noGround_180Coverage/key0(45547).mat']
matfilenames = ['teleOpPickUp_noGround_180Coverage/key1(11570).mat']
#matfilenames = ['teleOpPickUp_noGround_180Coverage/key1(16612).mat']
#matfilenames = ['teleOpPickUp_noGround_180Coverage/key1(19170).mat']

demos = []
demo_ctrls = []
maxlen = -1000000

x0_inits = []


import scipy.io as sio
for ctrl_file in matfilenames:
    demo_dict = sio.loadmat(ctrl_file)
    demo_jt = demo_dict['state'][::15, :]
    maxlen = max(maxlen, len(demo_jt))
    x0_inits.append(demo_jt[0])
    demos.append(demo_jt)

import IPython
IPython.embed()
shape2 = 100
shapectrl = 40
demos_reshaped = []
for demo_jt in demos:
    time_len = len(demo_jt)
    demo_jt_new = np.zeros((maxlen, shape2))
    demo_jt_new[:time_len,:] = demo_jt
    demo_jt_new[time_len:, :] = demo_jt[-1]
    demos_reshaped.append(demo_jt_new)
demos = demos_reshaped

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

x0_inits = []
for demo in demos:
    x0_init = np.zeros((100,))
    x0_init = demo[0]
    x0_init[24:30] = init_rod[key_id]
    x0_inits.append(x0_init)


x0_init_full = copy.copy(x0_inits) + copy.copy(x0_inits) 
import IPython
IPython.embed()
agent = {
    'type': AgentMuJoCo,
    'filename': './teleOpPickUp_noGround_180Coverage/Adroit_TPpneHand(teleOp)131.xml',
    'x0': x0_init_full,
    'dt': 0.03,
    'substeps': 15,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'T': maxlen,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'smooth_noise': True,
    'smooth_noise_sigma': 1.0,
    'smooth_noise_renormalize': True
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'iterations': 25,
    'kl_step': 2.0,
    'max_step_mult': 5.0,
    'min_step_mult': 0.05,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1
}


algorithm['init_traj_distr'] = [{
    'type': init_pd,
    'init_var': 1.0,
    'pos_gains': 1.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
    'x0': copy.copy(x0_inits[i]),
    'dU': SENSOR_DIMS[ACTION],
} for i in common['train_conditions']]

demo_costs = []
tube_costs = []
for demo_jt in demos:
    cost_tgt_ja = copy.copy(demo_jt[:, :30])
    cost_wt_ja = np.ones((30,))


    state_cost = {
        'type': CostStatel2,
        'l1': 0.0,
        'l2': 10.0,
        'alpha': 1e-5,
        'data_types': {
            JOINT_ANGLES: {
                'target_state': cost_tgt_ja,
                'wp': cost_wt_ja,
            }
        },
    }

    cost_tgt_ja_tube = np.ones((agent['T'],30)) #copy.copy(demo[:, :30])
    cost_wt_ja_tube = np.zeros((30,))
    cost_wt_ja_tube[26] = 1000

    tube_cost = {
        'type': CostStatel2,
        'l1': 0.0,
        'l2': 10.0,
        'alpha': 1e-5,
        'data_types': {
            JOINT_ANGLES: {
                'target_state': cost_tgt_ja_tube,
                'wp': cost_wt_ja_tube,
            }
        },
    }
    demo_costs.append(state_cost)
    tube_costs.append(tube_cost)

algorithm['cost'] = [{
    'type': CostSum,
    'costs': [demo_costs[i], tube_costs[i]],
    'weights': [1.0, 1.0],
} for i in common['train_conditions']]

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 40,
        'min_samples_per_cluster': 40,
        'max_samples': 50,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}


algorithm['policy_prior'] = {
    'type': PolicyPrior,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 10,
    'verbose_trials': 10,
    'verbose_policy_trials': 2,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
