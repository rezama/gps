""" Hyperparameters for MJC peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.Adroit.agent_adroit import AgentAdroit
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
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
import struct
import IPython
import copy
SENSOR_DIMS = {
    JOINT_ANGLES: 30,
    JOINT_VELOCITIES: 30,
    ACTION: 40,
    ACTIVATIONS: 40,
}

def read_demo_txt(filename):
    data =  []
    f = open(filename, 'rb')
    dat = f.read()
    dat_split_bytime = dat.split('\n')
    for dat_curr_time in dat_split_bytime:
        dat_split_bystate = dat_curr_time.split('\t')
        if len(dat_split_bystate)> 1:
            float_state_split_data = [float(i) for i in dat_split_bystate]
            data.append(float_state_split_data)
    return np.asarray(data)

PR2_GAINS = np.ones(40)

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/Adroit_hardware_update/'

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
    #'demo_filename': 'teleOpPickup_hardware/exp2/adroitPickup_c0_t1.state',
    #'demo_ctrl_filename': 'teleOpPickup_hardware/exp2/adroitPickup_c0_t1.control'
    'demo_filename': 'teleOpPickup_hardware/debugWrist1/adroitPickup_c0_t0.state',
    'demo_ctrl_filename': 'teleOpPickup_hardware/debugWrist1/adroitPickup_c0_t0.control'
}
demo = read_demo_txt(common['demo_filename'])[::6, :]
demo[:, 60:] = 1e-6*demo[:, 60:]
demo_ctrl = read_demo_txt(common['demo_ctrl_filename'])[::6, :]
x0_init = demo[0]

print("TIME IS " + str(len(demo)))
agent = {
    'type': AgentAdroit,
    'dt': 0.03,
    'substeps': 15,
    'conditions': common['conditions'],
    'T': len(demo),
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS],
    'smooth_noise': True,
    'smooth_noise_sigma': 1.0,
    'smooth_noise_renormalize': True,
    'x0': x0_init
    'demo_ctrls': [demo_ctrl]
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

# algorithm['init_traj_distr'] = {
#     'type': init_lqr,
#     'init_gains':  100*np.ones(40),
#     'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
#     'init_var': .06,
#     'stiffness': 0.0,
#     'stiffness_vel': 0.0,
#     'final_weight': 10.0, 
#     'dt': agent['dt'],
#     'T': agent['T'],
# }

"""algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 1.0,
    'pos_gains': 1.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
    'x0': x0_init,
    'dU': SENSOR_DIMS[ACTION],
}"""

algorithm['init_traj_distr'] = {
    'type': init_demo,
    'init_var': 0.000001,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
    'x0': x0_init,
    'dU': SENSOR_DIMS[ACTION],
    'demo_ctrl': demo_ctrl,
}

torque_cost = {
    'type': CostAction,
    'wu': 1e-2 / PR2_GAINS,
}



cost_tgt_ja = demo[:, :30]
cost_tgt_jv = demo[:, 30:60]
cost_tgt_act = demo[:, 60:]

cost_wt_ja = np.ones((30,))
cost_wt_ja[22] = 10;
cost_wt_jv = np.zeros((30,))
cost_wt_act = np.ones((40,))
cost_wt_act[36] = 10;


state_cost = {
    'type': CostStatel2,
    'l1': 0.0,
    'l2': 10.0,
    'alpha': 1e-5,
    'data_types': {
        JOINT_ANGLES: {
            'target_state': cost_tgt_ja,
            'wp': cost_wt_ja,
        },
        JOINT_VELOCITIES: {
            'target_state': cost_tgt_jv,
            'wp': cost_wt_jv,
        },
        ACTIVATIONS: {
            'target_state': cost_tgt_act,
            'wp': cost_wt_act,
        },
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

algorithm['cost'] = {
    'type': CostSum,
    'costs': [state_cost, tube_cost],
    'weights': [1.0, 1.0],
}

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

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 2,
    'verbose_trials': 2,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'demo_joints': demo,
    'demo_ctrl': demo_ctrl

}

common['info'] = generate_experiment_info(config)
