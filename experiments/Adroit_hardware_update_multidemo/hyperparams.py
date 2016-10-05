""" Hyperparameters for MJC peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.Adroit.agent_adroit import AgentAdroit
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
EXP_DIR = BASE_DIR + '/../experiments/Adroit_hardware_update_multidemo/'

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 4,
    'train_conditions': [0, 1, 2, 3],
    'test_conditions':[],
}

#joint_filenames = [ 'teleOpPickup_hardware/network/adroitPickup_c3_t4.state',
#                    'teleOpPickup_hardware/network/adroitPickup_c3_t5.state'
#]
#
#ctrl_filenames = [  'teleOpPickup_hardware/network/adroitPickup_c3_t4.control',
#                    'teleOpPickup_hardware/network/adroitPickup_c3_t5.control'
#]


joint_filenames = [ 'teleOpPickup_hardware/network/adroitPickup_c0_t0.state',
                    'teleOpPickup_hardware/network/adroitPickup_c1_t1.state',
                    'teleOpPickup_hardware/network/adroitPickup_c2_t0.state',
                    'teleOpPickup_hardware/network/adroitPickup_c4_t0.state'
]

ctrl_filenames = [  'teleOpPickup_hardware/network/adroitPickup_c0_t0.control',
                    'teleOpPickup_hardware/network/adroitPickup_c1_t1.control',
                    'teleOpPickup_hardware/network/adroitPickup_c2_t0.control',
                    'teleOpPickup_hardware/network/adroitPickup_c4_t0.control'
]

demos = []
demo_ctrls = []
maxlen = -1000000
x0_inits = []
for jt_file, ctrl_file in zip(joint_filenames, ctrl_filenames):
    demo_jt = read_demo_txt(jt_file)[::6, :]
    demo_jt[:, 60:] = 1e-6*demo_jt[:, 60:]
    demo_ctrl = read_demo_txt(ctrl_file)[::6, :]
    maxlen = max(maxlen, len(demo_jt))
    x0_inits.append(demo_jt[0])
    demos.append(demo_jt)
    demo_ctrls.append(demo_ctrl)
#adjust to same length
shape2 = 100
shapectrl = 40
demos_reshaped = []
demo_ctrls_reshaped = []
for demo_jt, demo_ctrl in zip(demos, demo_ctrls):
        time_len = len(demo_jt)
        demo_jt_new = np.zeros((maxlen, shape2))
        demo_jt_new[:time_len,:] = demo_jt
        demo_jt_new[time_len:, :] = demo_jt[-1]
        time_len_ctrl = len(demo_ctrl)
        if time_len_ctrl != time_len:
            print("BUG")
            import IPython
            IPython.embed()
        demo_ctrl_new = np.zeros((maxlen, shapectrl))
        demo_ctrl_new[:time_len,:] = demo_ctrl
        demo_ctrl_new[time_len:, :] = demo_ctrl[-1]
        demos_reshaped.append(demo_jt_new)
        demo_ctrls_reshaped.append(demo_ctrl_new)
demos = demos_reshaped
demo_ctrls = demo_ctrls_reshaped

print("TIME IS " + str(maxlen))
agent = {
    'type': AgentAdroit,
    'dt': 0.03,
    'substeps': 15,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions':common['test_conditions'],
    'T': maxlen,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS],
    'smooth_noise': True,
    'smooth_noise_sigma': 1.0,
    'smooth_noise_renormalize': True,
    'x0': x0_inits,
    'demo_ctrl': demo_ctrls
}

algorithm = {
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'iterations': 50,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.1,
    'init_pol_wt': 0.002,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 2.0,
    'min_step_mult': 0.05,
    'max_step_mult': 5.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'policy_sample_mode': 'replace'
}

algorithm['init_traj_distr'] = [{
    'type': init_demo,
    'init_var': 0.0001,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
    'x0': x0_inits[i],
    'dU': SENSOR_DIMS[ACTION],
    'demo_ctrl': demo_ctrls[i],
} for i in common['train_conditions']]

demo_costs = []
tube_costs = []
for demo_jt in demos:
    cost_tgt_ja = copy.copy(demo_jt[:, :30])
    cost_tgt_jv = copy.copy(demo_jt[:, 30:60])
    cost_tgt_act = copy.copy(demo_jt[:, 60:])

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

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS],
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': example_tf_network,
    'iterations': 20000,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPrior,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 3,
    'verbose_trials': 2,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
