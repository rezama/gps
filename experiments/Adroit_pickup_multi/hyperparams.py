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
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
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
    SENSORDATA: 5
}

PR2_GAINS = np.ones(40)

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/Adroit_pickup_multi/'

def load_demo(filename):
    with open(filename, mode='rb') as file:
        fileContent = file.read()
    headers = struct.unpack('iiiiii', fileContent[:24])
    nq = headers[0]
    nv = headers[1]
    nu = headers[2]
    nmocap = headers[3]
    nsensordata = headers[4]
    name_len = headers[5]
    name = struct.unpack(str(name_len) + 's', fileContent[24:24+name_len])[0]
    rem_size = len(fileContent[24 + name_len:])
    num_floats = int(rem_size/4)
    dat = np.asarray(struct.unpack(str(num_floats) + 'f', fileContent[24+name_len:]))
    recsz = 1 + nq + nv + nu + 7*nmocap + nsensordata
    if rem_size % recsz != 0:
        print("ERROR")
    else:
        dat = np.reshape(dat, (len(dat)/recsz, recsz))
        dat = dat.T

    skipamount = 15
    time = dat[0,:][::skipamount] - dat[0, 0]
    qpos = dat[1:nq + 1, :].T[::skipamount, :]
    qvel = dat[nq+1:nq+nv+1,:].T[::skipamount, :]
    ctrl = dat[nq+nv+1:nq+nv+nu+1,:].T[::skipamount,:]
    mocap_pos = dat[nq+nv+nu+1:nq+nv+nu+3*nmocap+1,:].T[::skipamount, :]
    mocap_quat = dat[nq+nv+nu+3*nmocap+1:nq+nv+nu+7*nmocap+1,:].T[::skipamount, :]
    sensordata = dat[nq+nv+nu+7*nmocap+1:,:].T[::skipamount,:]
    data = {'nq': nq, 'nv':nv, 'nu':nu,'nmocap':nmocap, 'nsensordata':nsensordata, 'name':name, \
            'time': time, 'qpos':qpos, 'qvel':qvel, 'ctrl':ctrl, 'mocap_pos':mocap_pos, 'mocap_quat':mocap_quat, \
            'sensordata':sensordata
           }
    return data


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 4,
    'train_conditions': [0,1,2,3],
    'test_conditions': [],
}

filenames = ['teleOpPickup_noGround_multi/demo1_neg95.mjl',
            'teleOpPickup_noGround_multi/demo1_35.mjl',
            'teleOpPickup_noGround_multi/demo1_neg35.mjl',
            'teleOpPickup_noGround_multi/demo1_0.mjl',
]
translation_offsets = [[np.array([0, -0.5, 0.02])], [np.array([0, -0.5, 0.02])], [np.array([0, -0.5, 0.02])], [np.array([0, -0.5, 0.02])]]
rotation_offsets = [[np.array([ 0.889 ,0 ,0, -0.457])], [np.array([0.985, 0 ,0 , 0.174])], [np.array([0.985, 0 ,0 , -0.174])], [np.array([1.0, 0 ,0 ,0])]]
demos = []
minlen = 100000
for file_to_open in filenames:
    demos.append(load_demo(file_to_open))
    minlen = min(minlen, demos[-1]['time'].shape[0])

fields = ['time', 'qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat', 'sensordata']
for demo in demos:
    for field in fields:
        demo[field] = demo[field][:minlen]

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

x0_inits = []
for file_to_open, demo in zip(filenames, demos):
    x0_init = np.zeros((105,))
    x0_init[:30] = demo['qpos'][0]
    x0_init[30:60] = demo['qvel'][0]
    x0_init[60:100] = 0.1*np.ones((40,))
    x0_init[100:] = demo['sensordata'][0][20:25]
    x0_inits.append(x0_init)

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/Adroit/teleOpPickup_noGround_diff_positions.xml',
    'x0': x0_inits,
    'dt': 0.03,
    'substeps': 15,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'pos_body_idx': [np.array([26])]*common['conditions'],
    'pos_body_offset': translation_offsets,
    'quat_body_offset': rotation_offsets,
    'T': minlen,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS, SENSORDATA],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS, SENSORDATA],
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

# algorithm = {
#     'type': AlgorithmBADMM,
#     'conditions': agent['conditions'],
#     'train_conditions': agent['train_conditions'],
#     'test_conditions': agent['test_conditions'],
#     'num_robots': num_robots,
#     'iterations': 25,
#     'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
#     'policy_dual_rate': 0.2,
#     'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
#     'fixed_lg_step': 3,
#     'kl_step': 5.0,
#     'min_step_mult': 0.01,
#     'max_step_mult': 1.0,
#     'sample_decrease_var': 0.05,
#     'sample_increase_var': 0.1,
#     'init_pol_wt': 0.005,
# }

# algorithm['policy_opt'] = {
#         'type': PolicyOptTf,
#         'network_model': multi_modal_network,
#         'network_params': {
#             'dim_hidden': [10],
#             'num_filters': [10, 20],
#             'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS, SENSORDATA],
#             'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS, SENSORDATA],
#             'obs_image_data':[],
#             'image_width': IMAGE_WIDTH,
#             'image_height': IMAGE_HEIGHT,
#             'image_channels': IMAGE_CHANNELS,
#             'sensor_dims': SENSOR_DIMS,
#             'batch_size': 25,
#         },
#         'iterations': 4000,
#         'fc_only_iterations': 5000,
#         'checkpoint_prefix': EXP_DIR + 'data_files/policy'
# }

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 1.0,
    'pos_gains': 1.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
    'x0': x0_init,
    'dU': SENSOR_DIMS[ACTION],
}

torque_cost = {
    'type': CostAction,
    'wu': 1e-2 / PR2_GAINS,
}

state_costs = []
lift_costs = []
<<<<<<< HEAD

=======
import IPython
IPython.embed()
>>>>>>> e30fbf7d0b0ec1d5de44e113951d938f46529c70
for demo in demos:
    cost_tgt = copy.copy(demo['qpos'])
    cost_tgt[:, :24] = copy.copy(demo['ctrl'])
    cost_wt = np.concatenate([np.ones((24,)), 10*np.ones((6,))])
    cost_wt[26] = 100

    state_cost = {
        'type': CostState,
        'l1': 0.0,
        'l2': 10.0,
        'alpha': 1e-5,
        'data_types': {
            JOINT_ANGLES: {
                'target_state': cost_tgt,
                'wp': cost_wt,
            },
        },
    }

    cost_tgt_lift = np.zeros(demo['qpos'].shape)
    cost_tgt_lift[26] = 0.2
    cost_wt_lift = np.zeros(30)
    cost_wt_lift[26] = 100

    lift_cost = {
        'type': CostState,
        'l1': 0.0,
        'l2': 10.0,
        'alpha': 1e-5,
        'data_types': {
            JOINT_ANGLES: {
                'target_state': cost_tgt_lift,
                'wp': cost_wt_lift,
            },
        },
    }
    state_costs.append(state_cost)
    lift_costs.append(lift_cost)

algorithm['cost'] = [{
    'type': CostSum,
    'costs': [state_costs[i], lift_costs[i]],
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

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 10,
    'verbose_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
