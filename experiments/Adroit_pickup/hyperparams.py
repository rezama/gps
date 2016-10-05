""" Hyperparameters for MJC peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
import IPython
IPython.embed()
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_linwp import CostLinWP
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, ACTIVATIONS, SENSORDATA
from gps.gui.config import generate_experiment_info
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.utility.data_logger import DataLogger
from gps.algorithm.policy.policy_prior import PolicyPrior
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
EXP_DIR = BASE_DIR + '/../experiments/Adroit_pickup/'

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
    time = dat[0,:][::skipamount][:100] - dat[0, 0]
    qpos = dat[1:nq + 1, :].T[::skipamount, :][:100,:]
    qvel = dat[nq+1:nq+nv+1,:].T[::skipamount, :][:100,:]
    ctrl = dat[nq+nv+1:nq+nv+nu+1,:].T[::skipamount,:][:100,:]
    mocap_pos = dat[nq+nv+nu+1:nq+nv+nu+3*nmocap+1,:].T[::skipamount, :][:100,:]
    mocap_quat = dat[nq+nv+nu+3*nmocap+1:nq+nv+nu+7*nmocap+1,:].T[::skipamount, :][:100,:]
    sensordata = dat[nq+nv+nu+7*nmocap+1:,:].T[::skipamount,:][:100,:]
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
    'conditions': 1,
    # 'demo_name':'pickup_demonstrations/pos=\'-.03 -.47 0.02\' euler=\'0 0 -.7\' (1).mjl'
    #'demo_name': 'teleOpPickup_force/pos=\'0 -.5 0.02\' euler=\'0 0 -.7\' (1).mjl'
    'demo_name': 'teleOpPickup_noGround/log_1.mjl'
}

demo = load_demo(common['demo_name'])
if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])
x0_init = np.zeros((105,))
x0_init[:30] = demo['qpos'][0]
x0_init[30:60] = demo['qvel'][0]
x0_init[60:100] = 0.1*np.ones((40,))
x0_init[100:] = demo['sensordata'][0][20:25]
agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/Adroit/Adroit_hand(teleOp)131_noground.xml',
    'x0': x0_init,
    'dt': 0.03,
    'substeps': 15,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [[]],
    'T': 81,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS, SENSORDATA],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS, SENSORDATA],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'smooth_noise': True,
    'smooth_noise_sigma': 1.0,
    'smooth_noise_renormalize': True
}

# algorithm = {
#     'type': AlgorithmTrajOpt,
#     'conditions': common['conditions'],
#     'iterations': 25,
#     'kl_step': 2.0,
#     'max_step_mult': 5.0,
#     'min_step_mult': 0.05,
#     'sample_decrease_var': 0.05,
#     'sample_increase_var': 0.1
# }


algorithm = {
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
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

cost_tgt_force = copy.copy(demo['sensordata'][:, 20:25])
cost_wt_force = 10.0*np.ones((5,))

force_cost = {
    'type': CostState,
    'l1': 0.0,
    'l2': 10.0,
    'alpha': 1e-5,
    'data_types': {
        SENSORDATA: {
            'target_state': cost_tgt_force,
            'wp': cost_wt_force,
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [state_cost],
    'weights': [1.0],
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

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, ACTIVATIONS],
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': example_tf_network,
    'iterations': 10000,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPrior,
}



config = {
    'iterations': algorithm['iterations'],
    'num_samples': 10,
    'verbose_trials': 10,
    'verbose_policy_trials': 5,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
