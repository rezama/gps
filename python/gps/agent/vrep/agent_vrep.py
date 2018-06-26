""" This file defines an agent for the V-REP simulator environment.

To run this agent:

1. Download V-REP.

http://www.coppeliarobotics.com/downloads.html

2. This program communicates with V-REP using its remote API:

http://www.coppeliarobotics.com/helpFiles/en/remoteApiOverview.htm
http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm

Help Python find the vrep module:

$ export PYTHONPATH=$PYTHONPATH:path/to/vrep/programming/remoteApiBindings/python/python

Copy the file remoteApi.so (or remoteApi.dll, or remoteApi.dylib) from the
V-REP installation folder to the root folder of your GPS installation:

$ cp path/to/vrep/programming/remoteApiBindings/lib/lib/64Bit/remoteApi.so path/to/gps/

3. Launch V-REP.  Leave it open prior to running this program.  This program
automatically loads and runs V-REP simulations as needed.

4. Run the experiment:

$ python python/gps/gps_main.py vrep_example

"""
import copy
import os
import sys
import time

import numpy as np

import vrep

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_VREP
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE

from gps.sample.sample import Sample

SERVER_STATE_SIMULATION_RUNNING = 1
# Part of the V-REP agent implementation is in a child script attached to the
# following object.  You can see it by opening the scene file in vrep_models
# folder and double clicking on the script icon next to the following object:
SCRIPT_OBJECT = 'Plane'

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

class PrettyFloat(float):
    def __repr__(self):
        return "%0.3f" % self

class AgentVREP(Agent):
    """
    All communication between the algorithms and V-REP is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_VREP)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._client_id = None
        self._last_filename = None
        self._sim_time = None
        self._setup_conditions()
        self._setup_vrep_client()
        self._setup_world()
        self._cached_vrep_state = None

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self):
        """
        Helper method for handling setup of the MuJoCo world.
        """
        # Initialize x0.
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            if END_EFFECTOR_POINTS in self.x_data_types:
                # TODO: this assumes END_EFFECTOR_VELOCITIES is also in datapoints right?
                self._load_world(i)
                self._init_world(i)
                self._fetch_vrep_state()
                self.x0.append(self._get_X_from_vrep_state())
                self._close_world()
            else:
                self.x0.append(self._hyperparams['x0'][i])

    def _setup_vrep_client(self):
        """
        Helper method for setting up connection to V-REP remote api.
        """
        self._client_id = vrep.simxStart('127.0.0.1',
                                         self._hyperparams['server_port'], True,
                                         True, 2000, 5)
        if self._client_id == -1:
            sys.exit('Could not connect to V-REP.  Please launch V-REP first '
                     'and try again.  There is no need to load any scene file '
                     'in V-REP.')

    def _vrep_book_keeping(self, last_fn, last_error_code):
        """
        Helper method for debugging communication with V-REP.
        """
        self._sim_time = vrep.simxGetLastCmdTime(self._client_id)
        print '%d: %s: %s' % (self._sim_time, last_fn, last_error_code)

    def _load_world(self, cond):
        """
        Helper method for handling loading of the V-REP world.
        Args:
            cond: Condition.
        """
        filename = self._hyperparams['filename'][cond]
        print filename
        if filename == self._last_filename:
            self._stop_simulation()
        else:
            self._last_filename = filename
            filepath = os.path.abspath(filename)
            print filepath
            error_code = vrep.simxLoadScene(self._client_id, filepath, 0,
                                            vrep.simx_opmode_blocking)
            self._vrep_book_keeping('simxLoadScene', error_code)

        error_code = vrep.simxSynchronous(self._client_id, True)
        self._vrep_book_keeping('simxSynchronous', error_code)
        error_code = vrep.simxStartSimulation(self._client_id,
                                              vrep.simx_opmode_blocking)
        self._vrep_book_keeping('simxStartSimulation', error_code)

    def _stop_simulation(self):
        """
        Helper method for stopping simulation.
        """
        error_code = vrep.simxStopSimulation(self._client_id,
                                             vrep.simx_opmode_blocking)
        self._vrep_book_keeping('simxStopSimulation', error_code)
        running = True
        while running:
            # Need to run some command so that simxGetInMessageInfo has access
            # to fresh server state.
            error_code, _ = vrep.simxGetPingTime(self._client_id)
            error_code, server_state = vrep.simxGetInMessageInfo(
                    self._client_id, vrep.simx_headeroffset_server_state)
            running = server_state & SERVER_STATE_SIMULATION_RUNNING
            self._vrep_book_keeping('Server state', server_state)

    def _close_world(self):
        """
        Helper method for handling setup of the MuJoCo world.
        """
        self._stop_simulation()
        error_code = vrep.simxCloseScene(self._client_id,
                                         vrep.simx_opmode_blocking)
        self._vrep_book_keeping('simxCloseScene', error_code)
        self._last_filename = None

    def close_vrep_client(self):
        """
        Helper method for closing connection to V-REP remote api.
        """
        vrep.simxFinish(self._client_id)

    def _fetch_vrep_state(self):
        """
        Helper method for getting the complete state from V-REP.
        Args:
            None.
        Returns:
            None.
        """
        error_code, _, out_floats, _, _ = vrep.simxCallScriptFunction(
                self._client_id, SCRIPT_OBJECT,
                vrep.sim_scripttype_childscript,
                'state_function', [], [], [], bytearray(),
                vrep.simx_opmode_blocking)
        self._vrep_book_keeping('simxCallScriptFunction(state)', error_code)
        self._cached_vrep_state = np.array(out_floats)

    def _transmit_vrep_action(self, vrep_U):
        """
        Helper method for getting the complete state from V-REP.
        Args:
            vrep_U: Actions to send to V-REP.
        Returns:
            None.
        """
        error_code, _, _, _, _ = vrep.simxCallScriptFunction(
                self._client_id, SCRIPT_OBJECT,
                vrep.sim_scripttype_childscript,
                'act_function', [], vrep_U, [], bytearray(),
                vrep.simx_opmode_blocking)
        self._vrep_book_keeping('simxCallScriptFunction(act)', error_code)

    def _get_sensor_from_vrep_state(self, sensor_name):
        """
        Helper method for getting individual sensor values from cached V-REP
          state.
        Args:
            sensor_name: Sensor to return state for.
        Returns:
            1-D array containing requested sensor's value.
        """
        regular_sensors = [JOINT_ANGLES, JOINT_VELOCITIES,
                           END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
                           END_EFFECTOR_POINT_JACOBIANS]
        image_sensors = [RGB_IMAGE, CONTEXT_IMAGE]
        if sensor_name not in regular_sensors:
            sys.exit('Support for vision not implemented.')
        idx = self._hyperparams['sensor_idx'][sensor_name]
        size = self._hyperparams['sensor_size'][sensor_name]
        return self._cached_vrep_state[idx:idx + size]

    def _get_X_from_vrep_state(self):
        """
        Helper method for getting the complete state from cached V-REP state.
        Args:
            sensor_name: Sensor to return state for.
        Returns:
            1-D array containing the state.
        """
        X = np.array([])
        for sensor_name in self._hyperparams['state_include']:
            val = self._get_sensor_from_vrep_state(sensor_name)
            X = np.concatenate((X, val))
        return X

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        self._load_world(condition)
        self._init_world(condition)

        # Create new sample, populate first time step.
        new_sample = Sample(self)
        vrep_X = self._hyperparams['x0'][condition]

        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            vrep_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                # self._model[condition]['body_pos'][idx, :] += \
                #         var * np.random.randn(1, 3)

        # Initialize x0.
        self._fetch_vrep_state()
        self._set_sample(new_sample, -1)

        # Take the sample.
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            print 'X_t: %s' % map(PrettyFloat, X_t)
            obs_t = new_sample.get_obs(t=t)
            vrep_U = policy.act(X_t, obs_t, t, noise[t, :])
            print 'vrep_U: %s' % map(PrettyFloat, vrep_U)
            U[t, :] = vrep_U
            self._transmit_vrep_action(vrep_U / 10)
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    error_code = vrep.simxSynchronousTrigger(self._client_id)
                    self._vrep_book_keeping('simxSynchronousTrigger', error_code)
                #TODO: Some hidden state stuff will go here.
                self._fetch_vrep_state()
                self._set_sample(new_sample, t)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        self._close_world()
        return new_sample

    def _init_world(self, condition):
        """
        Set the world to a given state.
        Args:
            condition: Which condition to initialize.
        """

        # Initialize world.
        x0 = self._hyperparams['x0'][condition]
        idx = len(x0) // 2
        joint_angles = x0[:idx]
        error_code, _, _, _, _ = vrep.simxCallScriptFunction(
                self._client_id, SCRIPT_OBJECT,
                vrep.sim_scripttype_childscript,
                'init_function', [], joint_angles, [], bytearray(),
                vrep.simx_opmode_blocking)
        self._vrep_book_keeping('simxCallScriptFunction(init)', error_code)

    def _set_sample(self, sample, t):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            t: Time step to set for sample.
        """
        for sensor_name in self._hyperparams['sensor_idx']:
            sensor_value = self._get_sensor_from_vrep_state(sensor_name)
            if sensor_name == END_EFFECTOR_POINT_JACOBIANS:
                jac_shape = (self._hyperparams['sensor_size'][END_EFFECTOR_POINTS],
                             self._hyperparams['sensor_size'][JOINT_ANGLES])
                sensor_value = sensor_value.reshape(jac_shape)
                print 'Jacobian: \n %s' % sensor_value
            print 'Setting %s: %s' % (sensor_name, sensor_value)
            sample.set(sensor_name, sensor_value, t=t + 1)

