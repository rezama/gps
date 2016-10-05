""" This file defines an agent for the MuJoCo simulator environment. """
import copy
import sys
import numpy as np
import struct
import IPython
import mjcpy
import time
from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_ADROIT
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, ACTIVATIONS, ACTION_OPEN, ACTION_NOISE, SENSORDATA

from gps.sample.sample import Sample
import libAdroitCom as ad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
import numpy as np
import time

class AgentAdroit(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_ADROIT)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self.x0 = config['x0']
        if not isinstance(self.x0, list):
            self.x0 = [self.x0]
        self._init_adroit()
        self.demo_ctrl = config['demo_ctrl']
        


    def _init_adroit(self):
        # Connect ===================
        print 'ADROIT:> Connecting'
        err = ad.hx_connect("128.208.4.243",1)
        if err==0:
            print 'connection established'
        else:
            print 'connection unsuccessfull'
        assert ad.hx_connected(), 'Check if Adroit Server is running'

        # Get robot info ============
        self.info = ad.hxRobotInfo()
        print 'ADROIT:> Getting robot''s info'
        ad.hx_robot_info(self.info)
        
        # Enulate a trial ============
        policyInfo = np.ones((5, 1))
        policyInfo[0] = 100 #Dx
        policyInfo[1] = 40 #Du
        policyInfo[2] = 1 #T
        policyInfo[3] = 1 #isAct
        policyInfo[4] = 1 #isPS
        self.resetjnt = []
        self.resetPrs = []
        for x0_init in self.x0:
            temp = np.zeros((28,1))
            temp[4:, :] = x0_init[:24, None]
            self.resetjnt.append(temp)  #np.zeros((self.info.nJnt, 1))

            temp_prs = np.zeros((self.info.nTdn,1))
            temp_prs[8:, :] = x0_init[60:, None]
            self.resetPrs.append(1e6*temp_prs)
        self.restCmd = -0.5*np.ones((self.info.nTdn, 1))
        self.sensors = ad.hxSensor()
        
        # clear all buffers =======
        ad.hx_sendBuff_free()
        ad.hx_recvBuff_free()
        ad.hx_msgBuff_free()

        # Allocation  ============
        print 'ADROIT:> Allocate'
        ad.hx_message('allocate',0)
        ad.hx_data_send(5,1, policyInfo,-1)
        while (ad.hx_recvBuff_isFree()==False):
            time.sleep(.01)

        # reset hardware ============
        self.rest()
        ad.hx_sensors(self.sensors, self.info)    #get sensors once as snity check 
        self.rbody = np.zeros((7, 1))  
        self.rbody_old = np.zeros((7, 1))

        self.control_range_min = -1.0
        self.control_range_max = 1.0
        self._joint_idx = range(30)
        self._vel_idx = range(30,60)
        self._act_idx = range(60,100)
        self._sensor_idx = []

    def rest(self):
        #make it rest
        print 'ADROIT:> Rest'
        ad.hx_message('rest',0)
        ad.hx_data_send(self.info.nTdn, 1, self.restCmd, -1)
        while (ad.hx_recvBuff_isFree()==False):
            time.sleep(.01)

    def disconnect(self):
        self.rest()
        # Deallocate controller at remote
        ad.hx_message('deallocate',1)
        print 'ADROIT:> Deallocated'

        # Close connection
        ad.hx_close()
        print 'ADROIT:> Connection closed'

    def squash(self, x, minval, maxval):
        #base = (maxval + minval)/2.0
        #swing = (maxval - minval)/2.0
        #return base + swing*x/np.sqrt(x**2 + swing**2)
        return np.clip(x, minval, maxval)
        #return x

    def reset(self, condition):
        # reset hardware ============
        #ad.hx_message('reset',0)
        #ad.hx_data_send(self.info.nJnt, 1, self.resetjnt, -1)
        #while (ad.hx_recvBuff_isFree()==False):
        #    print 'reset: processing'
        #    time.sleep(.01)

        #print("Condition" + str(condition))

        print 'Adroit:> Reset'
        ad.hx_message('reset',0)
        ad.hx_data_send(self.info.nJnt, 1, self.resetjnt[condition], -1)
        while (ad.hx_recvBuff_isFree()==False):
            time.sleep(.01)
        cmd = raw_input("ADROIT:> Resetting Joints. Press anything to continue ( 'd' to disconnect)")

        if cmd == "d":
            self.disconnect()
            sys.exit(0)

        print("ADROIT:> Resetting Pressure for 4 sec")
        ad.hx_message('resetPrs',0)
        ad.hx_data_send(self.info.nTdn, 1, self.resetPrs[condition], -1)
        #raw_input("Resetting Pressure. Press anything to continue.")
        time.sleep(4)


    def populate_data(self):
        mj_X = np.zeros((100,))
        mj_X[0:24] = np.array(self.sensors.Jnt[4:28]) #hand joints
        mj_X[24:30] = np.array(self.rbody[0:6, 0]) #Object dof
        mj_X[30:54] = np.array(self.sensors.dJnt[4:28]) #hand joints vel
        mj_X[54:60] = np.array((self.rbody[0:6, 0]-self.rbody_old[0:6, 0]))/self._hyperparams['dt'] #Object dof vel]
        mj_X[60:100] = 1e-6*np.array(self.sensors.Prs[8:48]) #activations
        return mj_X



    def read_demo_txt(self, filename):
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

    def sample(self, policy, condition, verbose=True, save=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
        """
        exit_loop = False
        while not exit_loop:
            # Create new sample, populate first time step.
            self.reset(condition)
            #may need some more careful tuning here
            new_sample = self._init_sample(condition)
            #maybe need to get from hardware?
            mj_X = new_sample.get_X()
            U = np.zeros([self.T, self.dU])
            noise = generate_noise(self.T, self.dU, self._hyperparams)
            #declare once or per sample?
            #reset rbody/rbody_old
            ad.hx_sensors(self.sensors, self.info)    # Update robot sensors
            ad.hx_message('rBody',-1)       # request for update
            ad.hx_data_recv(7, 1, self.rbody, -1)# Receive requested (x, y, z, Rx, Ry, Rz)
            t_start = self.sensors.time_stamp
            self.rbody_old = copy.copy(self.rbody)
            
            # get initial U from demo and send it to hardware
            #fix this according to which condition is being called
            U_hardware = np.zeros((48,1))
            U_hardware[8:] = self.demo_ctrl[condition][0][:, None]
            ad.hx_ctrl(U_hardware, 48)
            ad.hx_message('execute',0)

            notDone = True
            time_step_expected = 0
            while notDone:
                self.rbody_old = copy.copy(self.rbody)   
                # get an update from hardware
                ad.hx_sensors(self.sensors, self.info)    # Update robot sensors
                ad.hx_message('rBody',-1)       # request for update
                ad.hx_data_recv(7, 1, self.rbody, -1)# Receive requested (x, y, z, Rx, Ry, Rz)

                # relative time at hardware 
                t = self.sensors.time_stamp - t_start
                time_step = np.floor(t/self._hyperparams['dt'])
                #print(time_step)
                time_step_expected += 1
                if time_step >= self.T-1:
                    notDone = False
                    break
                # query the GPS traj to find control at t
                mj_X = self.populate_data()

                self._set_sample(new_sample, mj_X, time_step, condition)
                #interpolate and stuff
                #mj_U_O, mj_U_N = policy.act_sep(mj_X, mj_X, time_step, noise[time_step, :])
                mj_U = self.squash(policy.act(mj_X, mj_X, time_step, noise[time_step, :]), -1.0, 1.0)
                U[time_step, :] = mj_U
                # send it to hardware
                U_hardware = np.zeros((48,1))
                U_hardware[8:] = mj_U[:, None]
                ad.hx_ctrl(U_hardware, 48)

                if time_step == self.T:
                    notDone = False

            new_sample.set(ACTION, U)
            if save:
                self._samples[condition].append(new_sample)

            self.rest()
            cmd = raw_input("Done collecting sample. Retake sample ? y/[n]")
            if cmd == "y":
                exit_loop = False
            else:
                exit_loop = True

        return new_sample

        """demo = self.read_demo_txt('teleOpPickup_hardware/debugWrist1/adroitPickup_c0_t0.control')
        U_hardware = np.zeros((48,1))
        U_hardware[8:] = demo[0][:, None] #mj_U[:, None]
        
        ad.hx_message('execute',0)        
        for demo_t in demo:
            U_hardware = np.zeros((48,1))
            U_hardware[8:] = demo_t[:, None] #mj_U[:, None]
            U_hardware[8:] = U_hardware[8:]  
            ad.hx_ctrl(U_hardware, 48)
            time.sleep(0.0045)
        """  


    def _init_sample(self, condition):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
        """
        sample = Sample(self)
        ad.hx_sensors(self.sensors, self.info)    # Update robot sensors
        ad.hx_message('rBody',-1)       # request for update
        ad.hx_data_recv(7, 1, self.rbody, -1)# Receive requested (x, y, z, Rx, Ry, Rz)
        mj_X = self.populate_data()
        sample.set(JOINT_ANGLES,
                   mj_X[self._joint_idx], t=0)
        sample.set(JOINT_VELOCITIES,
                   mj_X[self._vel_idx], t=0)
        sample.set(ACTIVATIONS,
                   mj_X[self._act_idx], t=0)
        sample.set(SENSORDATA,
                  mj_X[self._sensor_idx], t=0)

        return sample

    def _set_sample(self, sample, mj_X, t, condition):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
        """
        sample.set(JOINT_ANGLES, np.array(mj_X[self._joint_idx]), t=t+1)
        sample.set(JOINT_VELOCITIES, np.array(mj_X[self._vel_idx]), t=t+1)
        sample.set(ACTIVATIONS, np.array(mj_X[self._act_idx]), t=t+1)
        sample.set(SENSORDATA, np.array(mj_X[self._sensor_idx]), t=t+1)
        
