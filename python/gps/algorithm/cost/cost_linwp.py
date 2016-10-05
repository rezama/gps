""" This file defines the state target cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_STATE
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, evallogl2term, get_ramp_multiplier


class CostLinWP(Cost):
    """ Computes l1/l2 distance to a fixed target state. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_STATE)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        waypoint_step = np.asarray([np.ceil(T*self._hyperparams['waypoint_time'])])
        A = np.zeros((T, Dx, Dx))
        b = np.zeros((T, Dx))
        wpm = np.zeros((T,))

        st = 0

        for i in range(len(waypoint_step)):
            if self._hyperparams['ramp_option'] == 'constant':
                wpm[st:waypoint_step[i]] = np.ones((waypoint_step[i]-st+1, ))
            elif self._hyperparams['ramp_option'] == 'linear':
                wpm[st:waypoint_step[i]]  = np.arange(1, (waypoint_step[i]-st+1))/(waypoint_step[i]-st+1)
            elif self._hyperparams['ramp_option'] == 'quadratic':

                wpm[st:waypoint_step[i]] = (np.arange(1, (waypoint_step[i]-st+1))/float(waypoint_step[i]-st+1))**2
            elif self._hyperparams['ramp_option'] == 'final_only':
                wpm[st:waypoint_step[i]] = np.zeros((waypoint_step[i] - st + 1, ))
            else:
                error('Unknown cost ramp requested!')
            wpm[waypoint_step[i]-1] = self._hyperparams['wp_final_multiplier']
            for t in range(int(waypoint_step[i])):
                A[t, :, :] = wpm[t]*self._hyperparams['A'][:,:]
                b[t, :] = wpm[t]*self._hyperparams['b'][:]
            st = waypoint_step[i] + 1

        x = sample.get_X()
        _, dim_sensor = x.shape
        dist = np.zeros((T,Dx))
        for t in range(int(waypoint_step[i])):
            dist[t] = A[t,:,:].dot(x[t, :]) + b[t, :]
        wpm = np.array([wpm,]*Dx).T
        final_l = np.linalg.norm(dist, axis = 1)
        for t in range(T):
            final_lx[t, :] = A[t, :, :].T.dot(A[t,:,:].dot(x[t, :]) + b[t, :])
            final_lxx[t, :, :] = A[t, :, :].T.dot(A[t,:,:])
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
