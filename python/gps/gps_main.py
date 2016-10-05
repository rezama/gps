""" This file defines the main object that runs experiments. """

import matplotlib as mpl
#mpl.use('Qt4Agg')

import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, ACTIVATIONS, ACTION_OPEN, ACTION_NOISE, SENSORDATA
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
import struct
import pickle


class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config):
        self._hyperparams = config
        self._conditions = config['common']['conditions']
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()
        self.gui = GPSTrainingGUI(config['common']) if config['gui_on'] else None

        config['algorithm']['agent'] = self.agent
        self.algorithm = config['algorithm']['type'](config['algorithm'])
        self.f1 = plt.figure()
        self.f1.canvas.set_window_title('GPS State/Action')
        self.af1 = self.f1.add_subplot(311)
        self.af2 = self.f1.add_subplot(312)
        self.af3 = self.f1.add_subplot(313)
        #self.af4 = self.f1.add_subplot(414)

        # Cost plots
        self.all_costs = self.algorithm.cost[0]._costs
       	cost_num = len(self.all_costs)
       	self.f2 = plt.figure()
        self.f2.canvas.set_window_title('GPS Cost')
       	self.f2figs = [None]*(cost_num+1)
       	for fig_num in range(cost_num):
        	self.f2figs[fig_num] = self.f2.add_subplot(cost_num+1, 1, fig_num+1)
        self.f2figs[cost_num] = self.f2.add_subplot(cost_num+1, 1, cost_num+1)
        self.f2figs[cost_num].set_xlim([-1, self._hyperparams['iterations']])
        plt.draw()
        plt.show(block=False)

    def run(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        itr_start = self._initialize(itr_load)
        print("in gps main")
        
        # maxlen = 115
        # filenames = ['hand_controller_1_demo7.pkl', 
        #             'hand_controller_2_demo0.pkl',
        #             'hand_controller_3_demo35.pkl']
        # # filenames = ['hand_controller_1_demo7.pkl', 
        # #     'hand_controller_1_demo7.pkl',
        # #     'hand_controller_1_demo7.pkl',
        # #     'hand_controller_1_demo7.pkl']
        # for demo_num in range(3):
        #     traj_dist = self.data_logger.unpickle(filenames[demo_num])
        #     len_curr = traj_dist.T
        #     traj_dist_K_new = np.zeros((maxlen, traj_dist.K.shape[1], traj_dist.K.shape[2]))
        #     traj_dist_k_new = np.zeros((maxlen, traj_dist.k.shape[1]))
        #     traj_dist_pol_covar_new = np.zeros((maxlen, traj_dist.pol_covar.shape[1], traj_dist.pol_covar.shape[2]))
        #     traj_dist_chol_pol_covar_new = np.zeros((maxlen, traj_dist.chol_pol_covar.shape[1], traj_dist.chol_pol_covar.shape[2]))
        #     traj_dist_inv_pol_covar_new = np.zeros((maxlen, traj_dist.inv_pol_covar.shape[1], traj_dist.inv_pol_covar.shape[2]))

        #     traj_dist_K_new[:len_curr, :, :] = traj_dist.K
        #     traj_dist_K_new[len_curr:, :, :] = traj_dist.K[-1]
        #     traj_dist_k_new[:len_curr, :] = traj_dist.k
        #     traj_dist_k_new[len_curr:, :] = traj_dist.k[-1]
        #     traj_dist_pol_covar_new[:len_curr, :, :] = traj_dist.pol_covar
        #     traj_dist_pol_covar_new[len_curr:, :, :] = traj_dist.pol_covar[-1]
        #     traj_dist_chol_pol_covar_new[:len_curr, :, :] = traj_dist.chol_pol_covar
        #     traj_dist_chol_pol_covar_new[len_curr:, :, :] = traj_dist.chol_pol_covar[-1]
        #     traj_dist_inv_pol_covar_new[:len_curr, :, :] = traj_dist.inv_pol_covar
        #     traj_dist_inv_pol_covar_new[len_curr:, :, :] = traj_dist.inv_pol_covar[-1]
        #     #do processing here
        #     traj_dist.T = maxlen
        #     traj_dist.K = traj_dist_K_new
        #     traj_dist.k = traj_dist_k_new
        #     traj_dist.pol_covar = traj_dist_pol_covar_new
        #     traj_dist.chol_pol_covar = traj_dist_chol_pol_covar_new
        #     traj_dist.inv_pol_covar = traj_dist_inv_pol_covar_new
        #     self.algorithm.cur[demo_num].traj_distr = traj_dist

        # import IPython
        # IPython.embed()
        

        print("BOOOYEAHH: Experiment time =====================")
        #import IPython
        #IPython.embed()
        
        #To load
        
        # val_vars, scale, bias, x_idx, pol_var = pickle.load(open('weights.pkl', 'rb'))
        # self.algorithm.policy_opt.var = pol_var
        # self.algorithm.policy_opt.policy.scale = scale
        # self.algorithm.policy_opt.policy.bias = bias
        # self.algorithm.policy_opt.policy.x_idx = x_idx
        # for v in self.algorithm.policy_opt.variables:
        #     if v.name in val_vars:
        #         print(v.name)
        #         assign_op = v.assign(val_vars[v.name])
        #         self.algorithm.policy_opt.sess.run(assign_op)
        # import IPython
        # IPython.embed()

        for itr in range(itr_start, self._hyperparams['iterations']):
            for cond in self._train_idx:
                print("================ CONDITION:  " + str(cond))
                for i in range(self._hyperparams['num_samples']):
                    print("------------ Sample number: " + str(i))
                    self._take_sample(itr, cond, i)
                    # print("starting things")
                    #raw_input('go to next sample')
            # self.agent.rest()

            traj_sample_lists = [
                self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                for cond in self._train_idx
            ]
            #import IPython
            #IPython.embed()
            # self.plot_x(copy.copy(traj_sample_lists))
            self.plot_cost(copy.copy(traj_sample_lists), itr)

            #import IPython
            #IPython.embed()
            self._take_iteration(itr, traj_sample_lists)
            
            # pol_sample_lists = self._take_policy_samples()
            # print("ITERATION" + str(itr))
            #tbegin_plot = time.time()
            #self.plot_x(copy.copy(traj_sample_lists))
            #self.plot_cost(copy.copy(traj_sample_lists), itr)
            #print("plotting took " + str( time.time() - tbegin_plot))

            
            #Saving weights
            
            # vars = {}
            # for v in self.algorithm.policy_opt.variables:
            #     vars[v.name] = self.algorithm.policy_opt.sess.run(v)
            # data_dump =[vars, self.algorithm.policy_opt.policy.scale, self.algorithm.policy_opt.policy.bias,\
            #                 self.algorithm.policy_opt.policy.x_idx, self.algorithm.policy_opt.var]
            # with open('weights.pkl','wb') as f:
            #     pickle.dump(data_dump, f)
            

            #Can test network here 
            # sample = self.agent.sample(self.algorithm.policy_opt.policy, 0, verbose=True, save=False))
            print "<< == TEST NETWORK == >"
            # import IPython
            # IPython.embed()

        self._end()

    def test_network(self):
        sample = self.agent.sample(self.algorithm.policy_opt.policy, 0, verbose=True, save=False)


    def load_network(self, filename):
        val_vars, scale, bias, x_idx, pol_var = pickle.load(open(filename, 'rb'))
        self.algorithm.policy_opt.var = pol_var
        self.algorithm.policy_opt.policy.scale = scale
        self.algorithm.policy_opt.policy.bias = bias
        self.algorithm.policy_opt.policy.x_idx = x_idx
        for v in self.algorithm.policy_opt.variables:
            if v.name in val_vars:
                print(v.name)
                assign_op = v.assign(val_vars[v.name])
                self.algorithm.policy_opt.sess.run(assign_op)

    def save_network(self, filename):
        vars = {}
        for v in self.algorithm.policy_opt.variables:
            vars[v.name] = self.algorithm.policy_opt.sess.run(v)
        data_dump =[vars, self.algorithm.policy_opt.policy.scale, self.algorithm.policy_opt.policy.bias,\
                        self.algorithm.policy_opt.policy.x_idx, self.algorithm.policy_opt.var]
        with open(filename,'wb') as f:
            pickle.dump(data_dump, f)


    def plot_x(self, traj_sample_lists):
    	self.af1.clear()
        self.af2.clear()
        #self.af3.clear()
        #self.af4.clear()
        for sample_list in traj_sample_lists:
            for sample in sample_list._samples:
                x = sample.get_X()
                # sensordata = sample.get(SENSORDATA)
                u = sample.get_U()
                # uo = sample.get(ACTION_OPEN)
                # un = sample.get(ACTION_NOISE)
                T, d = x.shape
                u_d = u.shape[1]
                iter_x = [70]
                
                for dim in iter_x:
                    self.af1.plot(range(T), x[:, dim])
                    self.af1.plot(range(T), self._hyperparams['demo_joints'][:, 70], color='black')
                self.af1.set_ylabel('states')
                for dim in range(1):
                    self.af2.plot(range(T), self._hyperparams['demo_joints'][:, 70])
                self.af2.set_ylabel('demos')
                for dim in range(1):
                    self.af3.plot(range(T), u[:, 36])
                    self.af3.plot(range(T), self._hyperparams['demo_ctrl'][:, 10], color='black')
                self.af3.set_ylabel('actions sent')
        self.f1.canvas.draw()

    def plot_cost(self, traj_sample_lists, itr):
    	#for subplot_fig in self.f2figs:
    	#	subplot_fig.clear()

        costs = self.algorithm.cost[0]._costs
        cost_num = len(costs)
        for c_num in range(cost_num):
            self.f2figs[c_num].clear()


        sum_cost = 0.0
        for sample_list in traj_sample_lists:
            for sample in sample_list._samples:
                x = sample.get_X()
                T, d = x.shape
                for c_num in range(cost_num):
                    self.f2figs[c_num].plot(range(T), costs[c_num].eval(sample)[0])
                    sum_cost += np.sum(costs[c_num].eval(sample)[0])
                    self.f2figs[c_num].set_ylabel('cost '+str(c_num))
                    self.f2figs[c_num].set_title('totalCost: '+str(sum_cost))
        print("SUM COST " + str(sum_cost))
        self.f2figs[-1].plot([itr],[sum_cost], marker='*', markersize=10)
        self.f2.canvas.draw()


    def test_policy(self, itr, N):
        """
        Take N policy samples of the algorithm state at iteration itr,
        for testing the policy to see how it is behaving.
        (Called directly from the command line --policy flag).
        Args:
            itr: the iteration from which to take policy samples
            N: the number of policy samples to take
        Returns: None
        """
        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit(), since t
        traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            ('traj_sample_itr_%02d.pkl' % itr))

        pol_sample_lists = self._take_policy_samples(N)
        self.data_logger.pickle(
            self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
            copy.copy(pol_sample_lists)
        )

        if self.gui:
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.set_status_text(('Took %d policy sample(s) from ' +
                'algorithm state at iteration %d.\n' +
                'Saved to: data_files/pol_sample_itr_%02d.pkl.\n') % (N, itr, itr))

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            if self.gui:
                self.gui.set_status_text('Press \'go\' to begin.')
            return 0
        else:
            algorithm_file = self._data_files_dir + 'algorithm_i_%02d.pkl' % itr_load
            self.algorithm = self.data_logger.unpickle(algorithm_file)
            if self.algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1) # called instead of sys.exit(), since this is in a thread
                
            if self.gui:
                traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('traj_sample_itr_%02d.pkl' % itr_load))
                pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('pol_sample_itr_%02d.pkl' % itr_load))
                self.gui.update(itr_load, self.algorithm, self.agent,
                    traj_sample_lists, pol_sample_lists)
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'go\' to begin.') % itr_load)
            return itr_load + 1

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """

        pol = self.algorithm.cur[cond].traj_distr
        if self.gui:
            self.gui.set_image_overlays(cond)   # Must call for each new cond.
            redo = True
            while redo:
                while self.gui.mode in ('wait', 'request', 'process'):
                    if self.gui.mode in ('wait', 'process'):
                        time.sleep(0.01)
                        continue
                    # 'request' mode.
                    if self.gui.request == 'reset':
                        try:
                            self.agent.reset(cond)
                        except NotImplementedError:
                            self.gui.err_msg = 'Agent reset unimplemented.'
                    elif self.gui.request == 'fail':
                        self.gui.err_msg = 'Cannot fail before sampling.'
                    self.gui.process_mode()  # Complete request.

                self.gui.set_status_text(
                    'Sampling: iteration %d, condition %d, sample %d.' %
                    (itr, cond, i)
                )
                self.agent.sample(
                    pol, cond,
                    verbose=(i < self._hyperparams['verbose_trials'])
                )

                if self.gui.mode == 'request' and self.gui.request == 'fail':
                    redo = True
                    self.gui.process_mode()
                    self.agent.delete_last_sample(cond)
                else:
                    redo = False
        else:
            self.agent.sample(
                pol, cond,
                verbose=(i < self._hyperparams['verbose_trials'])
            )

    def _take_iteration(self, itr, sample_lists):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Calculating.')
            self.gui.start_display_calculating()
        self.algorithm.iteration(sample_lists)
        if self.gui:
            self.gui.stop_display_calculating()

    def _take_policy_samples(self, N=None):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        if 'verbose_policy_trials' not in self._hyperparams:
            return None
        if not N:
            N = self._hyperparams['verbose_policy_trials']
        if self.gui:
            self.gui.set_status_text('Taking policy samples.')
        pol_samples = [[None for _ in range(N)] for _ in range(self._conditions)]
        for cond in range(len(self._test_idx)):
            for i in range(N):
                pol_samples[cond][i] = self.agent.sample(
                    self.algorithm.policy_opt.policy, self._test_idx[cond],
                    verbose=True, save=False)
        return [SampleList(samples) for samples in pol_samples]

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Logging data and updating GUI.')
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.save_figure(
                self._data_files_dir + ('figure_itr_%02d.png' % itr)
            )
        if 'no_sample_logging' in self._hyperparams['common']:
            return
        self.data_logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(self.algorithm)
        )
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )
        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

    def _end(self):
        """ Finish running and exit. """
        if self.gui:
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM only)')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    if args.new:
        from shutil import copy

        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)

        prev_exp_file = '.previous_experiment'
        prev_exp_dir = None
        try:
            with open(prev_exp_file, 'r') as f:
                prev_exp_dir = f.readline()
            copy(prev_exp_dir + 'hyperparams.py', exp_dir)
            if os.path.exists(prev_exp_dir + 'targets.npz'):
                copy(prev_exp_dir + 'targets.npz', exp_dir)
        except IOError as e:
            with open(hyperparams_file, 'w') as f:
                f.write('# To get started, copy over hyperparams from another experiment.\n' +
                        '# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.')
        with open(prev_exp_file, 'w') as f:
            f.write(exp_dir)

        exit_msg = ("Experiment '%s' created.\nhyperparams file: '%s'" %
                    (exp_name, hyperparams_file))
        if prev_exp_dir and os.path.exists(prev_exp_dir):
            exit_msg += "\ncopied from     : '%shyperparams.py'" % prev_exp_dir
        sys.exit(exit_msg)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))

    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    if args.targetsetup:
        try:
            import matplotlib.pyplot as plt
            from gps.agent.ros.agent_ros import AgentROS
            from gps.gui.target_setup_gui import TargetSetupGUI

            agent = AgentROS(hyperparams.config['agent'])
            TargetSetupGUI(hyperparams.config['common'], agent)

            plt.ioff()
            plt.show()
        except ImportError:
            sys.exit('ROS required for target setup.')
    elif test_policy_N:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        random.seed(0)
        np.random.seed(0)

        data_files_dir = exp_dir + 'data_files/'
        data_filenames = os.listdir(data_files_dir)
        algorithm_prefix = 'algorithm_itr_'
        algorithm_filenames = [f for f in data_filenames if f.startswith(algorithm_prefix)]
        current_algorithm = sorted(algorithm_filenames, reverse=True)[0]
        current_itr = int(current_algorithm[len(algorithm_prefix):len(algorithm_prefix)+2])

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            test_policy = threading.Thread(
                target=lambda: gps.test_policy(itr=current_itr, N=test_policy_N)
            )
            test_policy.daemon = True
            test_policy.start()

            plt.ioff()
            plt.show()
        else:
            gps.test_policy(itr=current_itr, N=test_policy_N)
    else:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        random.seed(1)
        np.random.seed(1)

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run(itr_load=resume_training_itr)
            )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            gps.run(itr_load=resume_training_itr)


if __name__ == "__main__":
    main()
