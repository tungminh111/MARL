from marl import MARL

import marl
from marl.tools import ClassSpec, _std_repr, is_done, reset_logging
from marl.policy.policy import Policy
from marl.exploration import ExplorationProcess
from marl.experience import ReplayMemory, PrioritizedReplayMemory
from marl.agent import TrainableAgent

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class COMA(MARL):
    def __init__(self, agents_list=[], name='coma', log_dir="logs"):
        MARL.__init__(self, agents_list=agents_list, name=name, log_dir=log_dir)

    def get_critic_model():
        return self.critic

    def learn(self, env, ep_iter, batch_size, max_num_step=100, test_freq=1000, save_freq=1000, save_folder="models", render=False, time_laps=0., verbose=1, timestep_init=0, log_file=None):

        assert timestep_init >= 0, "Initial timestep < 0"
        #assert timestep_init < nb_timesteps, "Initial timestep >= nbtimesteps"
        start_time = datetime.now()
        logging.basicConfig()

        reset_logging()

        if log_file is None:
            logging.basicConfig(stream=sys.stdout, format='%{message}s', level=logging.INFO)
        else:
            logging.basicConfig(filename=os.path.join(self.log_dir, log_file), format='%{message}s', level=logging.INFO)

        print("#> Start learning process.\n|\tDate : {}".format(start_time.strftime("%d/%m/%Y %H:%M:%S")))
        timestep = timestep_init
        cur_iter = 0
        best_rew = self.worst_rew()
        test = False
        self.reset_exploration(max_num_step * batch_size * ep_iter)
        #ep_iter is the number of training loop
        for _ in range(ep_iter):
            cur_iter += 1
            ep_num = batch_size
            experience_indices = []
            #collect batch_size episodes
            for _iter in range(ep_num):
                self.update_exploration(timestep)
                obs = env.reset()
                done = False
                step = 0
                if render:
                    env.render()
                    time.sleep(time_laps)
                while not is_done(done):
                    action = self.action(obs)
                    obs2, rew, done, _ = env.step(action)
                    experience_indices.append(self.experience.position)
                    self.store_experience(obs, action, rew, obs2, done)

                    obs = obs2

                    step += 1
                    timestep += 1
                    if render:
                        env.render()
                        time.sleep(time_laps)

            self.update_critic(experience_indices);
            self.update_actor(experience_indices);
            # Save model
            print("#> iteration {}/{} --- Save Model\n".format(cur_iter, ep_iter))
            self.save_policy(timestep=timestep, folder=save_folder)
            # Test the model
            res_test = self.test(env, 100, max_num_step=max_num_step, render=False)
            _, m_m_rews, m_std_rews = res_test['mean_by_step']
            _, s_m_rews, s_std_rews = res_test['mean_by_episode']
            # minh comment this //  self.writer.add_scalar("Reward/mean_sum", sum(s_m_rews)/len(s_m_rews) if isinstance(s_m_rews, list) else       s_m_rews, timestep)
            self.writer.add_scalar("Reward/mean_sum", sum(s_m_rews)/len(s_m_rews) if not np.isscalar(s_m_rews) else s_m_rews, timestep)
            duration = datetime.now() - start_time
            if verbose == 2:
                log = "#> Step {}/{} (ep {}) - {}\n\|\tMean By Step {} / Dev {}\n\
                    |\t / Dev {}) \n".format(cur_iter, ep_iter, duration,
                                                        np.around(m_m_rews, decimals=2),
                                                        np.around(m_std_rews, decimals=2),
                                                        np.around(s_m_rews, decimals=2),
                                                        np.around(s_std_rews, decimals=2))
                log += self.training_log(verbose)
            else:
                log = "#> Step {}/{} (ep {}) - {}\n\
                     |\tMean By Step {}\n".format(cur_iter,
                                                         ep_iter,
                                                         duration,
                                                         np.around(m_m_rews, decimals=2),
                                                         np.around(s_m_rews, decimals=2))
            best_rew = self.save_policy_if_best(best_rew, s_m_rews,         folder=save_folder)
            print(log)
        logging.info("#> End of learning process")

    def action(self, observation):
        return [ag.action([obs]) for ag, obs in zip(self.agents, observation)]

    def greedy_action(self, observation):
        return [ag.greedy_action([obs]) for ag, obs in zip(self.agents, observation)]


    def update_critic(self, indices):
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.update_critic(indices)

    def update_actor(self, indices):
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.update_actor(indices)

    def store_experience(self, *args):
        TrainableAgent.store_experience(self, *args)
        #observation, action, prob_action, reward, next_observation, done = args




