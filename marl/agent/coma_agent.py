import marl
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

from marl.tools import gymSpace2dim
from marl.agent import TrainableAgent, MATrainable

class COMAAgent(TrainableAgent, MATrainable):

    def __init__(self,  observation_space, action_space, critic_model='MlpNet', actor_policy=None, agent_quan = 1, actor_model = 'MlpNet', index = None, mas = None, experience = "ReplayMemory-1000", exploration="EpsGreedy", lr_actor=0.001, lr_critic=0.001, gamma=0.99, batch_size=32, tau=0.01, use_target_net=False, name="COMAAgent"):
        TrainableAgent.__init__(self, policy=actor_policy, model=actor_model, observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, lr=lr_actor, gamma=gamma, batch_size=batch_size)
        MATrainable.__init__(self, mas, index)

        self.policy = marl.policy.make('StochasticPolicy', model=marl.model.make(actor_model, obs_sp=gymSpace2dim(observation_space), act_sp=gymSpace2dim(action_space), last_activ=torch.nn.Softmax(dim=-1)), observation_space=gymSpace2dim(observation_space), action_space=action_space)

        self.tau = tau
        self.agent_quan = agent_quan

        self.actor_optimizer = optim.Adam(self.policy.model.parameters(), lr = self.lr)


        critic_input_space = gymSpace2dim(self.observation_space) + agent_quan
        self.critic_model = marl.model.make(critic_model, obs_sp=critic_input_space, act_sp=1)
        self.critic_criterion = nn.MSELoss()
        self.lr_critic = lr_critic
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.lr_critic)

        self.use_target_net = use_target_net
        if self.use_target_net:
            self.target_critic = copy.deepcopy(self.critic_model)
            self.target_critic.eval()

            self.target_policy = copy.deepcopy(self.policy)
            self.target_policy.model.eval()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_critic_input(self, ob, ac, other_ac):
        return torch.cat([ob.view(1, -1).squeeze(0), ac.view(1, -1).squeeze(0), other_ac.view(1, -1).squeeze(0)], -1)

    def get_critic_output(self, ob, ac, other_ac):
        return self.critic_model(self.get_critic_input(torch.tensor([ob], dtype=torch.float), torch.tensor([ac], dtype=torch.float), torch.tensor(other_ac, dtype=torch.float)))


    def update_critic(self, indices):
        loss = torch.tensor(0.0)
        temp = []
        for i in range(len(indices)):
            idx = indices[len(indices) - 1 - i]

            # get record
            obs, acs, rews, next_obs, dones = self.mas.experience.get_transition([idx])
            ob = obs.squeeze(0)[self.index]
            ac = acs.squeeze(0)[self.index]
            rew = rews.squeeze(0)[self.index]
            next_ob = next_obs.squeeze(0)[self.index]
            done = dones.squeeze(0)[self.index]
            other_ac = [acs.squeeze(0)[j] for j in range(len(acs.squeeze(0))) if j != self.index]
            #print(":::DONE", obs, acs, rews, next_obs, done, i)
            if done or i == 0:
                target_value = torch.tensor([rew], dtype=torch.float)
            else:
                # get next action
                next_idx = indices[len(indices) - 1 - (i - 1)]
                obs, acs, rews, next_obs, dones = self.mas.experience.get_transition([next_idx])
                next_ob = obs.squeeze(0)[self.index]
                next_other_ac = [acs.squeeze(0)[j] for j in range(len(acs.squeeze(0))) if j != self.index]
                # compute taget value
                target_value = 0

                # get log prob
                log_prob = self.policy.model(torch.tensor([next_ob], dtype=torch.float))
                log_prob = torch.distributions.Categorical(log_prob).probs

                for j in range(gymSpace2dim(self.action_space)):
                    target_value += log_prob[j] * self.get_critic_output(next_ob, j, next_other_ac).clone().detach()
                target_value *= rew + self.gamma * target_value
                #print(":::CRITIC_INPUT", self.get_critic_input(torch.tensor([next_ob], dtype=torch.float), torch.tensor([next_ac], dtype=torch.float), torch.tensor(next_other_ac, dtype=torch.float)), target_value)
            temp.append(target_value)
            #print(":::CRITIC", target_value)

            # compute current value
            cur_value = 0
            cur_value += self.get_critic_output(ob, ac, other_ac)

            loss += self.critic_criterion(cur_value, target_value)


        loss /= len(indices)
        print(":::CRITIC LOSS", self.index, loss)
        self.critic_optimizer.zero_grad()
        result = loss

        loss.backward()
        self.critic_optimizer.step()



        #TEST
        loss = 0
        for i in range(len(indices)):
            idx = indices[len(indices) - 1 - i]

            #get record
            obs, acs, rews, next_obs, dones = self.mas.experience.get_transition([idx])
            ob = obs.squeeze(0)[self.index]
            ac = acs.squeeze(0)[self.index]
            rew = rews.squeeze(0)[self.index]
            next_ob = next_obs.squeeze(0)[self.index]
            done = dones.squeeze(0)[self.index]
            other_ac = [acs.squeeze(0)[j] for j in range(len(acs.squeeze(0))) if j != self.index]
            #print(":::DONE", obs, acs, rews, next_obs, done, i)

            #compute current value
            target_value = temp[i]
            cur_value = 0
            cur_value += self.get_critic_output(ob, ac, other_ac)

            loss += self.critic_criterion(cur_value, target_value)

        loss /= len(indices)
        print(":::NEWLOSS", self.index, loss)

        return result




    def update_actor(self, indices):
        actor_loss = []
        p = []
        for i in range(len(indices)):
            idx = indices[len(indices) - 1 - i]

            # get record
            obs, acs, rews, next_obs, dones = self.mas.experience.get_transition([idx])
            ob = obs.squeeze(0)[self.index]
            ac = acs.squeeze(0)[self.index]
            rew = rews.squeeze(0)[self.index]
            next_ob = next_obs.squeeze(0)[self.index]
            done = dones.squeeze(0)[self.index]
            other_ac = [acs.squeeze(0)[j] for j in range(len(acs.squeeze(0))) if j != self.index]

            # get log prob
            prob = self.policy.model(torch.tensor([ob], dtype=torch.float))
            prob = torch.distributions.Categorical(prob).probs
            p.append(prob)

            # compute actor loss
            actor_loss.append(0)
            actor_loss[i] = self.get_critic_output(ob, ac, other_ac)

            for j in range(gymSpace2dim(self.action_space)):
                actor_loss[i] -= prob[j].clone().detach() * self.get_critic_output(ob, j, other_ac)
            actor_loss[i] *= prob[ac]

        # optimize
        loss = 0
        for i in range(len(indices)):
            loss -= actor_loss[i]
        loss /= len(indices)
        print("ACTORLOSS", self.index, self.lr, loss)

        self.actor_optimizer.zero_grad()
        loss.backward()
     #   for i in self.policy.model.parameters():
     #       print(":::BEFORE", i, i.grad)
     #       break
        self.actor_optimizer.step()
     #   for i in self.policy.model.parameters():
     #       print(":::AFTER", i)
     #       break


        actor_loss = []
        for i in range(len(indices)):
            idx = indices[len(indices) - 1 - i]

            # get record
            obs, acs, rews, next_obs, dones = self.mas.experience.get_transition([idx])
            ob = obs.squeeze(0)[self.index]
            ac = acs.squeeze(0)[self.index]
            rew = rews.squeeze(0)[self.index]
            next_ob = next_obs.squeeze(0)[self.index]
            done = dones.squeeze(0)[self.index]
            other_ac = [acs.squeeze(0)[j] for j in range(len(acs.squeeze(0))) if j != self.index]

            # get log prob
            prob = p[i]
            cur_prob = self.policy.model(torch.tensor([ob], dtype=torch.float))
            cur_prob = torch.distributions.Categorical(prob).probs

            # compute actor loss
            actor_loss.append(0)
            actor_loss[i] = self.get_critic_output(ob, ac, other_ac)

            for j in range(gymSpace2dim(self.action_space)):
                actor_loss[i] -= prob[j].clone().detach() * self.get_critic_output(ob, j, other_ac)

            actor_loss[i] *= cur_prob[ac]
                #print(":::ACTOR",actor_loss[i])

        # optimize
        loss = 0
        for i in range(len(indices)):
            loss -= actor_loss[i]
        loss /= len(indices)
        print(":::NEW ACTORLOSS", self.index, loss)




