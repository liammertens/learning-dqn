import random
from typing import Optional
from numpy import ndarray
import numpy as np
from model import QModel
import torch
from replay_buffer import ReplayBuffer
from torch.autograd import Variable

class DQNAgent:
    """
    The agent class for exercise 1.
    """

    def __init__(self,
                 obs_dim: int,
                 num_actions: int,
                 learning_rate: float,
                 gamma: float,
                 buffer_capacity: int,
                 batch_size: int, # has to be smaller than buffer_capacity and evaluate_every in runner.py!
                 epsilon_max: Optional[float] = None,
                 epsilon_min: Optional[float] = None,
                 epsilon_decay: Optional[float] = None,
                 soft_update: bool = False,
                 weight_copy_rate: int = 3000):
        """
        :param num_states: Number of states.
        :param num_actions: Number of actions.
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        """
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.nn = QModel(obs_dim, num_actions)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)

        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(self.buffer_capacity, tuple([self.obs_dim])) # initialise replay buffer

        self.target_nn = QModel(obs_dim, num_actions) # setup w-
        #self.copy_nn()

        self.training_steps = 0 # keep count of the amount of training steps done (to replace target network)
        self.soft_update = soft_update
        self.weight_copy_rate = weight_copy_rate

    def greedy_action(self, observation) -> int:
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        return torch.argmax(self.nn(observation)).item()

    def act(self, observation, training: bool = True) -> int:
        """
        Return the action.

        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        greedy_action = self.greedy_action(observation)
        if training:
            random_action = np.random.randint(0, self.num_actions)
            return np.random.choice([greedy_action, random_action], p=[1-self.epsilon, self.epsilon])
        else:
            return greedy_action

    def learn(self, obs, act, rew, done, next_obs) -> None:
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """

        self.buffer.add_transition(obs, act, rew, done, next_obs)

        if self.soft_update:
            # use soft target net updates
            # Average return does not improve
            for weight_idx in self.nn.state_dict():
                self.target_nn.state_dict()[weight_idx] = self.nn.state_dict()[weight_idx] * 0.005 + self.target_nn.state_dict()[weight_idx] * (1 - 0.005)
        else:
            self.training_steps += 1
            if (self.training_steps == self.weight_copy_rate): # chosen as constant, could also be a parameter
                self.training_steps = 0
                self.copy_nn()

        # sample random transitions from buffer
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        #q(St, a)
        state_action_values = self.nn(states) # [batch_size, 2]
        
        idx = torch.unsqueeze(torch.from_numpy(actions), 1) # expand the nr of dimensions of actions to allow for use in torch.gather => [batch_size, 1]
        estimates = torch.gather(state_action_values, 1, idx) # get correct action_values
    

        # max qw-(St+1, a)
        with torch.no_grad():
            next_state_values= self.target_nn(next_states).max(1)[0] # return max entry along axis 1
            for i in range(31):
                if dones[i]:
                    next_state_values[i] = 0 # ensure final states have target value of 0 !!!!
        targets = torch.from_numpy(rewards) + (self.gamma * next_state_values)# [batch_size]

        criterion = torch.nn.MSELoss()

        loss = criterion(estimates, targets.unsqueeze(1)) # ensure dimensions are the same
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self) -> None:
        if (self.epsilon_decay is not None) and (self.epsilon > self.epsilon_min):
            new_eps = self.epsilon * self.epsilon_decay
            if new_eps < self.epsilon_min:
                self.epsilon = self.epsilon_min
            else:
                self.epsilon = new_eps

    def copy_nn(self): # replaces w- with w
        self.learning_rate *= self.learning_rate
        #self.weight_copy_rate *= 2
        print("Copied weights")
        self.target_nn.load_state_dict(self.nn.state_dict())

