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
                 epsilon_decay: Optional[float] = None):
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
        self.training_steps = 0 # keep count of the amount of training steps done (to replace target network)

    def greedy_action(self, observation) -> int:
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        return torch.argmax(self.nn(observation)).item() # feed the observation through the neural net and convert to int

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
        self.training_steps += 1
        if (self.training_steps == 250):
            self.training_steps = 0
            self.copy_nn()

        self.buffer.add_transition(obs, act, rew, done, next_obs)

        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        next_state_action_values = torch.zeros(self.batch_size)
        for i in range(0, self.batch_size):
            next_state_action_values[i] = self.gamma*torch.max(self.target_nn(next_states[i])) # gamma*max_qw-(Si+1, A)
        targets = torch.tensor(torch.tensor(rewards) + next_state_action_values) # R + gamma*maxqw-(Si+1, A)

        action_value_estimates = torch.zeros(self.batch_size)
        for i in range(0, self.batch_size):
            action_value_estimates[i] = torch.tensor(self.nn(states[i]), requires_grad=True)[actions[i]] # qw(Si, A)


        criterion = torch.nn.MSELoss()

        loss = criterion(action_value_estimates, targets)
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
        self.target_nn.load_state_dict(self.nn.state_dict())

