from typing import Optional
import numpy as np


def create_q_table(num_states: int, num_actions: int) -> np.ndarray:
    """
    Function that returns a q_table as an array of shape (num_states, num_actions) filled with zeros.

    :param num_states: Number of states.
    :param num_actions: Number of actions.
    :return: q_table: Initial q_table.
    """
    return np.zeros((num_states, num_actions))


class QLearnerAgent:
    """
    The agent class for exercise 1.
    """

    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 learning_rate: float,
                 gamma: float,
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
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = create_q_table(num_states, num_actions)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max

    ## Actions => 0: Left, 1: Down, 2: Right, 3: Up
    
    ### Observations => current_row*nrows + current_col

    def greedy_action(self, observation: int) -> int:
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        s_row = self.q_table[observation]
        argmax = np.argmax(s_row) # Select action with highest return = index of highest value el. in this array
        if np.all(s_row == s_row[0]): # return random action to avoid getting stuck in initial state (= whenever all actions yield eq. returns)
            return np.random.randint(0, 4)
        else:
            return argmax

    def act(self, observation: int, training: bool = True) -> int:
        """
        Return the action.

        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        greedy_action = self.greedy_action(observation)
        if training:
            random_action = np.random.randint(0, 4)
            if greedy_action:
                return random_action
            return np.random.choice([greedy_action, random_action], p=[1-self.epsilon, self.epsilon]) # Choose A from S
        else:
            return greedy_action

    def learn(self, obs: int, act: int, rew: float, done: bool, next_obs: int) -> None:
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """
        if not done:
            old_q = self.q_table[obs, act]
            new_q = old_q + self.learning_rate * (rew + self.gamma*np.max(self.q_table[next_obs]) - old_q)
            self.q_table[obs, act] = new_q # update q_value

    def decay_epsilon(self) -> None:
        if (self.epsilon_decay is not None) and (self.epsilon > self.epsilon_min):
            new_eps = self.epsilon * self.epsilon_decay
            if new_eps < self.epsilon_min:
                self.epsilon = self.epsilon_min
            else:
                self.epsilon = new_eps