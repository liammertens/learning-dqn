from typing import Tuple, Optional
import matplotlib.pyplot as plt
import gym
import numpy as np
from gym import Env
from numpy import ndarray

from agent import DQNAgent


def run_episode(env: Env, agent: DQNAgent, training: bool, gamma) -> float:
    """
    Interact with the environment for one episode using actions derived from the q_table and the action_selector.

    :param env: The gym environment.
    :param agent: The agent.
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :param gamma: The discount factor.
    :return: The cumulative discounted reward.
    """
    done = False
    obs, _ = env.reset()
    cum_reward = 0.
    t = 0
    while not done:
        action = agent.act(obs, training)
        new_obs, reward, done, _, _ = env.step(action)
        if training:
            agent.learn(obs, action, reward, done, new_obs)
        obs = new_obs
        cum_reward += gamma ** t * reward
        t += 1
    agent.decay_epsilon()
    return cum_reward


def train(env: Env, gamma: float, num_episodes: int, evaluate_every: int, num_evaluation_episodes: int,
          alpha: float, buffer_capacity: int, batch_size:int,
          epsilon_max: Optional[float] = None, epsilon_min: Optional[float] = None,
          epsilon_decay: Optional[float] = None) -> Tuple[DQNAgent, ndarray, ndarray]:
    """
    Training loop.

    :param env: The gym environment.
    :param gamma: The discount factor.
    :param num_episodes: Number of episodes to train.
    :param evaluate_every: Evaluation frequency.
    :param num_evaluation_episodes: Number of episodes for evaluation.
    :param alpha: Learning rate.
    :param epsilon_max: The maximum epsilon of epsilon-greedy.
    :param epsilon_min: The minimum epsilon of epsilon-greedy.
    :param epsilon_decay: The decay factor of epsilon-greedy.
    :return: Tuple containing the agent, the returns of all training episodes and averaged evaluation return of
            each evaluation.
    """
    digits = len(str(num_episodes))
    agent = DQNAgent(4, 2, alpha, gamma, buffer_capacity, batch_size, epsilon_max,
                          epsilon_min, epsilon_decay)
    evaluation_returns = np.zeros(num_episodes // evaluate_every)
    returns = np.zeros(num_episodes)
    for episode in range(num_episodes):
        returns[episode] = run_episode(env, agent, True, gamma)

        if (episode + 1) % evaluate_every == 0:
            evaluation_step = episode // evaluate_every
            cum_rewards_eval = np.zeros(num_evaluation_episodes)
            for eval_episode in range(num_evaluation_episodes):
                cum_rewards_eval[eval_episode] = run_episode(env, agent, False, gamma)
            evaluation_returns[evaluation_step] = np.mean(cum_rewards_eval)
            print(f"Episode {(episode + 1): >{digits}}/{num_episodes:0{digits}}:\t"
                  f"Averaged evaluation return {evaluation_returns[evaluation_step]:0.3}")
    return agent, returns, evaluation_returns


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    
    agent, returns, eval_returns = train(env, .99, 1000, 50, 32, .01, 1000, 20, 1.0, .05, .99)

