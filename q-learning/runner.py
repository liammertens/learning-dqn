from typing import Tuple, Optional

import gym
import numpy as np
from gym import Env
from numpy import ndarray
from matplotlib import pyplot as plt

from agent import QLearnerAgent


def run_episode(env: Env, agent: QLearnerAgent, training: bool, gamma) -> float:
    """
    Interact with the environment for one episode using actions derived from the q_table and the action_selector.

    :param env: The gym environment.
    :param agent: The agent.
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :param gamma: The discount factor.
    :return: The cumulative discounted reward.
    """
    done = False

    initial_obs = env.reset()
    obs = initial_obs[0] # initial_obs is a tuple
    cum_reward = 0.
    t = 0
    while not done:
        action = agent.act(obs, training)
        new_obs, reward, done, _, _ = env.step(action) # returns 5-tuple instead of 4
        if training:
            agent.learn(obs, action, reward, done, new_obs)
        obs = new_obs
        cum_reward += gamma ** t * reward
        t += 1
    agent.decay_epsilon()
    return cum_reward


def train(env: Env, gamma: float, num_episodes: int, evaluate_every: int, num_evaluation_episodes: int,
          alpha: float, epsilon_max: Optional[float] = None, epsilon_min: Optional[float] = None,
          epsilon_decay: Optional[float] = None) -> Tuple[QLearnerAgent, ndarray, ndarray]:
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
    agent = QLearnerAgent(env.observation_space.n, env.action_space.n, alpha, gamma, epsilon_max,
                          epsilon_min, epsilon_decay)
    evaluation_returns = np.zeros(num_episodes // evaluate_every)
    returns = np.zeros(num_episodes)
    for episode in range(num_episodes):
        returns[episode] = run_episode(env, agent, True, gamma) # Train agent on every episode

        if (episode + 1) % evaluate_every == 0: # After evaluate_every episodes do:
            evaluation_step = episode // evaluate_every
            cum_rewards_eval = np.zeros(num_evaluation_episodes)
            for eval_episode in range(num_evaluation_episodes): # Evaluate for num_evaluation_episodes
                cum_rewards_eval[eval_episode] = run_episode(env, agent, False, gamma) # False as training parameter => no agent updates during these episodes
            evaluation_returns[evaluation_step] = np.mean(cum_rewards_eval)
            print(f"Episode {(episode + 1): >{digits}}/{num_episodes:0{digits}}:\t"
                  f"Averaged evaluation return {evaluation_returns[evaluation_step]:0.3}")
            
            env.reset(seed=np.random.randint(0, 2500)) # use a different seed after every evaluation episode
    return agent, returns, evaluation_returns


if __name__ == '__main__':
    try:
        env = gym.make('FrozenLake-v1')
    except gym.error.Error:
        env = gym.make('FrozenLake-v0')

    agent, returns, evaluation_returns = train(env, .99, 30000, 1000, 32, .01, 1.0, .05, .99)


    cum_ret = 0.0
    avg_returns = np.zeros(300) # make bins of 100 returns to average
    for i in range(returns.size):
        cum_ret += returns[i]
        if (i+1) % 100 == 0:
            idx = (i+1)//100 - 1
            avg_returns[idx] = cum_ret/100
            cum_ret = 0.0

    fig1 = plt.figure()
    plt.title("Average returns/100 episodes")
    plt.plot(range(300), avg_returns)
    fig1.savefig('avg_returns_training')

    fig2 = plt.figure()
    plt.title("Avg evaluation return/eval. episode")
    plt.plot(range(30), evaluation_returns)
    fig2.savefig('avg_returns_eval.png')