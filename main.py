import os
import time
import random
import argparse
import numpy as np
from collections import deque

import torch
import gym
from util import logger
from util.monitor import Monitor
from env.vec_normalize import VecNormalize
from env.dummy_vec_env import DummyVecEnv
from ppo import train
from agent import MlpAgent

def mujoco_arg_parser():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--updates', type=int, default=int(1e4))
    parser.add_argument('--logdir', help='log dir', type=str, default='logdir')
    parser.add_argument('--checkpoint', help='checkpoint', type=str, default=None)
    parser.add_argument('--play', default=False, action='store_true')
    return parser.parse_args()

def set_global_seeds(i):

    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

def main():
    args = mujoco_arg_parser()
    logger.configure(dir=args.logdir)
    
    nenv = 1
    envs = []
    for i in range(nenv):
        e = gym.make(args.env)
        e.seed(args.seed + i)  #for repeatability
        e = Monitor(e, logger.get_dir(), allow_early_resets=True)
        envs.append(e)
    envs = DummyVecEnv(envs)
    envs = VecNormalize(envs)

    set_global_seeds(args.seed) #for repeatability

    agent = MlpAgent(envs.observation_space.shape[0], envs.action_space.shape[0])
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    agent = train(agent, envs, N_steps=2048, N_updates=args.updates, batch_size=128,
                       lam=0.95, gamma=0.99, N_train_sample_epochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=3e-4,
                       cliprange=0.2,
                       save_interval = 100)

    if args.play:
        logger.log("Running trained model")
        obs = np.zeros((envs.num_envs,) + envs.observation_space.shape)
        obs[:] = envs.reset()
        while True:
            actions = agent.step(obs)[0]
            obs[:]  = envs.step(actions)[0]
            envs.render()


if __name__ == '__main__':
    main()
