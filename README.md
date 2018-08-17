# PPO-Pytorch

Minimal implementation of PPO, running in Mujoco env, using Gym-mujoco. It is based on the code [openai/baselines](https://github.com/openai/baselines).  

Now it is a Pytorch version and it works.  

# Requirement

Python3+  
Pytorch 0.4  
Mujoco  
Gym, Mujoco_py  

# Train

Using following command to train a model, more args can be set in 'main.py'.

    git clone git@github.com:/qqadssp/PPO-Pytorch
    cd PPO-Pytorch
    python3 main.py --env Ant-v2

# Demo

I have trained a model in Ant-v2, it is the file 'logdir/checkpoints/00100'. Using following command to run it.  

    python3 main.py --env Ant-v2 --updates 0 --play --checkpoint logdir/checkpoints/00100

