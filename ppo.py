import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import time
import os
from runner import Runner
from util import logger

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def constfn(val):
    def f(_):
        return val
    return f

def train(agent, env, N_steps, N_updates, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, batch_size=4, N_train_sample_epochs=4, cliprange=0.2,
            save_interval=0):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)

    runner = Runner(env, agent, nsteps=N_steps, gamma=gamma, lam=lam)
    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    for update in range(1, N_updates+1):

        obs, returns, dones, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        epinfobuf.extend(epinfos)

        frac = 1.0 - (update - 1.0) / N_updates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        optimazer = optim.Adam(agent.parameters(), lr=lrnow)

        mblossnames = ['policy_loss', 'value_loss', 'entropy', 'approxkl', 'clipfrac']
        mblossvals = []

#        print('obs = ', obs.shape, '\nreturns = ', returns.shape, '\ndones = ', dones.shape, '\nactions = ', actions.shape, '\nvalues = ', values.shape, '\nneglogpacs = ', neglogpacs.shape, '\nstates = ', type(states), '\nepinfos = ', len(epinfos))

        N_sample_steps = obs.shape[0]
        inds = np.arange(N_sample_steps)
        tstart = time.time()

        agent.train()
        for _ in range(N_train_sample_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, N_sample_steps):

#                _ = input('in:')
                end = start + batch_size
                mbinds = inds[start:end]
                obs_ = torch.tensor(obs[mbinds], requires_grad=True)
                returns_ = torch.tensor(returns[mbinds])
                actions_ = torch.tensor(actions[mbinds])
                values_ = torch.tensor(values[mbinds])
                neglogpacs_ = torch.tensor(neglogpacs[mbinds])
                advs_ = torch.tensor(advs[mbinds])

                optimazer.zero_grad()
                neglogp, entropy, vpred = agent.statistics(obs_, actions_)
                entropy = torch.mean(entropy)
                vf_loss = F.mse_loss(vpred, returns_)
                ratio = torch.exp(neglogpacs_ - neglogp)
                ratio = torch.clamp(ratio, 1.0-cliprangenow, 1.0+cliprangenow)
                pg_loss = torch.mean(- advs_ * ratio)
                approxkl = .5 * torch.mean(pow(neglogp - neglogpacs_, 2))
                clipfrac = torch.mean((torch.max(torch.abs(ratio - 1.0), torch.tensor(cliprangenow))).float())
                loss = torch.mean(pg_loss - entropy * ent_coef + vf_loss * vf_coef)
                loss.backward()
                optimazer.step()

                mblossvals.append([pg_loss.item(), vf_loss.item(), entropy.item(), approxkl.item(), clipfrac.item()])

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(N_sample_steps / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*N_steps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*N_sample_steps)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, mblossnames):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = os.path.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = os.path.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            torch.save(agent.state_dict(), savepath)
    env.close()
    return model

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

