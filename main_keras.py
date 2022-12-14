from pickletools import float8
from tabnanny import verbose
import RL_ADMM
import ADMM
from gym import Env
from gym.spaces import Discrete,Box,Tuple
import numpy as np
import random
from stable_baselines3 import A2C,PPO,DQN,DDPG
from stable_baselines3.common.env_util import make_vec_env
import math
from re import M
#from turtle import color
from cvxpy.reductions import solvers
import os
import time
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
import pandapower as pp
import cvxpy as cp
from pandapower.plotting.plotly import simple_plotly, pf_res_plotly
from pandapower.plotting import simple_plot, create_bus_collection
import multiprocessing as mp
from dataclasses import dataclass
import matplotlib.animation as anim
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy,LinearAnnealedPolicy,EpsGreedyQPolicy
from rl.memory import SequentialMemory
import reseau_elec
from parametre import*
from par_cvxpy import *
import sys
from stable_baselines3.common.callbacks import EvalCallback, CallbackList,CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from torch.nn import functional as F
import torch as th
import RL_ADMM_DDPG
import RL_ADMM_stocha
import RL_ADMM_DDPG_stocha
import RL_ADMM_stocha_10
import RL_ADMM_stocha_log_10
import RL_ADMM_stocha_log
import RL_ADMM_log
import RL_ADMM_10
from concurrent.futures import ProcessPoolExecutor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


num_cores=mp.cpu_count()
cp.installed_solvers()


if __name__=="__main__":
    G=reseau_elec.create_graph()
    # Choix de l'environneemnt d'apprentissage (diff RL_ADMM_*: stocha ou non, dim, log ou non)
    env=RL_ADMM_stocha_10.ADMM_Env_rho_unique()
    actions=env.action_space.n
    # def du r??seau de neuronnes
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(actions))
    model.add(Activation('linear'))
    #env=RL_ADMM_stocha.ADMM_Env_rho_unique()
    #Enregristrement de l'agent durant l'entrainement tous les 1500 steps d'apprentissage (voir doc Keras-RL)
    callbacks = [ModelIntervalCheckpoint('model_DQN_stocha_boltz_dim10_'+sys.argv[1]+'_weights_{step}.h5f', interval=1500)]
    states=env.observation_space.shape[0]
    print(states)
    
    #actions=env.action_space.shape[0]
    print(actions)
    #Def memoire et policy
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    #policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 
     #                         attr='eps',
     #                         value_max=1.,
     #                         value_min=.1,
     #                         value_test=.0,
     #                         nb_steps=20000)
     #Choix de l'agent: voir doc keras-RL
    dqn = DQNAgent(model=model, nb_actions=actions, memory=memory, nb_steps_warmup=60,
               target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=float(sys.argv[1])), metrics=['mae'])
    env.reset()
    #lancement de l'entrainement
    dqn.fit(env, nb_steps=100000, visualize=False, verbose=2,callbacks=callbacks)
    #enregistrement de l'agent avec une entrainement complet.
    dqn.save_weights('dqn_{}_keras.h5f'.format(sys.argv[1]), overwrite=True)
    #model=A2C("MlpPolicy",env,verbose=0,learning_rate=float(sys.argv[1]))
    #model=DDPG("MlpPolicy",env,verbose=0,learning_rate=float(sys.argv[1]),buffer_size=300000,policy_kwargs=policy_kwargs)
    
    print("fini")
