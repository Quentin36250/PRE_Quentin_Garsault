from pickletools import float8
from tabnanny import verbose
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
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import reseau_elec
from parametre import*
import sys
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from torch.nn import functional as F
import torch as th
if __name__=='__main__':
	G=reseau_elec.create_graph()
	nb_ligne=len(G.edges())
	nb_noeud=len(G.nodes())
	#Variables
	p=cp.Variable(nb_noeud)
	q=cp.Variable(nb_noeud)
	v=cp.Variable(nb_noeud)
	l=cp.Variable(nb_ligne)
	p_ij=cp.Variable(nb_ligne)
	q_ij=cp.Variable(nb_ligne)

	# Parameters
	P_cons=np.array([G.nodes[j]['P_cons']/BASE_KVA for j in G.nodes()])
	Q_cons=np.array([G.nodes[j]['Q_cons']/BASE_KVA for j in G.nodes()])
	P_max=np.array([G.nodes[j]['S_max']/BASE_KVA for j in G.nodes()])



	#constraint
	constraints=[v<=1.05**2,v>=0.95**2] 
	for j in G.nodes():
	  cont_pj=P_cons[j]
	  cont_qj=Q_cons[j]
	  for i in G.predecessors(j):
	    indice=G[i][j]['num_arete']
	    cont_pj-=p_ij[indice]-G[i][j]['r']*l[indice]
	    cont_qj-=q_ij[indice]-G[i][j]['x']*l[indice]
	  for k in G.successors(j):
	    indice=G[j][k]['num_arete']
	    cont_pj+=p_ij[indice]
	    cont_qj+=q_ij[indice]
	  constraints+=[p[j]==cont_pj]
	  constraints+=[q[j]==cont_qj]

	for j in G.nodes():
	  if j==0:
	    constraints+=[v[0]==1]
	  else:
	    for i in G.predecessors(j):
	      indice=G[i][j]['num_arete']
	      cont_vj=v[i]-2*(G[i][j]['r']*p_ij[indice]+G[i][j]['x']*q_ij[indice])+(G[i][j]['r']**2+G[i][j]['x']**2)*l[indice]
	      constraints+=[v[j]==cont_vj]

	for j in G.nodes():
	  if G.nodes[j]['S_max']==0:
	    if j!=0:
	      constraints+=[p[j]==0, q[j]==0]
	  else:
	    constraints+=[p[j]>=0, p[j]<= P_max[j],p[j]**2+q[j]**2<=(G.nodes[j]['S_max']/BASE_KVA)**2]
	for j in G.nodes():
	  for i in G.predecessors(j):
	    indice=G[i][j]['num_arete']
	    constraints+=[l[indice]>=cp.quad_over_lin(p_ij[indice],v[i])+cp.quad_over_lin(q_ij[indice],v[i])]
	summe=0 
	for j in G.nodes:
	  if(j!=0):
	    summe+=4*(P_max[j]-p[j])**2+2*q[j]**2
	summe+= sum((G[f][t]['x']+G[f][t]['r'])*l[G[f][t]['num_arete']] for (f,t) in G.edges())
	liste=[[f,t] for (f,t) in G.edges()]

	print(liste)

	objective=cp.Minimize(summe)
	prob=cp.Problem(objective,constraints)
	result=prob.solve(solver=cp.ECOS,verbose=True)