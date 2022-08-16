from gym import Env
from gym.spaces import Discrete,Box,Tuple
import numpy as np
import random
from stable_baselines3 import A2C,PPO,DQN
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
from parametre import *
from par_cvxpy import *
def creation_liste_problem():
  




  liste_problem=[]

  for j in G.nodes():
    constraints=[v_j<=1.05**2,v_j>=0.95**2]
    enfant=[k for k in G.successors(j)]
    for i in range(len(enfant),2):
      constraints+=[p_jk[i]==0]
      constraints+=[q_jk[i]==0]
      p_jk_mean_data.value[j][i]=0
      q_jk_mean_data.value[j][i]=0
      gamma_jk_data.value[j][i]=0
      delta_jk_data.value[j][i]=0
      
    

    if(j!=0):
      constraints+=[v_i<=1.05**2,v_i>=0.95**2, l_ij>=0]
      constraints+=[p_j-P_cons_j_data[j]==cp.sum([p_jk[k] for k in range(0,len(enfant))])-p_ij+r_ij_data[j]*l_ij]
      constraints+=[q_j-Q_cons_j_data[j]==cp.sum([q_jk[k] for k in range(0,len(enfant))])-q_ij+x_ij_data[j]*l_ij]
      constraints+=[v_j==v_i-2*(r_ij_data[j]*p_ij+x_ij_data[j]*q_ij)+(r_ij_data[j]**2+x_ij_data[j]**2)*l_ij]
      constraints+=[l_ij>=cp.quad_over_lin(p_ij,v_i)+cp.quad_over_lin(q_ij,v_i)]
    else:
      constraints+=[p_j-P_cons_j_data[j]==cp.sum([p_jk[k] for k in range(0,len(enfant))])]
      constraints+=[q_j-Q_cons_j_data[j]==cp.sum([q_jk[k] for k in range(0,len(enfant))])]
      constraints+=[v_i==0,l_ij==0,p_ij==0,q_ij==0]
      constraints+=[v_j==1]

    if(P_max_j_data.value[j]==0):
      if j!=0:
        constraints+=[p_j==0]
        constraints+=[q_j==0]
    else:
      constraints+=[p_j>=0]
      constraints+=[p_j<=P_max_j_data[j]]
      constraints+=[p_j**2+q_j**2<=(P_max_j_data[j])**2]
    #objectif:
    if(j==0):
      obj=0.5*rho_data_vj*(v_j-v_j_mean_data[j]+alpha_data[j]/rho_data_vj)**2+cp.sum([0.5*rho_data_qjk*(q_jk[k]-q_jk_mean_data[j][k]+delta_jk_data[j][k]/rho_data_qjk)**2+0.5*rho_data_pjk*(p_jk[k]-p_jk_mean_data[j][k]+gamma_jk_data[j][k]/rho_data_pjk)**2 for k in range(0,len(enfant))])

    else:
      obj=4*(P_max_j_data[j]-p_j)**2+2*(q_j)**2+cp.multiply(l_ij,r_ij_data[j]+x_ij_data[j])+0.5*cp.multiply(rho_data_vi,(v_i-v_i_mean_data[j] +alpha_i_data[j]/rho_data_vi)**2)+0.5*cp.multiply(rho_data_pij,(p_ij-p_ij_mean_data[j]+gamma_ij_data[j]/rho_data_pij)**2)+0.5*cp.multiply(rho_data_qij,(q_ij-q_ij_mean_data[j]+delta_ij_data[j])**2/rho_data_qij)+0.5*rho_data_vj*(v_j-v_j_mean_data[j]+alpha_data[j]/rho_data_vj)**2+cp.sum([0.5*rho_data_qjk*(q_jk[k]-q_jk_mean_data[j][k]+delta_jk_data[j][k]/rho_data_qjk)**2+0.5*rho_data_pjk*(p_jk[k]-p_jk_mean_data[j][k]+gamma_jk_data[j][k]/rho_data_pjk)**2 for k in range(0,len(enfant))])
    objective=cp.Minimize(obj)
    
    liste_problem.append(cp.Problem(objective,constraints))
  return liste_problem


def solving(x,liste_problem):
    #rho_data.value=rho
    rho_data_vj.value=x[1]
    rho_data_vi.value=x[1]
    rho_data_pij.value=x[1]
    rho_data_qij.value=x[1]
    rho_data_qjk.value=x[1]
    rho_data_pjk.value=x[1]
    result=np.zeros(11)
    liste_problem[x[0]].solve(solver=cp.ECOS)
    if(liste_problem[x[0]].status!='optimal'):
        return np.zeros(11)
    else:


        result[0]=p_j.value

        result[1]=q_j.value      

        result[2]=v_j.value

        result[3]=v_i.value

        result[4]=p_ij.value

        result[5]=q_ij.value

        result[6]=p_jk.value[0]
        result[7]=p_jk.value[1]

        result[8]=q_jk.value[0]
        result[9]=q_jk.value[1]

        result[10]=l_ij.value

        return result