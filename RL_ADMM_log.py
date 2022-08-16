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
from ADMM import *

def Rconv(rp,rd):
    if (abs(rp)<1e-4 and abs(rd)<1e-4):
        return 0
    else:
        return -1

class ADMM_Env_rho_unique(Env):
    def __init__(self):
        #self.action_space = Box(low=0,high=10,shape=(6,))
        self.action_space=Discrete(10)

        self.observation_space=Box(low=-6,high=1,shape=(2,))
        self.state=[1 for i in range(0,2)]
        self.length=1

    def step(self,action):
        #remote_function = ray.remote(solving)
        #result_init = ray.get([remote_function.remote(x) for x in [[g,9] for g in G.nodes()]])
        #result = ray.get([remote_function.remote(x) for x in [[g,action+1] for g in G.nodes()]])
        result=[]
        test=action+1

        for j in G.nodes():
            result.append(solving([j,test],liste_problem))
        primal_residual=0
        dual_residual=0
        for j in G.nodes():
            v_prev=v_j_mean_data.value[j]
            v_j_mean_data.value[j]=(1/(len([i for i in G.successors(j)])+1))*(result[j][2]+sum(result[i][3] for i in G.successors(j)))
            alpha_data.value[j]=alpha_data.value[j]+rho_data_vj.value*(-v_j_mean_data.value[j]+result[j][2])
            primal_residual+=abs(result[j][2]-v_j_mean_data.value[j])
            dual_residual+=abs(v_prev-v_j_mean_data.value[j])
            for i in G.predecessors(j):
                tab=[k for k in G.successors(i)]
                indice_tab=tab.index(j)
                p_ij_mean_data.value[j]=0.5*(result[j][4]+result[i][6+indice_tab])
                q_ij_mean_data.value[j]=0.5*(result[j][5]+result[i][8+indice_tab])
                gamma_ij_data.value[j]=gamma_ij_data.value[j]+rho_data_pij.value*(-p_ij_mean_data.value[j]+result[j][4])
                delta_ij_data.value[j]=delta_ij_data.value[j]+rho_data_qij.value*(-q_ij_mean_data.value[j]+result[j][5])
                primal_residual+=abs(result[j][4]- p_ij_mean_data.value[j])+abs(result[j][5]- q_ij_mean_data.value[j] )
            enfant2=[k for k in G.successors(j)]
            for k in range(0,len(enfant2)):
                v_i_mean_data.value[enfant2[k]]= v_j_mean_data.value[j]
                alpha_i_data.value[enfant2[k]]=alpha_i_data.value[enfant2[k]]+rho_data_vi.value*(-v_j_mean_data.value[j]+result[enfant2[k]][3])
                p_jk_prev=p_jk_mean_data.value[j][k]
                q_jk_prev=q_jk_mean_data.value[j][k]
                p_jk_mean_data.value[j][k]=0.5*(result[j][6+k]+result[enfant2[k]][4])
                q_jk_mean_data.value[j][k]=0.5*(result[j][8+k]+result[enfant2[k]][5])
                gamma_jk_data.value[j][k]=gamma_jk_data.value[j][k]+rho_data_pjk.value*(-p_jk_mean_data.value[j][k]+result[j][6+k])
                delta_jk_data.value[j][k]=delta_jk_data.value[j][k]+rho_data_qjk.value*(-q_jk_mean_data.value[j][k]+result[j][8+k])
                primal_residual+=abs(result[enfant2[k]][3]-v_j_mean_data.value[j])+abs(result[j][6+k]-p_jk_mean_data.value[j][k])+abs(result[j][8+k]-q_jk_mean_data.value[j][k])
                dual_residual+=abs(p_jk_mean_data.value[j][k]-p_jk_prev)+abs(q_jk_mean_data.value[j][k]-q_jk_prev)
        self.state=[]
        self.state.append(math.log10(primal_residual))
        self.state.append(math.log10(test*dual_residual))
   
            
        self.length+=1
       # if ((abs(primal_residual)-abs(primal_residual_init))/abs(primal_residual_init))>0:
       #     primal_res =-10
       # else:
       #     primal_res=max(-0.25+abs(primal_residual_init)/(min(abs(primal_residual),abs(primal_residual_init)-1e-5)-abs(primal_residual_init)),-5)

       # if ((abs(dual_residual)-abs(dual_residual_init))/abs(dual_residual_init))>0:
        #    dual_res =-10
       # else:
        #    dual_res =max(-0.25+ abs(dual_residual_init)/(min(abs(dual_residual),abs(dual_residual_init)-1e-5)-abs(dual_residual_init)),-5)
        
        #print(primal_res,dual_res)

        reward=Rconv(primal_residual,test*dual_residual)

        
        if primal_residual<=1e-4 and test*dual_residual<=1e-4:
            done=True
            #ray.shutdown()
            
        else:
            if self.length<=200:
                done=False
            else:
                done=True

                

                #ray.shutdown()
        info={}
        return self.state,reward,done,info
        


    def render(self):
        pass

    def reset(self):
        self.state=[1 for i in range(0,2)]
        rho_data_vj.value=1
        rho_data_vi.value=1        
        rho_data_pij.value=1
        rho_data_qij.value=1
        rho_data_qjk.value=1
        rho_data_pjk.value=1
        #rho_data.value=10
        v_i_mean_data.value=np.ones(nb_noeud)
        alpha_i_data.value=np.zeros(nb_noeud)
        p_ij_mean_data.value=np.zeros(nb_noeud)
        gamma_ij_data.value=np.zeros(nb_noeud)
        q_ij_mean_data.value=np.zeros(nb_noeud)
        delta_ij_data.value=np.zeros(nb_noeud)
        v_j_mean_data.value=np.ones(nb_noeud)
        alpha_data.value=np.zeros(nb_noeud)
        p_jk_mean_data.value=np.zeros((nb_noeud, nb_max_enfant))
        gamma_jk_data.value=np.zeros((nb_noeud, nb_max_enfant))
        q_jk_mean_data.value=np.zeros((nb_noeud, nb_max_enfant))
        delta_jk_data.value=np.zeros((nb_noeud, nb_max_enfant))
        for j in G.nodes():
            G.nodes[j]['P_cons']= mem_p_con[j]
            G.nodes[j]['Q_cons']= mem_q_con[j]
            G.nodes[j]['S_max']= mem_p_max[j]
            P_cons_j[j]=G.nodes[j]['P_cons']/BASE_KVA
            Q_cons_j[j]=G.nodes[j]['Q_cons']/BASE_KVA
            P_max_j[j]=G.nodes[j]['S_max']/BASE_KVA

        P_cons_j_data.value=P_cons_j
        Q_cons_j_data.value=Q_cons_j
        P_max_j_data.value=P_max_j
        global liste_problem
        liste_problem=creation_liste_problem()
 
        result=[]
        for j in G.nodes():
            result.append(solving([j,1],liste_problem))
        primal_residual=0
        dual_residual=0

        for j in G.nodes():
            v_prev=v_j_mean_data.value[j]
            v_j_mean_data.value[j]=(1/(len([i for i in G.successors(j)])+1))*(result[j][2]+sum(result[i][3] for i in G.successors(j)))
            alpha_data.value[j]=alpha_data.value[j]+rho_data_vj.value*(-v_j_mean_data.value[j]+result[j][2])
            primal_residual+=abs(result[j][2]-v_j_mean_data.value[j])
            dual_residual+=abs(v_prev-v_j_mean_data.value[j])
            for i in G.predecessors(j):
                tab=[k for k in G.successors(i)]
                indice_tab=tab.index(j)
                p_ij_mean_data.value[j]=0.5*(result[j][4]+result[i][6+indice_tab])
                q_ij_mean_data.value[j]=0.5*(result[j][5]+result[i][8+indice_tab])
                gamma_ij_data.value[j]=gamma_ij_data.value[j]+rho_data_pij.value*(-p_ij_mean_data.value[j]+result[j][4])
                delta_ij_data.value[j]=delta_ij_data.value[j]+rho_data_qij.value*(-q_ij_mean_data.value[j]+result[j][5])
                primal_residual+=abs(result[j][4]- p_ij_mean_data.value[j])+abs(result[j][5]- q_ij_mean_data.value[j] )
            enfant2=[k for k in G.successors(j)]
            for k in range(0,len(enfant2)):
                v_i_mean_data.value[enfant2[k]]= v_j_mean_data.value[j]
                alpha_i_data.value[enfant2[k]]=alpha_i_data.value[enfant2[k]]+rho_data_vi.value*(-v_j_mean_data.value[j]+result[enfant2[k]][3])
                p_jk_prev=p_jk_mean_data.value[j][k]
                q_jk_prev=q_jk_mean_data.value[j][k]
                p_jk_mean_data.value[j][k]=0.5*(result[j][6+k]+result[enfant2[k]][4])
                q_jk_mean_data.value[j][k]=0.5*(result[j][8+k]+result[enfant2[k]][5])
                gamma_jk_data.value[j][k]=gamma_jk_data.value[j][k]+rho_data_pjk.value*(-p_jk_mean_data.value[j][k]+result[j][6+k])
                delta_jk_data.value[j][k]=delta_jk_data.value[j][k]+rho_data_qjk.value*(-q_jk_mean_data.value[j][k]+result[j][8+k])
                primal_residual+=abs(result[enfant2[k]][3]-v_j_mean_data.value[j])+abs(result[j][6+k]-p_jk_mean_data.value[j][k])+abs(result[j][8+k]-q_jk_mean_data.value[j][k])
                dual_residual+=abs(p_jk_mean_data.value[j][k]-p_jk_prev)+abs(q_jk_mean_data.value[j][k]-q_jk_prev)
        self.state=[]
        self.state.append(math.log10(primal_residual))
        self.state.append(math.log10(dual_residual))
        self.length=1
        return self.state
        