from pickletools import float8
from tabnanny import verbose
from ADMM import *
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
from par_cvxpy import *
import sys
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from torch.nn import functional as F
import torch as th
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import ray



num_cores=mp.cpu_count()
cp.installed_solvers()

def solving_para(x):
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

if __name__=="__main__":
    G=reseau_elec.create_graph()
    nb_it=[]
    for rho in [2]:
        prim=[]
        du=[]
        done=False
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
        P_cons_j_data.value=P_cons_j
        Q_cons_j_data.value=Q_cons_j
        P_max_j_data.value=P_max_j
        global liste_problem
        liste_problem=creation_liste_problem()
        compt=0
        #ray.init(num_cpus=5)
        while done==False:
            result=[]
            for j in G.nodes():
                result.append(solving_para([j,rho]))
            #OU

            #data=[[j,rho] for j in G.nodes()]


            #with ProcessPoolExecutor(max_workers=10) as pool:
                #result = list(pool.map(solving_para, data))
            #OU

            #with Pool(processes=10) as pool:
                #result = pool.map(solving_para, data)
            #OU

            
            #remote_function = ray.remote(solving_para)
            #result = ray.get([remote_function(x) for x in data])



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
            compt+=1
            prim.append(primal_residual)
            du.append(dual_residual*rho)
            if(compt>=200 or (primal_residual<1E-4 and rho*dual_residual< 1E-4)):
                done=True
                nb_it.append(compt)
                print(compt)#afficher le nombre d'itérations
                print(prim)#afficher l'évolution des normes des résidus primaux
                print(du)#afficher l'évolution des normes des résidus duaux


