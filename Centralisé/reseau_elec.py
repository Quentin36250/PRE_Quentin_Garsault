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

def create_graph():
    G = nx.DiGraph()

    # Add the buses
    # bus index, P (the power consumed at bus, this is fixed), pos (only for plotting the graph)
    G.add_node(0, S_max=0, P_cons= 13, Q_cons=1,  pos=(3, 4))
    G.add_node(1, S_max=20, P_cons= 0, Q_cons=0,  pos=(3, 3))
    G.add_node(2, S_max=10, P_cons= 0, Q_cons=0,  pos=(2, 2))
    G.add_node(3, S_max=0, P_cons= 6, Q_cons=2, pos=(4, 2))
    G.add_node(4, S_max=10, P_cons= 0, Q_cons=0, pos=(1, 1))
    G.add_node(5, S_max=0, P_cons= 8, Q_cons=2, pos=(3, 1))
    G.add_node(6, S_max=0, P_cons= 15, Q_cons=5, pos=(0, 0))
    G.add_node(7, S_max=20, P_cons= 5, Q_cons=0, pos=(2, 0))
    G.add_node(8, S_max=20, P_cons= 0, Q_cons=0, pos=(3, -1))
    G.add_node(9, S_max=0, P_cons= 10, Q_cons=0, pos=(1, -1))

    # Add the lines
    # From bus, to bus, r (resistance in pu), x (reactance in pu)
    l01 = 1
    G.add_edge(0, 1,num_arete=0, r=0.1/Z_BASE*l01, x=0.3/Z_BASE*l01, closed=True, len = l01)
    l12 = 1
    G.add_edge(1, 2,num_arete=1, r=0.1/Z_BASE*l12, x=0.3/Z_BASE*l12, closed=True, len = l12)
    l13 = 1
    G.add_edge(1, 3,num_arete=2, r=0.1/Z_BASE*l13, x=0.3/Z_BASE*l13, closed=True, len = l13)
    l24 = 1
    G.add_edge(2, 4,num_arete=3, r=0.1/Z_BASE*l24, x=0.3/Z_BASE*l24, closed=True, len = l24)
    l25 = 1
    G.add_edge(2, 5,num_arete=4, r=0.1/Z_BASE*l25, x=0.3/Z_BASE*l25, closed=True, len = l25)
    l46 = 1
    G.add_edge(4, 6,num_arete=5, r=0.1/Z_BASE*l46, x=0.3/Z_BASE*l46, closed=True, len = l46)
    l47 = 1
    G.add_edge(4, 7,num_arete=6, r=0.1/Z_BASE*l47, x=0.3/Z_BASE*l47, closed=True, len = l47)
    l78 = 5
    G.add_edge(7, 8,num_arete=7, r=0.1/Z_BASE*l78, x=0.3/Z_BASE*l78, closed=True, len = l78)
    l79 = 1
    G.add_edge(7, 9,num_arete=8, r=0.1/Z_BASE*l79, x=0.3/Z_BASE*l79, closed=True, len = l79)
    
    return G


def build_net():
  net = pp.create_empty_network() 
  b0 = pp.create_bus(net, name = 'b0', vn_kv=BASE_VOLTAGE/1e3, slack = True, geodata = G.nodes[0]['pos'])
  b1 = pp.create_bus(net, name = 'b1', vn_kv=BASE_VOLTAGE/1e3, geodata = G.nodes[1]['pos'])
  b2 = pp.create_bus(net, name = 'b2', vn_kv=BASE_VOLTAGE/1e3, geodata = G.nodes[2]['pos'])
  b3 = pp.create_bus(net, name = 'b3', vn_kv=BASE_VOLTAGE/1e3, geodata = G.nodes[3]['pos'])
  b4 = pp.create_bus(net, name = 'b4', vn_kv=BASE_VOLTAGE/1e3, geodata = G.nodes[4]['pos'])
  b5 = pp.create_bus(net, name = 'b5', vn_kv=BASE_VOLTAGE/1e3, geodata = G.nodes[5]['pos'])
  b6 = pp.create_bus(net, name = 'b6', vn_kv=BASE_VOLTAGE/1e3, geodata = G.nodes[6]['pos'])
  b7 = pp.create_bus(net, name = 'b7', vn_kv=BASE_VOLTAGE/1e3, geodata = G.nodes[7]['pos'])
  b8 = pp.create_bus(net, name = 'b8', vn_kv=BASE_VOLTAGE/1e3, geodata = G.nodes[8]['pos'])  
  b9 = pp.create_bus(net, name = 'b9', vn_kv=BASE_VOLTAGE/1e3, geodata = G.nodes[9]['pos'])
  pp.create_line_from_parameters(net, from_bus = b0, to_bus = b1, length_km = G[0][1]['len'], r_ohm_per_km = G[0][1]['r']*Z_BASE/G[0][1]['len'], x_ohm_per_km = G[0][1]['x']*Z_BASE/G[0][1]['len'], c_nf_per_km = 0, max_i_ka = 1e20) 
  pp.create_line_from_parameters(net, from_bus = b1, to_bus = b2, length_km = G[1][2]['len'], r_ohm_per_km = G[1][2]['r']*Z_BASE/G[1][2]['len'], x_ohm_per_km = G[1][2]['x']*Z_BASE/G[1][2]['len'], c_nf_per_km = 0, max_i_ka = 1e20)
  pp.create_line_from_parameters(net, from_bus = b1, to_bus = b3, length_km = G[1][3]['len'], r_ohm_per_km = G[1][3]['r']*Z_BASE/G[1][3]['len'], x_ohm_per_km = G[1][3]['x']*Z_BASE/G[1][3]['len'], c_nf_per_km = 0, max_i_ka = 1e20)
  pp.create_line_from_parameters(net, from_bus = b2, to_bus = b4, length_km = G[2][4]['len'], r_ohm_per_km = G[2][4]['r']*Z_BASE/G[2][4]['len'], x_ohm_per_km = G[2][4]['x']*Z_BASE/G[2][4]['len'], c_nf_per_km = 0, max_i_ka = 1e20)
  pp.create_line_from_parameters(net, from_bus = b2, to_bus = b5, length_km = G[2][5]['len'], r_ohm_per_km = G[2][5]['r']*Z_BASE/G[2][5]['len'], x_ohm_per_km = G[2][5]['x']*Z_BASE/G[2][5]['len'], c_nf_per_km = 0, max_i_ka = 1e20)
  pp.create_line_from_parameters(net, from_bus = b4, to_bus = b6, length_km = G[4][6]['len'], r_ohm_per_km = G[4][6]['r']*Z_BASE/G[4][6]['len'], x_ohm_per_km = G[4][6]['x']*Z_BASE/G[4][6]['len'], c_nf_per_km = 0, max_i_ka = 1e20)
  pp.create_line_from_parameters(net, from_bus = b4, to_bus = b7, length_km = G[4][7]['len'], r_ohm_per_km = G[4][7]['r']*Z_BASE/G[4][7]['len'], x_ohm_per_km = G[4][7]['x']*Z_BASE/G[4][7]['len'], c_nf_per_km = 0, max_i_ka = 1e20)
  pp.create_line_from_parameters(net, from_bus = b7, to_bus = b8, length_km = G[7][8]['len'], r_ohm_per_km = G[7][8]['r']*Z_BASE/G[7][8]['len'], x_ohm_per_km = G[7][8]['x']*Z_BASE/G[7][8]['len'], c_nf_per_km = 0, max_i_ka = 1e20)
  pp.create_line_from_parameters(net, from_bus = b7, to_bus = b9, length_km = G[7][9]['len'], r_ohm_per_km = G[7][9]['r']*Z_BASE/G[7][9]['len'], x_ohm_per_km = G[7][9]['x']*Z_BASE/G[7][9]['len'], c_nf_per_km = 0, max_i_ka = 1e20)
  pp.create_ext_grid(net, bus=b0)
  pp.create_sgen(net, bus=b0, p_mw = -25/1e3, q_mvar = 29/1e3, sn_mva = BASE_KVA/1e3)
  pp.create_load(net, bus=b0, p_mw=G.nodes[0]['P_cons']/1e3, q_mvar = G.nodes[0]['Q_cons']/1e3, sn_mva = BASE_KVA/1e3)
  pp.create_load(net, bus=b3, p_mw=G.nodes[3]['P_cons']/1e3, q_mvar = G.nodes[3]['Q_cons']/1e3, sn_mva = BASE_KVA/1e3)
  pp.create_load(net, bus=b5, p_mw=G.nodes[5]['P_cons']/1e3, q_mvar = G.nodes[5]['Q_cons']/1e3, sn_mva = BASE_KVA/1e3)
  pp.create_load(net, bus=b6, p_mw=G.nodes[6]['P_cons']/1e3, q_mvar = G.nodes[6]['Q_cons']/1e3, sn_mva = BASE_KVA/1e3)
  pp.create_load(net, bus=b7, p_mw=G.nodes[7]['P_cons']/1e3, q_mvar = G.nodes[7]['Q_cons']/1e3, sn_mva = BASE_KVA/1e3) 
  pp.create_load(net, bus=b9, p_mw=G.nodes[9]['P_cons']/1e3, q_mvar = G.nodes[9]['Q_cons']/1e3, sn_mva = BASE_KVA/1e3) 
  pp.create_sgen(net, bus=b1, p_mw=19/1e3, q_mvar=2.2/1e3, sn_mva = BASE_KVA/1e3)
  pp.create_sgen(net, bus=b2, p_mw=9/1e3, q_mvar=1.5/1e3, sn_mva = BASE_KVA/1e3)
  pp.create_sgen(net, bus=b4, p_mw=29/1e3, q_mvar=3.3/1e3, sn_mva = BASE_KVA/1e3)
  pp.create_sgen(net, bus=b7, p_mw=29/1e3, q_mvar=-2/1e3, sn_mva = BASE_KVA/1e3)  
  pp.create_sgen(net, bus=b8, p_mw=29/1e3, q_mvar=-2/1e3, sn_mva = BASE_KVA/1e3)     
  
  return net 

def get_key(bus_idx,net):
    for key, value in net.sgen['bus'].items():
         if bus_idx == value:
             return key
 
    return "key doesn't exist"

def get_lineidx(val,dic_line):
     for key, value in dic_line.items():
          if val == value:
              return eval(key)

     return "key doesn't exist"

def modify_nodes_val():
    perturb=[]
    for i in range(0,10):
        perturb.append(random.randint(-10,10))
    return perturb  
