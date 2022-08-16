import numpy as np
import cvxpy as cp
import reseau_elec
from parametre import *
G=reseau_elec.create_graph()
n=0
rho_init=1
nb_noeud=len(G.nodes())
nb_ligne=len(G.edges())
nb_max_enfant=2
result=np.zeros((nb_noeud,11))
p_ij_mean=np.zeros(nb_noeud)
q_ij_mean=np.zeros(nb_noeud)
v_i_mean=np.ones(nb_noeud)
v_j_mean=np.ones(nb_noeud)
p_jk_mean=np.zeros((nb_noeud, nb_max_enfant))
q_jk_mean=np.zeros((nb_noeud, nb_max_enfant))

r_ij=np.zeros(nb_noeud)
x_ij=np.zeros(nb_noeud)

P_cons_j=np.zeros(nb_noeud)
Q_cons_j=np.zeros(nb_noeud)
P_max_j=np.zeros(nb_noeud)

mem_p_con=np.zeros(nb_noeud)
mem_q_con=np.zeros(nb_noeud)
mem_p_max=np.zeros(nb_noeud)
for j in G.nodes():
  P_cons_j[j]=G.nodes[j]['P_cons']/BASE_KVA
  Q_cons_j[j]=G.nodes[j]['Q_cons']/BASE_KVA
  P_max_j[j]=G.nodes[j]['S_max']/BASE_KVA
  mem_p_con[j]=G.nodes[j]['P_cons']
  mem_q_con[j]=G.nodes[j]['Q_cons']
  mem_p_max[j]=G.nodes[j]['S_max']
  for i in G.predecessors(j):
    r_ij[j]=G[i][j]['r']
    x_ij[j]=G[i][j]['x']


alpha=np.zeros(nb_noeud)
alpha_i=np.zeros(nb_noeud)
gamma_ij=np.zeros(nb_noeud)
delta_ij=np.zeros(nb_noeud)
gamma_jk=np.zeros((nb_noeud, nb_max_enfant))
delta_jk=np.zeros((nb_noeud, nb_max_enfant))
max_iter=200
primal_residual=[1]
dual_residual=[1]

#Definition du modèle de résolution:
#Paramètres
rho_data_vj=cp.Parameter(pos=True)
rho_data_vi=cp.Parameter(pos=True)
rho_data_pij=cp.Parameter(pos=True)
rho_data_qij=cp.Parameter(pos=True)
rho_data_qjk=cp.Parameter(pos=True)
rho_data_pjk=cp.Parameter(pos=True)
v_i_mean_data=cp.Parameter(nb_noeud)
alpha_i_data=cp.Parameter(nb_noeud)
p_ij_mean_data=cp.Parameter(nb_noeud)
gamma_ij_data=cp.Parameter(nb_noeud)
q_ij_mean_data=cp.Parameter(nb_noeud)
delta_ij_data=cp.Parameter(nb_noeud)
v_j_mean_data=cp.Parameter(nb_noeud)
alpha_data=cp.Parameter(nb_noeud)
p_jk_mean_data=cp.Parameter((nb_noeud,2))
gamma_jk_data=cp.Parameter((nb_noeud,2))
q_jk_mean_data=cp.Parameter((nb_noeud,2))
delta_jk_data=cp.Parameter((nb_noeud,2))
r_ij_data=cp.Parameter(nb_noeud)
x_ij_data=cp.Parameter(nb_noeud)
P_cons_j_data=cp.Parameter(nb_noeud)
Q_cons_j_data=cp.Parameter(nb_noeud)
P_max_j_data=cp.Parameter(nb_noeud)

rho_data_vj.value=1
rho_data_vi.value=1
rho_data_pij.value=1
rho_data_qij.value=1
rho_data_qjk.value=1
rho_data_pjk.value=1
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
r_ij_data.value=r_ij
x_ij_data.value=x_ij
P_cons_j_data.value=P_cons_j
Q_cons_j_data.value=Q_cons_j
P_max_j_data.value=P_max_j

#Variables



p_j=cp.Variable()
q_j=cp.Variable()
v_j=cp.Variable()
v_i=cp.Variable()
p_ij=cp.Variable()
q_ij=cp.Variable()
q_jk=cp.Variable(2)
p_jk=cp.Variable(2)
l_ij=cp.Variable()