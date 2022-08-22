# PRE_Quentin_Garsault
Codes PRE Quentin Garsault ENSTA/ULiege
Pour le RL:

Pour lancer le programme d'apprentissage, il faut utiliser le fichier main_keras.py

Pour lancer le programme de test des agents, il faut utiliser le fichier test.py

Pour ces deux fichiers, pour pouvoir les éxecuter il faut donner en argument le learning rate, ex: main_keras.py 0.00001. 

Pour le problème centralisé:

Pour lancer la résolution du pb: il faut execucter main.py dans le dossier centralsié

Pour l'ADMM:
Dans le dossier ADMM_Para

Pour lancer le programme avec la parallélisation il faut choisir la librairie voulue dans le fichier main.py (en mettant en commentaires les autres)
et exécuter main.py


Pour lancer le programme avec l'algorithme de rho adaptatif trouvé dans "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers", il faut executer main_adapt.py

Dans le fichier requirement.txt, vous avez accés à toutes les librairies nécessaires. 

Pour utiliser les fichiers .sh, il faut utiliser le cluster Nic5. Le tuto pour se connecter et lancer les programmes est dans le PPT tuto_nic5.PDF. De plus, il faut exécuter la lignes suivante à chaque overture de terminal pour éxecuter notre programme:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ulg/sysmod/garsault/miniconda3/lib
