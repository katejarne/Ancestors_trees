#################################################################
#     One ancestors' tree link generator and .txt save          #
#                                                               #
#    C. Jarne 01/25/2018 V3.0                                   #
#    https://arxiv.org/pdf/1612.08368.pdf                       #
#                                                               #
#################################################################

from generate_random_tree import *
import numpy as np

N          = 7
trees       = 1
generation, ancestor_tree, separo_trees,ances_mariano = get_tree(N,trees)

f_out      = open('%s.txt' %('Ancestors_tree_of_'+str(N)+'_generetations_ii'), 'w')  
xxx        = np.c_[generation, ancestor_tree]

np.savetxt(f_out,xxx,fmt='%f %f',delimiter='\t',header="gen   #ances") 

print("Ready")
