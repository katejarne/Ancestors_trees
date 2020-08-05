#################################################################
#     Ancestors' tree link generator and ploter                 #
#                                                               #
#    C. Jarne 01/25/2018 V3.0                                   #
#    https://arxiv.org/pdf/1612.08368.pdf                       #
#                                                               #
#################################################################

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from graphviz import *
import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout
from generate_random_tree import *
import random
from random import choice
from math import log, exp

import matplotlib

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

#Calling the tree generating function "get_tree" and asigne the values

N                                                                   = 7
trees                                                               = 1

lista_women_removed_cantidad=[]
lista_men_removed_cantidad=[]
####    For a random tree
# generation, ancestor_tree, separo_trees,ances_mariano               = get_tree(N,trees)

####     To use always the same tree

#pepe=np.loadtxt("/home/kathy/Dropbox/2015-ancestros-simulacion/paper/tree_python_ok/New-paper-Code-ancestros/Ancestors_tree_of_15_generetations.txt")
pepe=np.loadtxt("/home/kathy/Dropbox/2015-ancestros-simulacion/paper/tree_python_ok/New-paper-Code-ancestros/Ancestors_tree_of_7_generetations_ii.txt")
r_dir="plots"

cc            =pepe.T
generation    =cc[0]
ancestor_tree =cc[1]

print( "Generation:",generation, "gen 2:", generation[2])
print( "Ancestror Tree:", ancestor_tree)

#Building the Digraph of the ancestor number vector
#random.seed(1)
seed=1
random.seed(seed)
n     = 2                 # The number of children for each node of the digraph 
depth = len(generation)   # number of levels, starting from 0 or generations
ulim  = 0

G                          = nx.DiGraph()
G.add_node(1) # initialize root

G_sin                       = nx.DiGraph()# nx.Graph()
G_sin.add_node(1,label='firts person')
colors                      = cm.rainbow(np.linspace(0, 1, len(generation)+1))
node_color                  = []
node_color2                 = []
print("colors",colors)
attrs = {'gender': 'M'}
color_map=[]
color_map_bi=[]


for level in range(depth): 
  # loop over each level
  
  print("***level",level)
  nl   = n**level             # number of nodes on a given level
  llim = ulim + 1             # index of first node on a given level colors
  ulim = ulim + nl            # index of last node on a given level
  for i in range(nl):         # loop over nodes (parents) on a given level
    parent = llim + i
    offset = ulim + i * n + 1 # index pointing to node just before first child    
    node_color2.append(colors[level])
    for j in range(n):        # loop over children for a given node (parent)
      #node_color2.append(colors[level])
      child = offset + j      
      G.add_node(child)
      G.add_edge(parent, child)
      if j%2==0:
          G_sin.add_node(child,gender='M')
         
      else:
          G_sin.add_node(child,gender='F')
          
      G_sin.add_edge(parent, child)
      
      #print("gender:", G_sin.node[child]['gender'])

      #print("gender",gender)
      #node_color.append(colors[level])
      #print '{:d}-->{:d}'.format(parent, child),

# Reversing the tree to get the proper order in the plot

G_sin                       = nx.reverse(G_sin, copy=True)
G_sin_                      = nx.reverse(G_sin, copy=True)
G_sin_                      = nx.reverse(G_sin_, copy=True)
G                           = nx.reverse(G, copy=True)
todos                       = G.nodes()
expo_tree                   = [2**(i+1) for i in range(N)]
list_a                      = [2**(i+1)-element for i,element in enumerate(ancestor_tree)]
lista_labels_estos_quedan   = []
lista_labels_full_tree      = []



lista_nodos_bi= todos
    
for node_ in lista_nodos_bi:

    if node_%2==0: 
        color_map_bi.append('lightcoral')
        print("aaaaaaaaaaaaaa")
    else:
        color_map_bi.append('mediumseagreen')
        print("bbbbbbbbbbbb")


fig     = plt.figure(figsize=cm2inch(12,10))
pos=graphviz_layout(G,prog='dot')
#nodes.set_edgecolor('None')
nx.draw(G,pos,with_labels=True,node_color=color_map_bi, edge_color='gray',font_size=5,node_size=150,arrows=True,graph_border=True,width=0.5,)
vmin = 0
vmax = len(generation)
#sm = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
#sm._A = []
plt.savefig(r_dir+"/test_tree_bi"+str(seed)+".png",dpi=300, bbox_inches = 'tight')

print( '-----------------------------------------------------------------------------------------------------------------------------------')
print( "Ancestral exponential tree (all different): ",expo_tree)
print( "Generation: ", generation," Ancestror's tree: ", ancestor_tree)
print( "Ancestors that must be taken out of the tree in each generation:",list_a)
print( '-----------------------------------------------------------------------------------------------------------------------------------')


for j,i in enumerate(list_a):
   
     rev = -len(lista_labels_estos_quedan)+j
     print("J-generation",j)
     cantidad             =int(i)
     generacion           =j
     ultima               =[]
     ultima_mujeres       =[]
     ultima_hombres       =[]

     if j<depth and j>0:
         ultima=lista_labels_estos_quedan[rev]
     
     todos_estos_hay      =[x for x in range(2**(generacion+1), 2**(generacion+2))]
     lista_labels_full_tree.append(todos_estos_hay)
     print ("Number of ancestors to remove",cantidad)

     if j==0:
         lista_labels_estos_quedan.append([1])
         #print"lista_labels_estos_quedan",lista_labels_estos_quedan
     if j==1:
         lista_labels_estos_quedan.append([2,3])

     if j>0:
             
     ##############################
         if cantidad==0:
             #print"lista_labels_estos_quedan",lista_labels_estos_quedan
             lista_labels_estos_quedan.append(todos_estos_hay)
   
     ##############################
         if cantidad>0:

             women_removed_cantidad = int(cantidad*0.5) #Half woman and half man (or half+1)
             men_removed_cantidad   = cantidad-women_removed_cantidad#cantidad-women_removed_cantidad
             
             women_removed =[]
             men_removed   =[]
             #if cantidad==1:   
             #    women_removed_cantidad = int(cantidad*0.5) #Half woman and half man (or half+1)
             #    men_removed_cantidad   = cantidad-women_removed_cantidad#cantidad-women_removed_cantidad             
             #    women_removed =[]
             #    men_removed   =[]
             
             while len(women_removed)< women_removed_cantidad:

                 insetar_elemento=choice(range(2**(generacion+1), 2**(generacion+2)-1,2)) # woman are even labels
                 if insetar_elemento in women_removed:
                     #print"esta"
                     pass
                 else:
                     women_removed.append(insetar_elemento)

             while len(men_removed)< men_removed_cantidad:

                 insetar_elemento=choice(range(2**(generacion+1)+1, 2**(generacion+2)-1,2)) # men are odd labels
                 if insetar_elemento in men_removed:
                      #print"esta"
                      pass
                 else:
                     men_removed.append(insetar_elemento)
             
             print("Woman removed:", women_removed,"\n women_removed_cantidad: ",women_removed_cantidad)
             print("Men removed:", men_removed,"\n men_removed_cantidad: ",men_removed_cantidad)
             print("cantidad",cantidad)
             lista_men_removed_cantidad.append(men_removed_cantidad)
             lista_women_removed_cantidad.append(women_removed_cantidad)
             todos_estos_saco     = women_removed
             todos_estos_saco.extend(men_removed)
  
             labels_estos_quedan  =list(set(todos_estos_hay) - set(todos_estos_saco))

             print("Generation with less ancestry than the complete one: ",generacion)
             print("Todos_estos_hay: ",todos_estos_hay)
             print("Removed labels: ",todos_estos_saco)
             print("Remain Labels: ",labels_estos_quedan)
             lista_labels_estos_quedan.append(labels_estos_quedan)
            

             for jj in todos_estos_saco:
                if G_sin.has_node(jj)==True:
                    G_sin.remove_node(jj)
                    G_sin_.remove_node(jj)
                else:
                    pass
     print("-------------------------------")
     if j<depth and j>0:
         ultima=lista_labels_estos_quedan[j+1]
         ultima_mujeres=[]
         ultima_hombres=[]

         for j_ultima in ultima:
             if j_ultima%2==0:
                 ultima_mujeres.append(j_ultima)
                 
             if (j_ultima+1)%2==0:
                 ultima_hombres.append(j_ultima)
                 
       
     desde =lista_labels_estos_quedan[j]
     print( "hombres y mujeres disponibles:")
     print("********************************")
     print("H:",ultima_mujeres," M:",ultima_hombres)
     print("From generation with labels: ",desde)
     print("Len of lista_labels_estos_quedan: ",len(lista_labels_estos_quedan))
                 
     for ii,jjj in enumerate(desde):
         #node_color.append(colors[j])  #color each level 
         #print"i- element: ",ii," From:",jjj," rev: ",rev
         #if jjj%2==0:
         #    node_color.append() #mujeres verdes
         #if (jjj+1)%2==0:
         #    pass
         #    #node_color.append('blue') #hombres magenta
         element= jjj
         

         if G_sin.has_node(element)==True:

             if len(G_sin.in_edges(element))<2:
                 for element_u in ultima:
                     if len(G_sin.out_edges(element_u))==0 and len(G_sin.in_edges(element))!=2: 
                        print("G_sin.in_edges(element)",G_sin.in_edges(element))
                        mama_papa_edge=G_sin.in_edges(element)
                        if len(mama_papa_edge):# Me fijo si tiene mama o papa
                            #print("mama_papa_cero",mama_papa_edge[0])
                            mama_papa_=mama_papa_edge[0]
                            mama_papa= mama_papa_[0]
                            #print("mama_papa?",mama_papa)
                            if mama_papa%2==0 and element_u%2==0: # no puede tener dos mamas
                                pass
                            if (mama_papa+1)%2==0 and (element_u+1)%2==0:# no puede tener dos papas
                                pass
                            if (mama_papa+1)%2==0 and element_u%2==0: #si tiene mama, elegi papa  
                                G_sin.add_edge(element_u,element)
                            if mama_papa%2==0 and (element_u+1)%2==0: #si tiene papa, elegi mama 
                                G_sin.add_edge(element_u,element)
                        else:
                            #G_sin.add_edge(element_u,element)
                            pass
     for ii,jjj in enumerate(desde):
         element= jjj
         
         if G_sin.has_node(element)==True:
             if len(G_sin.in_edges(element))<2:  # Si le pongo esto y hay poca gente en el otro piso la cago len(G_sin.out_edges(element_u))==1 and
                 mama_papa_edge=G_sin.in_edges(element)
                 if len(mama_papa_edge):
                     #print("mama_papa_cero",mama_papa_edge[0])
                     mama_papa_=mama_papa_edge[0]
                     mama_papa= mama_papa_[0]
                     
                 if  len(G_sin.in_edges(element))!=2:
                     if len(G_sin.in_edges(element))==1 and mama_papa%2==0: #si tiene mama, elegi papa                         
                         G_sin.add_edge(random.choice(ultima_hombres),element) 

                     if len(G_sin.in_edges(element))==1 and (mama_papa+1)%2==0: #si tiene papa, elegi mama                         
                         G_sin.add_edge(random.choice(ultima_mujeres),element) 

                     if len(G_sin.in_edges(element))==0: #Si no tiene ni papa ni mama elegile los 2
                         G_sin.add_edge(random.choice(ultima_mujeres),element) 
                         G_sin.add_edge(random.choice(ultima_hombres),element)
                     
ultima_gen=lista_labels_estos_quedan[-1]
for kk in ultima_gen:
  node_color.append(colors[len(generation)]) 

ultima_full= lista_labels_full_tree[-1]
for kk in  ultima_full:
  node_color2.append(colors[len(generation)])

#Useful Prints  for extra debugging

lista_nodos= nx.nodes(G_sin)
    
for node in lista_nodos:

  if node%2==0: 
      color_map.append('lightcoral')
  else:
      color_map.append('mediumseagreen')

'''
print"lista total quedan",lista_labels_estos_quedan
print"----------------"
print"----------------"
print "G_sin.nodes()",G_sin.nodes()
print "G_sin.edges()",G_sin.edges()
print "G_sin.number_of_nodes()",G_sin.number_of_nodes()
print "G_sin.number_of_edges()",G_sin.number_of_edges()
print "G_sin.neighbors(1)",G_sin.neighbors(1)
'''
'''
print "G.degree()",G_sin.degree()
print "G.nodes()",G.nodes()
print "G.edges()",G.edges()
print "G.number_of_nodes()",G.number_of_nodes()
print "G.number_of_edges()",G.number_of_edges()
print "G.neighbors(1)",G.neighbors(1)
#G.remove_node(11)
'''


print ("Generation:",generation, " Ancestror Tree:", ancestor_tree)

################## Figure: 

fig     = plt.figure(figsize=cm2inch(12,10))
pos=graphviz_layout(G,prog='dot')
#nodes.set_edgecolor('None')
nx.draw(G_sin,pos,with_labels=True,node_color=color_map, edge_color='gray',font_size=5,node_size=150,arrows=True,graph_border=True,width=0.5,)
vmin = 0
vmax = len(generation)
sm = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
plt.savefig(r_dir+"/test_tree"+str(seed)+".png",dpi=300, bbox_inches = 'tight')

#####################################
fig     = plt.figure(figsize=cm2inch(12,10))

pos=graphviz_layout(G,prog='dot')
nx.draw(G_sin_,pos,with_labels=True,node_color=color_map,font_size=5, node_size=150,edge_color='gray',arrows=True,graph_border=True,width=0.5)
#nodes.set_edgecolor('None')
vmin = 0
vmax = len(generation)
sm = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
plt.savefig(r_dir+"/test_tree_sin"+str(seed)+".png",dpi=300, bbox_inches = 'tight')


#####################################

A_matrix = nx.adjacency_matrix(G_sin)
B_matrix = nx.adjacency_matrix(G)


#print("My tree adjacency matrix: \n",A_matrix.todense())

print("--------------------------------")

#print("Binary tree adjacency matrix \n",B_matrix.todense())


adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool)
adjacency_matrix_my = nx.to_numpy_matrix(G_sin, dtype=np.bool)

x=np.arange(0,350)
recta=2*x

#plt.figure(figsize=(14,10))
fig     = plt.figure(figsize=cm2inch(12,10))
plt.title('Adjacency matrix for a '+str(N)+' Generation-Tree \n (given the ancestor\'s number obtain with the model)',fontsize=8)
#grid(True)

plt.imshow(adjacency_matrix_my,cmap="viridis",interpolation="none",label='adjacency matrix')
plt.plot(x,recta,color='green',linewidth=1, label="No inbreeding line")
#plt.colorbar()
plt.xlim([0,60])
plt.ylim([0,60])
plt.ylabel('Parent [i]',fontsize = 10)
plt.xlabel('Child [j]',fontsize = 10)
plt.xticks(np.arange(0, 70, 10.0),fontsize = 5)
plt.yticks(np.arange(0,70,10.0),fontsize = 5)
plt.legend(fontsize= 10,loc=4)
plt.savefig(r_dir+"/adjacency_my_tree"+str(seed)+".png",dpi=300, bbox_inches = 'tight')


#plt.figure(figsize=(14,10))
fig     = plt.figure(figsize=cm2inch(12,10))
plt.title('Adjacency matrix for a '+str(N)+' Generation-Tree \n (for a binary tree)',fontsize=8)
plt.imshow(adjacency_matrix,cmap="viridis",interpolation="none")
#plt.colorbar()
plt.xlim([0,60])
plt.ylim([0,60])
#plt.xticks(np.arange(0, 62, 10.0),fontsize = 5)
#plt.yticks(np.arange(0,62,10.0),fontsize = 5)

plt.xticks(np.arange(0, 70, 10.0),fontsize = 5)
plt.yticks(np.arange(0,70,10.0),fontsize = 5)
plt.ylabel('Parent [i]',fontsize = 10)
plt.xlabel('Child [j]',fontsize = 10)
plt.savefig(r_dir+"/adjacency_binary"+str(seed)+".png",dpi=300, bbox_inches = 'tight')

####################################
#Nodes study:
lista_nodos = G_sin.nodes()
lista_degree= []

lista_degree_depurada=[]
lista_nodos_depurada=[]

for i_node in lista_nodos:
    lista_degree.append(G_sin.out_degree(i_node))
    if G_sin.out_degree(i_node)>0:
        lista_degree_depurada.append(G_sin.out_degree(i_node))
        lista_nodos_depurada.append(i_node) 

#print lista_nodos_depurada
print lista_degree_depurada
#mean_deg       = np.average(lista_nodos_depurada, axis=0)  

fig     = plt.figure(figsize=cm2inch(15,7))
plt.hist(lista_degree_depurada,bins=max(lista_degree),color='pink',label="Histogram: Number of nodes with out degree")
plt.xticks(np.arange(0,max(lista_degree)+2,1),fontsize = 8)
plt.ylim([0,50])
plt.yticks(np.arange(0,52,5),fontsize = 8)
plt.ylabel('# of Nodes (parents)',fontsize = 8)
plt.xlabel('Out Degree (Child\'s Number for each parent)',fontsize = 8)
plt.legend(fontsize= 8,loc=1)
plt.savefig(r_dir+"/degree_test_tree"+str(seed)+".png",dpi=300, bbox_inches = 'tight')
#plt.show()

print("lista_men_removed_cantidad",lista_men_removed_cantidad)
print("lista_women_removed_cantidad",lista_women_removed_cantidad)

##Aplying the count of all possible trees.s

data_1 = lista_men_removed_cantidad
data_2 = lista_women_removed_cantidad
data_1.insert(0, 0)
data_2.insert(0, 0)
data_1.insert(0, 0)
data_2.insert(0, 0)

l_1 =[ 2**(i-1) for i in range(len(data_1)+2)]
l_1=l_1[1:-1]
print"l_1" ,l_1

l_2 =[ 2**(i-1) for i in range(len(data_2)+2)]
l_2=l_2[1:-1]
print"l_2" ,l_2


termino_1 = [(a_i-b_i)**(b_i) for a_i, b_i in zip(l_1,data_1)]

termino_2 = [(a_i-b_i)**(b_i) for a_i, b_i in zip(l_2,data_2)]

print("termino W",termino_1)
print("termino M",termino_2)

termino_1=np.array(termino_1)
termino_2=np.array(termino_2)
W=np.prod(termino_1)
M=np.prod(termino_2)
print("W: ",W," M: ",M)
print("Final: ",W*M)
print("{:.2e}".format(W*M))
