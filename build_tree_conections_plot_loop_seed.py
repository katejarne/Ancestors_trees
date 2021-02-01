#################################################################
#     Ancestors' tree link generator and ploter                 #
#                                                               #
#    C. Jarne 01/25/2018 V5.0                                   #
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

import scipy.stats as ss
import matplotlib

#Figure size settings

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

#Calling the tree generating function "get_tree" and asigning the values

N                                                                   =5#5#6#10#9#7
trees                                                               = 1

lista_women_removed_cantidad=[]
lista_men_removed_cantidad=[]

###  For a random tree:

# generation, ancestor_tree, separo_trees,ances_mariano  = get_tree(N,trees)

#### For using always the same tree (open the ancestors' list)
# Files with more gen same ances number.

pepe = np.loadtxt("Ancestors_tree_of_5_generetations_ii.txt")


#To save the plots

r_dir="plots"

cc            =pepe.T
generation    =cc[0]
ancestor_tree =cc[1]
print("**************************")
print("Starting:")
print("**************************")

lista_full=[]
###############################################################
lista=np.arange(50)#[1,2,3,4,5,6,7,8,9,10,]
for seed_ in (lista):
    #Building the Digraph of the ancestor number vector
    #random.seed(1)
    seed   =seed_
    random.seed(seed)
    n     = 2                 # The number of children for each node of the digraph 
    depth = len(generation)   # number of levels, starting from 0 or generations
    ulim  = 0
    G                          = nx.DiGraph()
    G.add_node(1)              # initialize root
    G_sin                       = nx.DiGraph()# nx.Graph()
    G_sin.add_node(1,label='firts person')


    #One color per generation (Not currently used)
    #############################################################
    colors                      = cm.rainbow(np.linspace(0, 1, len(generation)+1))
    node_color                  = []
    node_color2                 = []
    #print("colors",colors)
    attrs       = {'gender': 'M'}
    color_map   =[]
    color_map_bi=[]
    #############################################################



      # loop over each level

    for level in range(depth): 
      print("***level",level)
      nl   = n**level             # number of nodes at a given level
      llim = ulim + 1             # index of first node at a given level colors
      ulim = ulim + nl            # index of last node at a given level
      for i in range(nl):         # loop over nodes (parents) at a given level
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


    ##############################################################

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
            #print("female color")
        else:
            color_map_bi.append('mediumseagreen')
            #print("male color")

    ########################################################
    #Full binary tree plot

    fig     = plt.figure(figsize=cm2inch(12,10))
    pos=graphviz_layout(G,prog='dot')
    nx.draw(G,pos,with_labels=True,node_color=color_map_bi, edge_color='gray',font_size=5,node_size=150,arrows=True, width=0.5)
    vmin = 0
    vmax = len(generation)
    #sm = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    #sm._A = []
    plt.savefig(r_dir+"/test_tree_bi"+str(seed)+".png",dpi=300, bbox_inches = 'tight')
    #######################################################


    print( '-----------------------------------------------------------------------------------------------------------------------------------')
    print( "Full Binary Tree (all different ancestors): ",expo_tree)
    print( "Ancestors that must be romove out of the tree in each generation, acording to the simulation:",list_a)
    print( "Generation: ", generation,"Endogamic Ancestror's tree: ", ancestor_tree)
    print( '-----------------------------------------------------------------------------------------------------------------------------------')
    print(" ")
    print("**************************")
    print("Level analysis:")
    print("**************************\n")

    for j,i in enumerate(list_a):

         rev = -len(lista_labels_estos_quedan)+j
         print("-------------------------------")
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
         print ("Number of ancestors to be removed",cantidad)

         if j==0:
             lista_labels_estos_quedan.append([1])

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
                 print("Total amount",cantidad)
                 print("Label of Woman removed:", women_removed,"\n How many women are removed: ",women_removed_cantidad)
                 print("Label of Men removed:", men_removed,"\n How many men are removed: ",men_removed_cantidad)
                 lista_men_removed_cantidad.append(men_removed_cantidad)
                 lista_women_removed_cantidad.append(women_removed_cantidad)
                 todos_estos_saco     = women_removed
                 todos_estos_saco.extend(men_removed)

                 labels_estos_quedan  =list(set(todos_estos_hay) - set(todos_estos_saco))

                 print("Generation with less ancestry than the complete one: ",generacion)
                 #print("Todos_estos_hay: ",todos_estos_hay)
                 print("Total of Removed labels: ",todos_estos_saco)
                 print("Total of Remain Labels: ",labels_estos_quedan)
                 lista_labels_estos_quedan.append(labels_estos_quedan)


                 for jj in todos_estos_saco:
                    if G_sin.has_node(jj)==True:
                        G_sin.remove_node(jj)
                        G_sin_.remove_node(jj)
                    else:
                        pass
         #print("-------------------------------")
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
         print( "\nAvaliable men and women:")
         #print("-------------------------------")
         print("H:",ultima_mujeres," M:",ultima_hombres)
         print("From generation with labels: ",desde)
         #print("Len of lista_labels_estos_quedan: ",len(lista_labels_estos_quedan))
         #print("-------------------------------") 

         print("Link assignment")        

         for ii,jjj in enumerate(desde):
             element= jjj
             print("element",element)

             if G_sin.has_node(element)==True:

                 if len(list(G_sin.in_edges(element)))<2:
                     for element_u in ultima:
                         if len(list(G_sin.out_edges(element_u)))==0 and len(list(G_sin.in_edges(element)))!=2: 
                            print("G_sin.in_edges(element)",list(G_sin.in_edges(element)))
                            mama_papa_edge=list(G_sin.in_edges(element))
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
                                G_sin.add_edge(element_u,element)
                                #pass

         for ii,jjj in enumerate(desde):
             element= jjj

             if G_sin.has_node(element)==True:
                 if len(list(G_sin.in_edges(element)))<2:  # Si le pongo esto y hay poca gente en el otro piso la cago len(G_sin.out_edges(element_u))==1 and
                     mama_papa_edge__=list(G_sin.in_edges(element))
                     mama_papa_edge=mama_papa_edge__
                     if len(mama_papa_edge):
                         #print("mama_papa_cero",mama_papa_edge[0])
                         mama_papa_=mama_papa_edge[0]
                         #print("ACAAAAAAA", mama_papa_)
                         mama_papa= mama_papa_[0]
                         #print("ACAAAAAAA", mama_papa_,mama_papa)
                         #print( mama_papa)

                     if  len(list(G_sin.in_edges(element)))!=2:
                         if len(list(G_sin.in_edges(element)))==1 and mama_papa%2==0: #si tiene mama, elegi papa                         
                             G_sin.add_edge(random.choice(ultima_hombres),element) 

                         if len(list(G_sin.in_edges(element)))==1 and (mama_papa+1)%2==0: #si tiene papa, elegi mama                         
                             G_sin.add_edge(random.choice(ultima_mujeres),element) 

                         if len(list(G_sin.in_edges(element)))==0: #Si no tiene ni papa ni mama elegile los 2
                             G_sin.add_edge(random.choice(ultima_mujeres),element) 
                             G_sin.add_edge(random.choice(ultima_hombres),element)

         print("Done")

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


    print ("Generation:",generation, " Ancestror Tree:", ancestor_tree)

    ################## Figure I: 

    fig     = plt.figure(figsize=cm2inch(12,10))
    pos=graphviz_layout(G,prog='dot')
    #nodes.set_edgecolor('None')
    nx.draw(G_sin,pos,with_labels=True,node_color=color_map, edge_color='gray',font_size=5,node_size=150,arrows=True,width=0.5)
    vmin = 0
    vmax = len(generation)
    sm = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    plt.savefig(r_dir+"/test_tree"+str(seed)+".png",dpi=300, bbox_inches = 'tight')

    ################### Figure II:
    fig     = plt.figure(figsize=cm2inch(12,10))

    pos=graphviz_layout(G,prog='dot')
    nx.draw(G_sin_,pos,with_labels=True,node_color=color_map,font_size=5, node_size=150,edge_color='gray',arrows=True,width=0.5)
    vmin = 0
    vmax = len(generation)
    sm = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    plt.savefig(r_dir+"/test_tree_sin"+str(seed)+".png",dpi=300, bbox_inches = 'tight')


    #####################################

    A_matrix = nx.adjacency_matrix(G_sin)
    B_matrix = nx.adjacency_matrix(G)


    print("--------------------------------")
    print("Node Analysis")

    adjacency_matrix    = nx.to_numpy_matrix(G, dtype=np.bool)
    adjacency_matrix_my = nx.to_numpy_matrix(G_sin, dtype=np.bool)

    x=np.arange(0,350)
    recta=2*x

    ###################################
    fig     = plt.figure(figsize=cm2inch(12,10))
    plt.title('Adjacency matrix for a '+str(N)+' Generation-Tree \n (given the ancestor\'s number obtained with the model)',fontsize=8)
    plt.imshow(adjacency_matrix_my,cmap="viridis",interpolation="none",label='adjacency matrix')
    plt.plot(x,recta,color='green',linewidth=1, label="No inbreeding line")
    #plt.colorbar()
    plt.xlim([0,60])
    plt.ylim([0,60])
    plt.ylabel('Progenitors [i]',fontsize = 10)
    plt.xlabel('Descendants[j]',fontsize = 10)
    plt.xticks(np.arange(0, 70, 10.0),fontsize = 5)
    plt.yticks(np.arange(0,70,10.0),fontsize = 5)
    plt.legend(fontsize= 10,loc=4)
    plt.savefig(r_dir+"/adjacency_my_tree"+str(seed)+".png",dpi=300, bbox_inches = 'tight')


    #################################
    fig     = plt.figure(figsize=cm2inch(12,10))
    plt.title('Adjacency matrix for a '+str(N)+' Generation-Tree \n (for a binary tree)',fontsize=8)
    plt.imshow(adjacency_matrix,cmap="viridis",interpolation="none")
    #plt.colorbar()
    plt.xlim([0,60])
    plt.ylim([0,60])

    plt.xticks(np.arange(0, 70, 10.0),fontsize = 5)
    plt.yticks(np.arange(0,70,10.0),fontsize = 5)
    plt.ylabel('Progenitors [i]',fontsize = 10)
    plt.xlabel('Descendants[j]',fontsize = 10)
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
            lista_degree_depurada.append(int(G_sin.out_degree(i_node)))
            lista_nodos_depurada.append(int(i_node))

    #print lista_nodos_depurada
    print("degree list",lista_degree_depurada)
    #mean_deg       = np.average(lista_nodos_depurada, axis=0)  
 
    lista_full.extend(lista_degree_depurada)  
    
    fig     = plt.figure(figsize=cm2inch(15,7))
    
    P1=[]
    xhist_=lista_degree_depurada
    P1= ss.norm.fit(lista_degree_depurada)
    #P1= ss.exponnorm.fit(lista_degree_depurada)
    print("fit",P1)
    rX1 = np.linspace(1,14, 100)
    #rP1 = ss.exponnorm.pdf(rX1, *P1)
    rP1 = ss.norm.pdf(rX1, *P1)
    mean, var, skew, kurt = ss.norm.stats(P1, moments='mvsk')
    #mean, var, skew, kurt = ss.exponorm.stats(P1, moments='mvsk')
    plt.hist(lista_degree_depurada,bins=max(lista_degree),color='pink', label="Descendants "+str(N) +"-generation Tree"+"\n Mean/Sigma:"+str(np.around(P1,decimals=4)))
    #plt.plot(rX1, rP1, 'r--', linewidth=2, label="Mean/Sigma:"+str(np.around(P1,decimals=4)),alpha=0.99)
    
    #+"\n  mean= "+str(np.around(mean,decimals=4))+" \n var="+str(np.around(var,decimals=4))+"\n skew= "+str(np.around(skew,decimals=4))+"\n kurt="+str(np.around(kurt,decimals=4)))
    #plt.plot(rX1, rP1, 'r--', linewidth=2, label="Fit:"+str(P1)+"\n  mean= "+str(mean)+" \n var="+str(var)+"\n skew= "+str(skew)+"\n kurt="+str(kurt))
    plt.xticks(np.arange(0,max(lista_degree)+2,1),fontsize = 8)
    plt.xlim([0,15])
    #plt.yticks(np.arange(0,52,5),fontsize = 8)
    plt.ylabel('# of Nodes (Progenitors)',fontsize = 8)
    plt.xlabel('Direct Descendants',fontsize = 8)
    plt.legend(fontsize= 8,loc=1)
    plt.savefig(r_dir+"/degree_test_tree"+str(seed)+".png",dpi=300, bbox_inches = 'tight')
    #plt.show()

    #print("lista_men_removed_cantidad",lista_men_removed_cantidad)
    #print("lista_women_removed_cantidad",lista_women_removed_cantidad)
    
    ##Aplying the count of all possible trees.s

    '''
    data_1 = lista_men_removed_cantidad
    data_2 = lista_women_removed_cantidad
    data_1.insert(0, 0)
    data_2.insert(0, 0)
    data_1.insert(0, 0)
    data_2.insert(0, 0)

    l_1 =[ 2**(i-1) for i in range(len(data_1)+2)]
    l_1=l_1[1:-1]
    print("l_1" ,l_1)

    l_2 =[ 2**(i-1) for i in range(len(data_2)+2)]
    l_2=l_2[1:-1]
    print("l_2" ,l_2)


    termino_1 = [(a_i-b_i)**(b_i) for a_i, b_i in zip(l_1,data_1)]

    termino_2 = [(a_i-b_i)**(b_i) for a_i, b_i in zip(l_2,data_2)]

    #print("term W",termino_1)
    #print("term M",termino_2)

    termino_1=np.array(termino_1)
    termino_2=np.array(termino_2)
    W=np.prod(termino_1)
    M=np.prod(termino_2)
    #print("Total W: ",W," Total M: ",M)
    #print("Final: ",W*M)
    #print("{:.2e}".format(W*M))
    print("************************")
    '''
#best fit of data

lista_ordenada= sorted(lista_full)   
fig     = plt.figure(figsize=cm2inch(15,7))
#yhist, xhist, patches =plt.hist(lista_full,bins=max(lista_full),color='mediumseagreen',label="Histogram: Number of nodes with out degree average")
#P = ss.norm.fit(lista_full)
P = ss.norm.fit(lista_ordenada[0:-1])
print("lista",lista_full)
print("lista ordenada",lista_ordenada)
weights = np.ones_like(lista_full)/float(len(lista_full))
print("fit",P)
xhist=lista_full
mean, var, skew, kurt = ss.norm.stats(P, moments='mvsk')
rX = np.linspace(1,14, 100)
rP = ss.norm.pdf(rX, *P)


plt.hist(lista_full, bins=2, color='mediumseagreen',  weights=weights ,label="Average Descendants "+str(N) +"-generation Tree"+"\n Mean/Sigma:"+str(mean)+" "+str(var), alpha=0.2, density = False,)
#plt.hist(lista_full, color='mediumseagreen',label="Average Decendents "+str(N) +"-generation Tree"+"\n Mean/Sigma:"+str(np.around(P,decimals=4)),density = True,alpha=0.2)#max(lista_full)
#plt.hist(lista_full, bins=2, color='mediumseagreen',label="Average Descendants "+str(N) +"-generation Tree"+"\n Mean/Sigma:"+str(mean)+" "+str(var), alpha=0.2, stacked=True, density = True,)
#plt.hist(lista_full, bins=max(lista_full), color='mediumseagreen',label="Average Descendants "+str(N) +"-generation Tree"+"\n Mean/Sigma:"+str(np.around(P,decimals=4)),density = True,alpha=0.2)
#plt.plot(rX, rP, 'r--', linewidth=2, label="Fit:"+str(np.around(P,decimals=4)))#+" \n var="+str(np.around(var,decimals=4))+"\n skew= "+str(np.around(skew,decimals=4))+"\n kurt="+str(np.around(kurt,decimals=4)))
#plt.plot(rX, rP, 'r--', linewidth=2, label="Mean/Sigma:"+str(np.around(P,decimals=4)))
plt.ylabel('# of Nodes (Progenitors)',fontsize = 8)
plt.xlabel('Direct Descendants',fontsize = 8)
plt.legend(fontsize= 8,loc=1)
plt.xlim([0,15])
plt.savefig(r_dir+"/degree_full_test_tree"+str(seed)+".png",dpi=300, bbox_inches = 'tight')
#pl    
    

