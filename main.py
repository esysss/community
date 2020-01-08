import networkx
import pickle
import numpy
import matplotlib.pyplot as plt
import others
import The_assignment_of_initial_community_labels as first
import The_modification_of_vertexes_labels as second
import Parameters_update as third
import GN_maker

# net = networkx.read_gml('Real database/polbooks.gml',label='id')#threshold = .03
# net = networkx.read_gml('Real database/netscience.gml',label='id')
# net = networkx.read_gml('Real database/football.gml',label='id')#threshold = .16
#net = networkx.read_gml('Real database/dolphins.gml',label='id')   #threshold = .0388348


net = networkx.read_gml('Real database/karate.gml',label='id')#threshold = 0.1074
dic = {}
for i in range(34):
    dic[i+1]=i

net = networkx.relabel_nodes(net,dic)


# net = GN_maker.gnMaker()#threshold .05
#net = networkx.algorithms.community.LFR_benchmark_graph(100,3,1.5,.1,average_degree=20,max_degree=50,min_community=20,max_community=100)

# pos = networkx.spring_layout(net)

M = networkx.adjacency_matrix(net)
M = M.todense()#the sparse adjacency matrix of our network

#table 1:

F = first.neighbor_evaluation_vector(M) #Step1: calculate otherNodes evaluation vectors
sequences = first.Order_access_sequence(F)    #Step2: generate order access sequences


thresholdCondition =1
communities = []
communityCounter = -1
sequenceListPast = []
test = []
for sequence in sequences:               #Step3
    sequenceListPast.append(sequence)
    if not others.in_it(sequence, communities):
        communities.append([])
        communityCounter += 1
        communities[communityCounter].append(sequence)
        currentC = communityCounter
    else:
        currentC = others.currentCummunity(sequence, communities)

    for otherNodes in sequenceListPast:
        DVi = first.average_neighbor_distance(sequence,M,F)
        DVj = first.average_neighbor_distance(otherNodes, M,F)
        DVij = first.average_mutual_neighbor_distance(sequence, otherNodes, M,F)
        test.append(abs(DVij - (DVi + DVj)/2))
        if abs(DVij - (DVi + DVj)/2) <= thresholdCondition:
            pervC = others.currentCummunity(otherNodes, communities) #find the previous community label and delete it
            if pervC is not None :
                communities[pervC].remove(otherNodes)

            communities[currentC].append(otherNodes)

isolate = first.isolate_nodes(communities)        #step4
if len(isolate)>0:
    for nodes in isolate:                        #step5
        mindist = first.minimum_average_distance(nodes, communities,M)
        pervC = others.currentCummunity(nodes, communities)  # find the previous community label and delete it
        communities[pervC].remove(nodes)
        communities[mindist].append(nodes)
communities = others.not_empty(communities)

'''
table 2 ...
'''


status_threshold = 4.4
objfP = float('inf')
objfC = third.JtKLFCM(communities,F,M)                #step 1  #current objective function
counter = 0
saveobjfC = 0
while True:                              #step 2
    #if not abs(objfP - objfC) < status_threshold:
    if not saveobjfC==objfC:
        counter+=1
        if counter%2 == 0:
            saveobjfC = objfC

        unstables = []
        for community in communities:
            for node in community:       #select the unstable vertexes and revise the labels
                Isunasteble = second.\
                    selecting_maximum_membership_for_unstable_vertexes(node,communities,F)
                if Isunasteble[0]:
                    unstables.append((node,Isunasteble[1],Isunasteble[2]))

        for nodes in unstables:
            communities[nodes[1]].remove(nodes[0])
            communities[nodes[2]].append(nodes[0])

        objfP = objfC
        objfC = third.JtKLFCM(communities, F, M)
    else:
        break

communities = others.not_empty(communities)

# theFile = open("dick.pickle","wb")#it says to write in bite
# pickle.dump(communities,theFile)
# theFile.close()

# others.draw(net,communities,pos)