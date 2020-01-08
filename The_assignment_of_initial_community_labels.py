import numpy

#(a) The establishment of otherNodes evaluation vectors

# now it's time to calculate the neighbor evaluation
def neighbor_evaluation_vector(M):
    '''
    :param M: is the adjacency matrix of the network
    :return: all the otherNodes evaluation vectors
                F[0,0] is The otherNodes evaluation vector of node 1 and it's neighbor
    '''
    beta = betaCalculator(M)
    F = []
    for column in range(len(M)):
        F.append([])
        for i,j in enumerate(M[:,column]):
            if j==1:
                Fik = numpy.exp(-(numpy.linalg.norm(M[:,column]-M[:,i]))/beta)
                F[column].append((Fik,i))

    return F

def betaCalculator(M):
    '''
    :param M: is the adjacency matrix
    :return: the beta
    '''
    N = len(M)
    Mbar = numpy.mean(M,axis=1)

    bata = 0
    for i in range(N):
        bata += numpy.linalg.norm(Mbar - M[:, i])  # sum of M(i) and Mbar distance
    bata = bata/N
    return bata

#(b) The formation of initial communities
def average_neighbor_distance(vN, M, F):
    neighbors = []
    for f in F[vN]:
        neighbors.append(f[1])

    NbVn = len(neighbors)

    if NbVn == 1 or NbVn == 0:
        return 0
    sumdist = 0
    pased = []
    for nei1 in neighbors:
        pased.append(nei1)
        for nei2 in list(set(neighbors) - set(pased)):
            sumdist += numpy.linalg.norm(M[:, nei1] - M[:, nei2])

    return (1 / (NbVn * (NbVn - 1))) * sumdist


def average_mutual_neighbor_distance(vN, vM, M, F):
    Nneighbors = []
    for f in F[vN]:
        Nneighbors.append(f[1])
    NBVn = len(Nneighbors)

    Mneighbors = []
    for f in F[vM]:
        Mneighbors.append(f[1])
    NBVm = len(Mneighbors)

    if NBVn == 0 or NBVm == 0:
        return 0

    sumdist = 0
    for n in Nneighbors:
        for m in Mneighbors:
            sumdist += numpy.linalg.norm(M[:, n] - M[:, m])

    return 1 / (NBVn * NBVm) * sumdist



def Order_access_sequence(F):
    su = []
    for i,Fs in enumerate(F):
        su.append((len(Fs),i))#make a list with nude number and number of neighbors
    su = sorted(su)
    so = []
    for item in su:
        so.insert(0,item[1])
    return so

#(c) The fusion of isolated vertexes

def isolate_nodes(communities):
    isolate = []
    for community in communities:
        if len(community)==1:
            isolate.append(community[0])
    return isolate

#The average distance d(Vk, Ci) between isolated vertex Vk and community Ci
def minimum_average_distance(Vk,communities,M):
    #it returns the number of that community with minimum distance
    sumdist = 0
    average_dists = []
    for i,community in enumerate(communities):
        if len(community)>1:
            for j in community:
                sumdist += numpy.linalg.norm(M[:,j]-M[:,Vk])
            temp = 2/(len(community)*(len(community)-1))*sumdist
            average_dists.append((temp,i))
    return min(average_dists)[1]
