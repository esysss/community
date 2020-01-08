import networkx
import numpy
import re
import matplotlib.pyplot as plt
import community as cm
import sympy
from scipy import spatial
import pickle

def nmi(communities):
    m = max(sum(communities,[]))
    after = numpy.ones(m+1)
    for i, community in enumerate(communities):
        for node in community:
            after[node] = i
    
    return after

# to calculate the beta
def betaCalculator(M):

    N = len(M)
    Mbar = numpy.mean(M,axis=1)
    bata = 0
    for i in range(N):
        bata += numpy.linalg.norm(Mbar-M[:,i])#sum of M(i) and Mbar distance
    bata = bata/N
    return bata

# now it's time to calculate the neighbor evaluation
def neighbor_evaluation_vector(M):

    beta = betaCalculator(M)
    F = []
    for column in range(len(M)):
        F.append([])
        for i,j in enumerate(M[:,column]):
            if j == 1:
                Fik = numpy.exp(-(numpy.linalg.norm(M[:,column]-M[:,i]))/beta)
                F[column].append((Fik,i))

    return F
"""
def average_neighbor_distance(M, neibor):
    neighbors = neibor
    NbVn = len(neighbors)
    
    if NbVn == 1:
        return numpy.inf
    sumdist = 0
    pased = []
    for nei1 in neighbors:
        pased.append(nei1)
        for nei2 in list(set(neighbors)-set(pased)):
            sumdist += numpy.linalg.norm(M[:,nei1]-M[:,nei2])

    for past,nei1 in enumerate(neighbors[:-1]):
        for nei2 in neighbors[past+1:]:
            sumdist += numpy.linalg.norm(M[:, nei1] - M[:, nei2])

    sumdist = [numpy.linalg.norm(M[:, nei1] - M[:, nei2]) for past,nei1 in enumerate(neighbors[:-1]) for nei2 in neighbors[past+1:] ]

    return (1/(NbVn*(NbVn-1)))*numpy.sum(sumdist)
"""
def average_neighbor_distance(M, neibor):
    neighbors = neibor
    NbVn = len(neighbors)
    
    if NbVn == 1:
        return numpy.inf

    sumdist = [numpy.linalg.norm(M[:, nei1] - M[:, nei2]) for past,nei1 in enumerate(neighbors[:-1]) for nei2 in neighbors[past+1:] ]

    return (1/(NbVn*(NbVn-1)))*numpy.sum(sumdist)


def average_mutual_neighbor_distance(M, ineibor, jneibor):
    Nneighbors = ineibor
    NBVn = len(Nneighbors)
    
    Mneighbors = jneibor
    NBVm = len(Mneighbors)
    """
    sumdist = 0
    for n in Nneighbors:
        for m in Mneighbors:
            sumdist += numpy.linalg.norm(M[:,n]-M[:,m])
            
    """

    sumdist = [numpy.linalg.norm(M[:,n]-M[:,m]) for n in Nneighbors for m in Mneighbors]

    return 1/(NBVn*NBVm)*numpy.sum(sumdist)


# now it's time to deal with isolated nodes
def isolate_nodes(communities):
    isolate = []
    for community in communities:
        if len(community)==1:
            isolate.append(community[0])
    return isolate

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

"""
def Order_access_sequence(F):
    su = []
    for i,Fs in enumerate(F):
        su.append((len(Fs),i))#make a list with node number and number of neighbors
    su = sorted(su,reverse = True)
    so = []
    for item in su:
        so.append(item[1])
    return so
"""
#it's an order the who has the most neghbors ? who goes first
def Order_access_sequence(F):

    su = [(len(Fs),i) for i,Fs in enumerate(F)]
    su = sorted(su,reverse = True)

    so = [item[1] for item in su]

    return so

#to check if any empety community is there
def not_empty(communities): # deletes the empty communities
    newCommunities = []
    for i in communities:
        if len(i) > 0:
            newCommunities.append(i)

    return newCommunities

#check that if a node is in a communities or it has any labels
"""
def in_it(item,to_search):# is the node in the community?
    for row in range(len(to_search)):#number of cummunities
        for column in range(len(to_search[row])):#nodes in communities
            if item == to_search[row][column]:
                return True
    return False
"""
def in_it(item,to_search):# is the node in the community?
    for community in to_search:
        if item in community:
            return True
    return False


#returns the number of current community of the node
"""
def currentCummunity(item,communities):
    for i,community in enumerate(communities):
        for node in community:
            if item == node:
                return i
"""
def currentCummunity(item,communities):
    for i,community in enumerate(communities):
        if item in community :
            return i


def draw(net,communities,pos,title):
    color = ['g','y','b','r','c','m','k','#2763c4','#c40daf','w','#720a0a','#ffff16']
    #communities = [[val + 1 for val in lis] for lis in communities]
    for i in range(len(communities)):
        networkx.draw_networkx(net,pos,nodelist=communities[i],node_color = color[i])
    # plt.savefig('gn.png')
    plt.title(title)
    plt.show()

# to calculate the mojularity of community detection
def modd(communities,net):
    di = {}
    #communities = [[val + 1 for val in lis] for lis in communities]
    for i, comunity in enumerate(communities):
        for nodes in comunity:
            di[nodes] = i
    
    return cm.modularity(di,net)

# now we set the membership functions

#(a) The establishment of membership vector
def C_mean_membership_vector(community,n,F):

    neighbor = [] #get the neighbors of node n
    [neighbor.append(row[1]) for row in F[n]]
    sumf1 = 0     #the Numerator of the fraction
    for node in list(set(community).intersection(set(neighbor))):
        for f in F[n]:
            if node==f[1]:
                sumf1 += f[0]
    alll = []
    for i in F[n]:
        alll.append(i[0])
    sumf2 = numpy.sum(alll)      #the denominator of the fraction

    return sumf1/sumf2

# we make the U matrix
def UMaker(communities,M,F) :
    U = numpy.zeros([len(M),len(communities)])
    for nodes in range(len(M)):
        for communityCounter,community in enumerate(communities):
            U[nodes,communityCounter] = C_mean_membership_vector(community,nodes,F)
    return U


def NumberOfNeibors(node, M,net, more):
    neighbors = numpy.array(list(net[node]))
    number = []
    for neighbor in neighbors:
        number.append(numpy.sum(M[neighbor]))
    number = numpy.array(number)

    if more != 'better':
        number = 1/number
    
    return number 


def P_calculator(node,U,F,M, status):
    neighbors = []
    for f in F[node]:
        neighbors.append(f[1])
    averageOfU = U[neighbors,:]
    
#     weight = NumberOfNeibors(node, M, F, 'better')
    weight = NumberOfNeibors(node, M, F, status)
    mat = numpy.reshape(weight,[len(weight),1])*averageOfU
    mat = numpy.sum(mat,axis = 0)
    averageOfU = mat/numpy.sum(weight)
    return averageOfU

def P_maker(U,F,M,status = "better"):
    P1 = []
    for i in range(len(F)):
        p1 = P_calculator(i,U,F,M,status)
        P1.append(p1)
    
    P1 = numpy.array(P1)
    # P1 = P1.astype('object')
    # for i in range(len(P1)):
    #     for j in range(len(P1[i,:])):
    #         P1[i,j] = sympy.Rational(P1[i,j])
    return P1


def P_updater_body(node,P,net,M,status = "better"):
    neighbors = numpy.array(list(net[node]))
    averageOfP = P[neighbors,:]
    
#     weight = NumberOfNeibors(node, M, F, 'better')
    weight = NumberOfNeibors(node, M, net, status)
    mat = numpy.reshape(weight,[len(weight),1])*averageOfP
    mat = numpy.sum(mat,axis = 0)
    averageOfP = mat/numpy.sum(weight)
    return averageOfP

def P_updater(P,net,M, status = "better"):
    P1 = []
    for i in range(len(net)):
        P1.append(P_updater_body(i, P, net, M, status))
    
    P1 = numpy.array(P1)
    # P1 = P1.astype('object')
    # for i in range(len(P1)):
    #     for j in range(len(P1[i,:])):
    #         P1[i,j] = sympy.Rational(P1[i,j])
    return P1

#(b) The selection of unstable vertexes

def unstable_vertexes(vs, P):
    u = []
    for communityCounter in range(P.shape[1]):#look in all communities
        u.append((P[vs,communityCounter],communityCounter)) #it calculates the Uin
    return max(u)[1]

def selecting_maximum_membership_for_unstable_vertexes(vs,communities,P):
    cj = currentCummunity(vs,communities) # current community of the node
    ups = unstable_vertexes(vs, P)
    if cj is not ups:
        return [True,cj,ups]    #if it is an unstable node return True , current community ,
                                #the community that it should been there
    else:
        return [False,cj,ups]


#get the real communitois of football dataset
def foot():    
    f = open("dataset/Real networks/football.gml", "r")
    text = f.read()

    Vals = []
    for i,letters in enumerate(text):
        if letters == 'e' and text[i-1] == 'u' and text[i-2] == 'l':
            Vals.append(text[i+2]+text[i+3])

    for i in range(len(Vals)):
        Vals[i] = int(re.findall(r'[0-9]+',Vals[i])[0])

    communities = []
    for l in range(len(numpy.unique(Vals))):
        communities.append([])
    for i,vals in enumerate(Vals):
        communities[vals].append(i)

    return communities

#and poolbook dataset...
def pool():
    f = open("dataset/Real networks/polbooks.gml", "r")
    text = f.read()

    vals = []
    for i, letters in enumerate(text):
        if letters == 'e' and text[i - 1] == 'u' and text[i - 2] == 'l':
            vals.append(text[i + 3])

    communitis = [[], [], []]
    for i, v in enumerate(vals):
        if v == 'n':
            communitis[0].append(i)
        elif v == 'c':
            communitis[1].append(i)
        elif v == 'l':
            communitis[2].append(i)
        else:
            print('error')

    return communitis

def karate():
    communities = [[0,11,17,12,21,19,1,7,3,4,10,13,6,16,5,2],[8,9,33,18,20,14,22,26,28,15,29,32,30,27,24,23,25,31]]

    return communities

####to get the separation in the clusters....
####it just works on undirected networks

def sep(p,mmin):
    siz = p.shape
    p = p.astype(float)
    if mmin :
        minn = numpy.inf
        for i in range(siz[1]-1):
            for j in range(i+1, siz[1]):
                dist = numpy.linalg.norm(p[:,i]-p[:,j])
                if minn > dist :
                    minn = dist
    else :
        minn = []
        for i in range(siz[0]-1):
            for j in range(i+1, siz[0]):
                minn.append(numpy.linalg.norm(p[i,:]-p[j,:]))

        minn = numpy.mean(minn)

    p = p.astype('object')
    for i in range(len(p)):
        for j in range(len(p[i,:])):
            p[i,j] = sympy.Rational(p[i,j])

    return minn


def saveiit(net):
    networkx.draw(net)
    fig = plt.gcf()
    fig.set_size_inches(18.5 * 10, 10.5 * 10)
    fig.savefig('test2png.png', dpi=100)


class LFR:
    def lfrMacker(n = 500,max_community = 200):
        """
        it returns a net = the LFR network
        newCom = the communities of this networks
        noodesCommunites = says each node belong to what community
        """
        net = networkx.algorithms.community.community_generators.LFR_benchmark_graph(n, 6, 3, .2, min_degree=5,
                                                                                    max_community=max_community)
        newcom = []
        for v in net:
            newcom.append(net.nodes[v]['community'])
        
        newcom = numpy.unique(newcom)
        
        nodesCommunities = []
        for nodes in net:
            nodesCommunities.append([])
            for i,coms in enumerate(newcom):
                if nodes in coms:
                    nodesCommunities[nodes].append(i)
        
        return (net, newcom, nodesCommunities)

    def pij(net, nodesCommunities, newcom):
        p = numpy.zeros([len(net.nodes),len(newcom)])
        
        for nodes in net:
            for i, community in enumerate(newcom):
                neighbors = list(net.neighbors(nodes))
                neighbors.append(nodes)
                nomonator = 0
                for nei in neighbors:
                    if nei in community:
                        nomonator += 1/len(nodesCommunities[nodes])
                
                denomonator = len(neighbors)
                
                p[nodes,i] = nomonator/denomonator
        
        return p
    
    def cos(mat):
        out = numpy.zeros([len(mat),len(mat)])
        for i,m1 in enumerate(mat):
            for j,m2 in enumerate(mat):
                out[i,j] = 1 - spatial.distance.cosine(m1,m2)
        
        return out

    def SFEC(simrowU, simrowP):
        erorr = (simrowU - simrowP)/len(simrowP)**2
        return erorr.sum()


class evaluation:
    def sep(p):
        siz = p.shape
        p = p.astype(float)
        minn = numpy.inf
        for i in range(siz[1] - 1):
            for j in range(i + 1, siz[1]):
                dist = numpy.linalg.norm(p[:, i] - p[:, j])
                if minn > dist:
                    minn = dist

        # p = p.astype('object')
        #
        # for i in range(len(p)):
        #     for j in range(len(p[i, :])):
        #         p[i, j] = sympy.Rational(p[i, j])

        return minn

    def comp(net, u, communities):
        p = numpy.zeros([len(net.nodes), len(communities)])

        for nodes in net:
            neighbors = list(net.neighbors(nodes))

            temp = u[neighbors,:] #to get the nodes that belong to multiple communities

            newp = temp.sum(axis=0)
            newp = newp / len(neighbors)

            p[nodes,:] = newp

        temp = numpy.abs(p - u)


        return temp.sum()/(len(communities)*len(net.nodes))

    def UFEC(net, u, communities):

        se = evaluation.sep(u)
        co = evaluation.comp(net, u, communities)

        return co/se

def new_weight1(p,net,communities):
    #make the p matrix
    new_p = p.copy()
    for i in range(len(net)):
        temp_p = p[list(net[i]),:]
        new_p[i] = numpy.mean(temp_p,axis=0)

    #apply the weights
    p = new_p

    for i in range(len(net)):
        for k in range(len(communities)):
            p[i,k] *= new_weight1_body(i,k,net,communities)

    #normalize the matrix
    row_sums = p.sum(axis=1)
    p = p / row_sums[:, numpy.newaxis]

    return p


def new_weight1_body(node, community, net, communities):
    # works on neighbors of neighbors
    neighbors = list(net[node])

    out = []

    for neis in neighbors:
        counter = 0
        neighbors_of_negh = list(net[neis])
        total = len(neighbors_of_negh)
        for neisofneis in neighbors_of_negh:
            if neisofneis in communities[community]:
                counter+=1

        out.append(counter/total)

    return numpy.mean(out)

def new_weight1_body1(node, community, net, communities):
    #works only on neighbors
    neighbors = list(net[node])

    total = len(neighbors)
    for neis in neighbors:
        counter = 0
        if neis in communities[community]:
            counter+=1

    return counter/total

def new_weight2(p,net):
    #make the p matrix
    new_p = p.copy()
    for i in range(len(net)):
        temp_p = p[list(net[i]),:]
        new_p[i] = numpy.mean(temp_p,axis=0)

    p = new_p

    for i in range(len(net)):
        p[i] *= new_weight2_body(i,net,p)

    row_sums = p.sum(axis=1)
    p = p / row_sums[:, numpy.newaxis]
    return p

def new_weight2_body(node, net,p):
    neighbors = list(net[node])

    out = []
    for neis in neighbors:
        neighbors_of_neighbors = list(net[neis])
        out.append(p[neighbors_of_neighbors,:])

    weight = numpy.zeros([len(out),p.shape[1]])
    for i,o in enumerate(out):
        weight[i] = numpy.mean(o,axis=0)

    return numpy.mean(weight,axis=0)

def save(file,name):
    theFile = open(name, "wb")  # it says to write in bite
    pickle.dump(file, theFile)
    theFile.close()

def load(name):
    pickleIN = open(name, "rb")  # says read it to bite
    out = pickle.load(pickleIN)
    pickleIN.close()
    return out

def log(string,name):
    File_object = open(name, "a+")
    string +="\n"
    File_object.write(string)
    File_object.close()