import numpy
import The_assignment_of_initial_community_labels as first
import others

#(a) The establishment of membership vector
def C_mean_membership_vector(community,n,F):
    '''
    :param i: the community address
    :param n: the node
    :param F: neighbor evaluation vector
    :return: C-mean membership of the node in the community (the u)
    '''
    neighbor = [] #get the neighbors of node n
    [neighbor.append(row[1]) for row in F[n]]
    sumf1 = 0     #the Numerator of the fraction
    for i in list(set(community).intersection(set(neighbor))):
        for j in F[n]:
            if i==j[1]:
                sumf1 += j[0]
    a = []
    for i in F[n]:
        a.append(i[0])
    sumf2 = numpy.sum(a)      #the denominator of the fraction

    return sumf1/sumf2

#(b) The selection of unstable vertexes
def unstable_vertexes(vs, communities, F):
    u = []
    for i,community in enumerate(communities):#look in all communities
        u.append((C_mean_membership_vector(community,vs,F),i)) #it calculates the Uin
    return max(u)[1]


#(c) Labels modification
def selecting_maximum_membership_for_unstable_vertexes(vs,communities,F):
    cj = others.currentCummunity(vs,communities) # current community of the node
    ups = unstable_vertexes(vs, communities, F)
    if cj is not ups:
        # communities[cj].remove(vs)
        # communities[ups].append(vs)
        # return communities
        return [True,cj,ups]    #if it is a isolated node return True , current community , the community that it should been there
    else:
        return [False,cj,ups]
