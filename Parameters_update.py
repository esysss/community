import numpy
import math
import The_modification_of_vertexes_labels as second
import The_assignment_of_initial_community_labels as first

#(a) Cut-off condition
def pii(node, community, F):
    neighbor = []
    [neighbor.append(q[1]) for q in F[node]] #q has two component (evaluation vector and the number of node)
    NbjCi = list(set(neighbor).intersection(set(community)))
    sumu = 0
    for node in NbjCi:
        sumu += second.C_mean_membership_vector(community, node, F)
    if len(NbjCi) is 0:
        return float('inf')
    else:
        return sumu/len(NbjCi)

def FDij(community,j,M):
    tempMat = M[:,community]
    meau = numpy.mean(tempMat,axis=1)
    Mj = M[:, j]
    nominator = numpy.linalg.norm(Mj-meau)**2
    varia = numpy.sum(numpy.var(tempMat,axis=1))
    ret = .5 * numpy.log10(2*numpy.pi) + numpy.log10(varia) + nominator/(2*varia**2)
    if math.isnan(ret):
        return 0
    else:
        return ret

def JtKLFCM(communities,F,M):
    firstPart = 0
    for community in communities:
        for node in community:
            Uij = second.C_mean_membership_vector(community,node,F)
            Fdij = FDij(community,node,M)
            firstPart += Uij*Fdij

    secondPart = 0
    for community in communities:
        for node in community:
            Uij = second.C_mean_membership_vector(community, node, F)
            lg = Uij/pii(node,community,F)
            secondPart += Uij*lg

    return firstPart + secondPart