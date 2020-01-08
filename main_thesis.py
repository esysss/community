import networkx
import algos
import otherFuns


def lpa(net):
    LPcommunities = networkx.algorithms.community.label_propagation_communities(net)
    LPcommunities = [list(lp) for lp in list(LPcommunities)]
    return LPcommunities

def cmeans(net,number_of_communities):
    u, CMConnumities = algos.CMeans(net, number_of_communities)
    CMConnumities = [list(a) for a in CMConnumities]

    return u,CMConnumities

def nmf(net,number_of_communities,tol=1e-4):
    nmfCommunities, u = algos.NMFf(net, number_of_communities,tol=tol)

    return u, nmfCommunities

def lpa_fcm(net,thresholdCondition,status_threshold):
    communities = algos.lpa_fcm(net,thresholdCondition,status_threshold)

    return communities

def lpa_rb(net,status="worst",alpha=0.5):
    P,communities = algos.lpa_rb(net, status, alpha)

    return P,communities

def lpa_rb2(net, weight = 3):
    p,communitiese = algos.lpa_rb2(net, weight)

    return p,communitiese

def ufec(net,u,communities):
    otherFuns.evaluation.UFEC(net,u,communities)