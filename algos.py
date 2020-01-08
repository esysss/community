import networkx
import numpy
from sklearn.decomposition import NMF
import skfuzzy as fuzz
import others
import The_assignment_of_initial_community_labels as first
import The_modification_of_vertexes_labels as second
import Parameters_update as third
import otherFuns
from tqdm import tqdm
import time
import extras


def LPANB(net):
    print('')
    return 'hi'

def NMFf(net,numberOfCommunities,tol=1e-4):
    M = networkx.adjacency_matrix(net)
    M = M.todense()  # the sparse adjacency matrix of our network
    model = NMF(n_components=numberOfCommunities, init='random', random_state=0, max_iter=200,tol=tol)

    W = model.fit_transform(M)
    H = model.components_
    communities = []
    for i, arr in enumerate(H.T):
        communities.append(numpy.argmax(arr))

    q = []
    [q.append([]) for i in range(max(communities) + 1)]
    for i, arr in enumerate(communities):
        q[arr].append(i)

    return q

def CMeans(net, clusterNumbers):
    M = networkx.adjacency_matrix(net)
    M = M.todense()  # the sparse adjacency matrix of our network

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        M, clusterNumbers, 2, error=0.005, maxiter=1000, init=None)
    cluster_membership = numpy.argmax(u, axis=0)

    communities = []
    for i in range(clusterNumbers):
        communities.append(numpy.where(cluster_membership == i)[0])

    return u,communities

def lpa_fcm(net,thresholdCondition,status_threshold):
    M = networkx.adjacency_matrix(net)
    M = M.todense()  # the sparse adjacency matrix of our network

    # table 1:

    F = first.neighbor_evaluation_vector(M)  # Step1: calculate otherNodes evaluation vectors
    sequences = first.Order_access_sequence(F)  # Step2: generate order access sequences

    # thresholdCondition = 1
    communities = []
    communityCounter = -1
    sequenceListPast = []
    for sequence in sequences:  # Step3
        sequenceListPast.append(sequence)
        if not others.in_it(sequence, communities):
            communities.append([])
            communityCounter += 1
            communities[communityCounter].append(sequence)
            currentC = communityCounter
        else:
            currentC = others.currentCummunity(sequence, communities)

        for otherNodes in sequenceListPast:
            DVi = first.average_neighbor_distance(sequence, M, F)
            DVj = first.average_neighbor_distance(otherNodes, M, F)
            DVij = first.average_mutual_neighbor_distance(sequence, otherNodes, M, F)
            if abs(DVij - (DVi + DVj) / 2) <= thresholdCondition:
                pervC = others.currentCummunity(otherNodes,
                                                communities)  # find the previous community label and delete it
                if pervC is not None:
                    communities[pervC].remove(otherNodes)

                communities[currentC].append(otherNodes)

    isolate = first.isolate_nodes(communities)  # step4
    if len(isolate) > 0:
        for nodes in isolate:  # step5
            mindist = first.minimum_average_distance(nodes, communities, M)
            pervC = others.currentCummunity(nodes, communities)  # find the previous community label and delete it
            communities[pervC].remove(nodes)
            communities[mindist].append(nodes)
    communities = others.not_empty(communities)

    '''
    table 2 ...
    '''

    # status_threshold = 4.4
    objfP = float('inf')
    objfC = third.JtKLFCM(communities, F, M)  # step 1  #current objective function

    while True:  # step 2
        if not abs(objfP - objfC) < status_threshold:
            unstables = []
            for community in communities:
                for node in community:  # select the unstable vertexes and revise the labels
                    Isunasteble = second. \
                        selecting_maximum_membership_for_unstable_vertexes(node, communities, F)
                    if Isunasteble[0]:
                        unstables.append((node, Isunasteble[1], Isunasteble[2]))

            for nodes in unstables:
                communities[nodes[1]].remove(nodes[0])
                communities[nodes[2]].append(nodes[0])

            objfP = objfC
            objfC = third.JtKLFCM(communities, F, M)
        else:
            break

    communities = others.not_empty(communities)

    return communities

def lpa_rb(net, status, alpha):
    M = networkx.adjacency_matrix(net)
    M = M.todense()  # the sparse adjacency matrix of our network

    # table 1:

    F = first.neighbor_evaluation_vector(M)  # Step1: calculate otherNodes evaluation vectors
    sequences = first.Order_access_sequence(F)  # Step2: generate order access sequences

    # thresholdCondition = 1
    communities = []
    communityCounter = -1
    sequenceListPast = []
    for sequence in sequences:  # Step3
        sequenceListPast.append(sequence)
        if not others.in_it(sequence, communities):
            communities.append([])
            communityCounter += 1
            communities[communityCounter].append(sequence)
            currentC = communityCounter
        else:
            currentC = others.currentCummunity(sequence, communities)

        for otherNodes in sequenceListPast:
            DVi = first.average_neighbor_distance(sequence, M, F)
            DVj = first.average_neighbor_distance(otherNodes, M, F)
            DVij = first.average_mutual_neighbor_distance(sequence, otherNodes, M, F)

            thresholdCondition = DVij * alpha / 100
            if abs(DVij - (DVi + DVj) / 2) <= thresholdCondition:
                pervC = others.currentCummunity(otherNodes,
                                                communities)  # find the previous community label and delete it
                if pervC is not None:
                    communities[pervC].remove(otherNodes)

                communities[currentC].append(otherNodes)

    isolate = first.isolate_nodes(communities)  # step4
    if len(isolate) > 0:
        for nodes in isolate:  # step5
            mindist = otherFuns.minimum_average_distance(nodes, communities, M)
            pervC = otherFuns.currentCummunity(nodes, communities)  # find the previous community label and delete it
            communities[pervC].remove(nodes)
            communities[mindist].append(nodes)

    communities = otherFuns.not_empty(communities)

    U = otherFuns.UMaker(communities, M, F)
    P = otherFuns.P_maker(U, F, M, status)  # best or worst

    for i in range(50):
        # print(f"we are in loop {i}")
        unstables = []
        for community in communities:
            for node in community:  # select the unstable vertexes and revise the labels
                Isunasteble = otherFuns.selecting_maximum_membership_for_unstable_vertexes(node, communities, P)
                if Isunasteble[0]:
                    unstables.append((node, Isunasteble[1], Isunasteble[2]))

        for nodes in unstables:
            communities[nodes[1]].remove(nodes[0])
            communities[nodes[2]].append(nodes[0])

        P = otherFuns.P_updater(P, F, M, status)  # or worst
        shouldDelet = []
        for i, community in enumerate(communities):
            if len(community) == 0:
                for j, p in enumerate(P):
                    temp = P[j, i]
                    P[j, i] = 0
                    s = numpy.sum(P[j, :])
                    for reminders in range(len(p)):
                        P[j, reminders] += P[j, reminders] / s * temp
                shouldDelet.append(i)

        P = numpy.delete(P, shouldDelet, axis=1)

        communities = otherFuns.not_empty(communities)

    return P,communities

def lpa_rb2(net,weight):

    M = networkx.adjacency_matrix(net)
    M = M.todense()

    start = time.time()

    sortedNodes = extras.netProperation.netsort(net)

    neighbours, importantNodes = extras.chooseImportants(net, sortedNodes)  # , edges = "NOT-counts"

    communities = [[i] for i in importantNodes]

    print("let's do the first part")
    for nodes in tqdm(neighbours):
        distances = [networkx.shortest_path_length(net, source=nodes, target=im) for im in importantNodes]
        communities[numpy.array(distances).argmin()].append(nodes)

#this part can be deleted !!!
    isolate = otherFuns.isolate_nodes(communities)

    if len(isolate) > 0:
        for nodes in isolate:  # step5
            mindist = otherFuns.minimum_average_distance(nodes, communities, M)
            pervC = otherFuns.currentCummunity(nodes, communities)  # find the previous community label and delete it
            communities[pervC].remove(nodes)
            communities[mindist].append(nodes)

    communities = otherFuns.not_empty(communities)

    u = extras.Umaker(net, communities)

    if weight == 1:
        P = otherFuns.new_weight1(u, net, communities)
    elif weight == 2:
        P = otherFuns.new_weight2(u, net)
    else:
        P = otherFuns.P_updater(u, net, M)

    print("let's do the second part")
    for i in tqdm(range(50)):
        # print(f"we are in loop {i}")
        unstables = []
        for community in communities:
            for node in community:  # select the unstable vertexes and revise the labels
                Isunasteble = otherFuns.selecting_maximum_membership_for_unstable_vertexes(node, communities, P)
                if Isunasteble[0]:
                    unstables.append((node, Isunasteble[1], Isunasteble[2]))

        for nodes in unstables:
            communities[nodes[1]].remove(nodes[0])
            communities[nodes[2]].append(nodes[0])

        u = extras.Umaker(net, communities)
        if weight == 1:
            P = otherFuns.new_weight1(u, net, communities)
        elif weight == 2:
            P = otherFuns.new_weight2(u, net)
        else:
            P = otherFuns.P_updater(u, net, M)

        shouldDelet = []
        for i, community in enumerate(communities):
            if len(community) == 0:
                for j, p in enumerate(P):
                    temp = P[j, i]
                    P[j, i] = 0
                    s = numpy.sum(P[j, :])
                    for reminders in range(len(p)):
                        P[j, reminders] += P[j, reminders] / s * temp
                shouldDelet.append(i)

        P = numpy.delete(P, shouldDelet, axis=1)

        communities = otherFuns.not_empty(communities)

    finished = (time.time() - start) / 3600
    print(f"all the computations got just {finished} hours")

    return P,communities