import numpy
import networkx
import pickle
import re

class netProperation :

    def netLoader(string):

        if string == "football":
            PATH = "dataset/Real networks/football.gml"
            net = networkx.read_gml(PATH, label='id')
            M = networkx.adjacency_matrix(net)
            M = M.todense()
            realCommunities = foot(PATH)

        elif string == "youtube":
            net = networkx.Graph()
            PATH = "dataset/Real networks/youtube.txt"
            file = open(PATH, 'r')
            with file:
                for line in file:
                    p = re.findall('\d+', line)
                    net.add_edge(int(p[0]), int(p[1]))
            file.close()

            M = networkx.adjacency_matrix(net)
            M = M.todense()
            net = networkx.from_numpy_matrix(M)
            realCommunities = "Null"

        elif string == "facebook":
            net = networkx.Graph()
            PATH = "dataset/Real networks/facebook_combined.txt"
            file = open(PATH, 'r')
            with file:
                for line in file:
                    p = re.findall('\d+', line)
                    net.add_edge(int(p[0]), int(p[1]))
            file.close()

            M = networkx.adjacency_matrix(net)
            M = M.todense()
            net = networkx.from_numpy_matrix(M)
            realCommunities = "Null"

        elif string == "karate":
            PATH = "dataset/Real networks/karate.gml"
            net = networkx.read_gml(PATH, label='id')
            dic = {}
            for i in range(34):
                dic[i+1]=i
            net = networkx.relabel_nodes(net,dic)
            M = networkx.adjacency_matrix(net)
            M = M.todense()
            realCommunities = karate()

        elif string == "dolphons":
            PATH = "dataset/Real networks/dolphons.gml"
            net = networkx.read_gml(PATH, label='id')
            M = networkx.adjacency_matrix(net)
            M = M.todense()

        elif string == "polbooks":
            PATH = "dataset/Real networks/polbooks.gml"
            net = networkx.read_gml(PATH, label='id')
            M = networkx.adjacency_matrix(net)
            M = M.todense()
            realCommunities = pool(PATH)

        elif string == "net1":
            PATH = "dataset/artifitial networks/GN networks/net1/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "net2":
            PATH = "dataset/artifitial networks/GN networks/net2/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "net3":
            PATH = "dataset/artifitial networks/GN networks/net3/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "net4":
            PATH = "dataset/artifitial networks/GN networks/net4/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "net5":
            PATH = "dataset/artifitial networks/GN networks/net5/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "net6":
            PATH = "dataset/artifitial networks/GN networks/net6/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "net7":
            PATH = "dataset/artifitial networks/GN networks/net7/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "net8":
            PATH = "dataset/artifitial networks/GN networks/net8/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "lfr1":
            PATH = "dataset/artifitial networks/LFR/LFR1/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "lfr2":
            PATH = "dataset/artifitial networks/LFR/LFR2/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "lfr3":
            PATH = "dataset/artifitial networks/LFR/LFR3/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "lfr4":
            PATH = "dataset/artifitial networks/LFR/LFR4/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "lfr5":
            PATH = "dataset/artifitial networks/LFR/LFR5/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        elif string == "lfr6":
            PATH = "dataset/artifitial networks/LFR/LFR6/others.p"
            pickleIN = open(PATH, "rb")  # says read it to bite
            (sequences, F, net, M, realCommunities) = pickle.load(pickleIN)
            pickleIN.close()

        else:
            print("we don't have that network man !!!!!")
            exit()

        return net, M, realCommunities



    def netsort(net):
        out = [len(list(net[node])) for node in net]
        out = numpy.argsort(out)

        return out[::-1]

def foot(path):
    f = open(path, "r")
    text = f.read()

    Vals = []
    for i, letters in enumerate(text):
        if letters == 'e' and text[i - 1] == 'u' and text[i - 2] == 'l':
            Vals.append(text[i + 2] + text[i + 3])

    for i in range(len(Vals)):
        Vals[i] = int(re.findall(r'[0-9]+', Vals[i])[0])

    communities = []
    for l in range(len(numpy.unique(Vals))):
        communities.append([])
    for i, vals in enumerate(Vals):
        communities[vals].append(i)

    return communities

# and poolbook dataset...
def pool(path):
    f = open(path, "r")
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
    communities = [[0, 11, 17, 12, 21, 19, 1, 7, 3, 4, 10, 13, 6, 16, 5, 2],
                   [8, 9, 33, 18, 20, 14, 22, 26, 28, 15, 29, 32, 30, 27, 24, 23, 25, 31]]

    return communities

def chooseImportants(net,sortedNodes,edges = "counts"):

    if edges == "counts":
        hiddenNeigbors = []
        importantNodes = []
        for nodes in sortedNodes:
            if nodes in hiddenNeigbors:
                continue

            importantNodes.append(nodes)
            hiddenNeigbors += list(net[nodes])
            hiddenNeigbors = list(set(hiddenNeigbors))

        return hiddenNeigbors, importantNodes

    else:
        copynet = net.copy()
        hiddenNeigbors = []
        importantNodes = []

        while len(sortedNodes) > 0:
            nodes = sortedNodes[0]
            sortedNodes = sortedNodes[1:]
            if nodes in hiddenNeigbors or nodes in importantNodes:
                continue

            importantNodes.append(nodes)
            hiddenNeigbors += list(copynet[nodes])
            hiddenNeigbors = list(set(hiddenNeigbors))

            copynet.remove_node(nodes)
            sortedNodes = netProperation.netsort(copynet)
            sumOfTheNodes = set(hiddenNeigbors) | set(importantNodes)
            sortedNodes = list(set(sortedNodes) - sumOfTheNodes)

        return hiddenNeigbors, importantNodes

def nmi(communities):
    m = max(sum(communities, []))
    after = numpy.ones(m + 1)
    for i, community in enumerate(communities):
        for node in community:
            after[node] = i

    return after

def Umaker(net,communities):
    u = numpy.zeros([len(net), len(communities)])
    for i, community in enumerate(communities):
        u[community, i] = 1

    return u