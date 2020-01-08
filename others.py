import networkx
import matplotlib.pyplot as plt
import community

def in_it(item,to_search):# is the node in the community?
    for row in range(len(to_search)):
        for column in range(len(to_search[row])):
            if item == to_search[row][column]:
                return True
    return False

def currentCummunity(item,communities):#returns the current community of a node
    for i,community in enumerate(communities):
        for node in community:
            if item == node:
                return i

#to check if any empety community is there
def not_empty(communities): # deletes the empty communities
    newCommunities = []
    for i in communities:
        if len(i) > 0:
            newCommunities.append(i)

    return newCommunities

def draw(net,communities,pos):
    color = ['b','g','r','c','m','y','k','w','#2763c4','#c40daf','#720a0a','#ffff16']
    #communities = [[val + 1 for val in lis] for lis in communities]
    for i in range(len(communities)):
        networkx.draw_networkx(net,pos,nodelist=communities[i],node_color = color[i])
    # plt.savefig('gn.png')
    plt.show()
    plt.close()

def modd(communities,net):
    dick = {}
    #communities = [[val + 1 for val in lis] for lis in communities]
    for i, comunity in enumerate(communities):
        for nodes in comunity:
            dick[nodes] = i

    return community.modularity(dick,net)