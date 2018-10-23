import random
import math
import numpy


def produceG1G2(n, edges, s):
    g1 = numpy.array(edges)
    g2 = numpy.array(edges)
    deletes = 0
    totalEdges = 0
    print("s: ",s)
    for i in range(n):
        for j in range(n):
            if (j > i):
                if g1[i][j] == 1:
                    totalEdges += 1
                    ran1 = random.uniform(0, 1)
                    if ran1 > s:
                        g1[i][j] = 0
                        g1[j][i] = 0
                    ran2 = random.uniform(0, 1)
                    if ran2 > s:
                        g2[i][j] = 0
                        g2[j][i] = 0
                    if (ran1 - s)*(ran2-s) < 0:
                        deletes += 1
    print("total edges: ", totalEdges)
    print("differance in edges of g1 g2: ", deletes)
    return g1, g2


def sortDegs(n, g1, g2, permu):
    D1 = numpy.reshape(numpy.array(numpy.sum(g1, axis=1)), (n, 1))
    D2 = numpy.reshape(numpy.array(numpy.sum(g2, axis=1)), (n, 1))
    # print("D1 avval:", D1.transpose())
    # print("D2 dovvom:", D2.transpose())
    # print("correct permu: ",permu)
    D12 = [(D1[i], permu[i], i) for i in range(n)]
    D12.sort(key=lambda x: x[0])
    # print("D12:", D12)
    sortedD1, permuNew, permu1 = zip(*D12)
    permMatrix = numpy.zeros((n, n))
    for i in range(n):
        permMatrix[permu1[i]][i] = 1
    g12 = numpy.array(numpy.matmul(permMatrix.transpose(),numpy.matmul(g1, permMatrix)))
    # print("g12: ", g12)

    D22 = [(D2[i], i) for i in range(n)]
    D22.sort(key=lambda x: x[0])
    sortedD2, permu2 = zip(*D22)
    permMatrix2 = numpy.zeros((n, n))
    for i in range(n):
        permMatrix2[permu2[i]][i] = 1
    g22 = numpy.array(numpy.matmul(permMatrix2.transpose(), numpy.matmul(g2, permMatrix2)))

    # print("D22: ",D22)
    # print("D1: \n", sortedD1)
    # print("D2: \n", sortedD2)
    perm1T = numpy.array(permuNew).astype(int)
    perm2T = numpy.array(permu2).astype(int)
    # print("perm1T: ", perm1T)
    # print("perm2T: ",perm2T)
    # print(perm2T)
    newPerm = numpy.zeros(n)
    for i in range(n):
        newPerm[perm2T[i]] = i
    # print("new perm", newPerm)
    # print("perm1T", perm1T)
    permNew = numpy.array([newPerm[perm1T[i]] for i in range(n)])
    # print("new permu:",permNew)
    # print("correct perm new: ", permNew)
    return g12,g22, permNew


def initialGraph(n, p):
    edges = numpy.array([[0] * n] * n)
    for i in range(n):
        for j in range(n):
            if i < j:
                ran = random.uniform(0, 1)
                if ran < p:
                    edges[i][j] = 1
                    edges[j][i] = 1
    # print(edges)
    return edges

def dist(v1, v2):
    return int(numpy.sum(((v1 - v2 != 0))))


def greedyMatch(n, costMatrix):
    matchingPerm = numpy.zeros(n)
    matched = numpy.zeros(n)
    for i in range(n):
        min = 1000000
        tempJ = 0
        for j in range(n):
            if (matched[j] == 0) & (costMatrix[i][j] < min):
                min = costMatrix[i][j]
                tempJ = j
        matchingPerm[i] = tempJ
        matched[tempJ] = 1
    return matchingPerm


def greedymatchN3(n, costMatrix):
    matched2 = numpy.zeros(n)
    matched1 = numpy.zeros(n)
    matchingPerm = numpy.zeros(n)
    tempI = 0
    tempJ = 0
    for k in range(n):
        tempMin = 1000000
        for i in range(n):
            if matched1[i] == 0:
                for j in range(n):
                    if matched2[j] == 0:
                        if costMatrix[i][j] < tempMin:
                            tempMin = costMatrix[i][j]
                            tempI = i
                            tempJ = j
        matchingPerm[tempI] = tempJ
        matched1[tempI] = 1
        matched2[tempJ] = 1
    return matchingPerm


n = int(input("graph size: "))
m = int(input("trials: "))
p = (2 * math.log2(n)) / n
s = float(input("s = "))
k = int(input("k (high degs)= "))
sum = 0
for i in range(m):
    edges = initialGraph(n,p)
    g1, g2 = produceG1G2(n, edges, s)

    permutation = numpy.random.permutation(n)
    per = numpy.zeros((n, n))
    for i in range(n):
        per[i][permutation[i]] = 1
    g2 = numpy.matmul(per.transpose(), numpy.matmul(g2, per))

    g1, g2, perm = sortDegs(n, g1, g2, permutation)

    finalPerm = numpy.zeros(n)
    finalPerm[n - k : ] = numpy.array(range(k)) + (n - k)
    # print(finalPerm[-k : ])
    vectors1 = numpy.array(g1[0: n - k, n - k : ])
    vectors2 = numpy.array(g2[0: n - k, n - k : ])
    # print(vectors1.shape)
    # print(vectors2.shape)
    costMatrix = [[dist(vectors1[i,:],vectors2[j,:]) for j in range(n - k)] for i in range(n - k)]
    colInd = greedymatchN3(n - k, costMatrix)

    finalPerm[0 : n - k] = colInd
    # print("correct perm = " , perm)
    corrects = numpy.sum(finalPerm - perm == 0)
    print("corrects = ", corrects, "out of ",n)
    sum += corrects
print("avg corrects = ", sum / m)