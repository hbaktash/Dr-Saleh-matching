import random
import math
import numpy
import scipy.optimize
import networkx


# import matplotlib

def dist(v1, v2):
    return math.sqrt(sum((v1 - v2) * (v1 - v2)))


def embedding(edges, k):
    n = edges.shape[0]
    D = numpy.diag(numpy.sum(edges, axis=1))
    laplacian = D - edges
    w, v = numpy.linalg.eig(laplacian)

    # w2 = [(w[i], i) for i in range(len(w))]
    # w2.sort(key=lambda x: x[0])
    # sortedW, permu = zip(*w2)
    # permMatrix = numpy.zeros((len(w), len(w)))
    # for i in range(len(w)):
    #     permMatrix[permu[i]][i] = 1
    # v = numpy.array(numpy.matmul(v, permMatrix))

    wv = numpy.zeros((n + 1, n))
    wv[1:, :] = v
    wv[0, :] = w
    vNew = numpy.array(sorted(wv.transpose(), key=lambda x: x[0])).transpose()
    # print("sorted eigs and vals:", vNew)
    v = vNew[1:, :]
    for i in range(n):
        if (numpy.sum(v[:, i] > 0) <= (len(w) / 2)):
            v[:, i] = -v[:, i]
    feature = numpy.array(v[:, -k:])
    # print("features:",feature )
    D = numpy.array(numpy.sum(edges, axis=1))
    vals = numpy.array(numpy.reshape(vNew[0, -k:], (1, k)))
    return feature, D, vals


def produceG1G2(n, edges, s):
    g1 = numpy.array(edges)
    g2 = numpy.array(edges)
    deletes = 0
    totalEdges = 0
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
                    if (ran1 - s) * (ran2 - s) <= 0:
                        deletes += 1
    # print("total edges: ", totalEdges)
    # print("differance in edges of g1 g2: ", deletes)
    return g1, g2


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


def greedyMatch(size, costMatrix):
    matchingPerm = numpy.zeros(size)
    matched = numpy.zeros(size)
    for i in range(size):
        min = 1000000
        tempJ = 0
        for j in range(size):
            if (matched[j] == 0) & (costMatrix[i][j] < min):
                min = costMatrix[i][j]
                tempJ = j
        matchingPerm[i] = tempJ
        matched[tempJ] = 1
    return matchingPerm


def produceAndMatch(n, edges, p, s, k, weights):
    g1, g2 = produceG1G2(n, edges, s)
    # permute now
    permutation = numpy.random.permutation(n)
    p = numpy.zeros((n, n))
    for i in range(n):
        p[i][permutation[i]] = 1
    g2 = numpy.matmul(p.transpose(), numpy.matmul(g2, p))
    # extract features
    feature1, D1 = embedding(g1, k)
    feature2, D2 = embedding(g2, k)
    # weight applied
    feature1 = numpy.array(numpy.repeat(weights, n, axis=0) * feature1)
    feature2 = numpy.array(numpy.repeat(weights, n, axis=0) * feature2)
    print("computing costMatrix")
    costMatrix = numpy.array([[dist(feature1[i, :], feature2[j, :]) for j in range(n)] for i in range(n)])
    print("matching:...")
    # colInd = greedyMatch(n, costMatrix)
    colInd = greedymatchN3(n, costMatrix)
    # rowInd, colInd = scipy.optimize.linear_sum_assignment(costMatrix)
    # print("cost = ")
    # print(sum([costMatrix[i][colInd[i]] for i in range(n)]))
    return colInd, permutation


def produceAndMatchWithDegs(n, edges, s, k, weights, highDegNum):
    g1, g2 = produceG1G2(n, edges, s)
    # permute now
    permutation = numpy.random.permutation(n)
    p = numpy.zeros((n, n))
    for i in range(n):
        p[i][permutation[i]] = 1
    g2 = numpy.matmul(p.transpose(), numpy.matmul(g2, p))
    # print(g1)
    # print(g2)
    # extract features
    feature11, D1, vals1 = embedding(g1, k)
    feature21, D2, vals2 = embedding(g2, k)
    # weight applied

    # print("shape of val", vals1.shape)
    feature1 = numpy.array(numpy.repeat(vals1, n, axis=0) * feature11)
    feature2 = numpy.array(numpy.repeat(vals2, n, axis=0) * feature21)
    # print("weighting is bugged if not zero:", feature11 - feature1)

    f1, f2, permu = sortDegs(n, feature1, D1, feature2, D2, permutation)
    # print("permu after sorted", permu)
    diff = matchHighDegs(n, f1, f2, permu, highDegNum)
    m = highDegNum
    # print("a maximum of ", numpy.sum(permu[-m:] >= n-m), " could be matched")
    return diff, numpy.sum(permu[-m:] >= n - m)


def matchHighDegs(n, f1, f2, permu, highDegNum):
    m = highDegNum
    newf1 = f1[-m:, :]
    newf2 = f2[-m:, :]
    costMatrix = numpy.array([[dist(newf1[i, :], newf2[j, :]) for j in range(m)] for i in range(m)])
    # colInd = greedyMatch(m, costMatrix)
    colInd = greedymatchN3(m, costMatrix)
    # rowInd, colInd = scipy.optimize.linear_sum_assignment(costMatrix)
    colInd = colInd + (n - m)
    # print("aligns:", colInd)
    # print("colInd: ", colInd)
    # print("shape of permu: ",permu.shape)
    # print(" permu last ms: ", permu[-m:])
    diff = numpy.array(colInd - permu[-m:])
    # corrects = numpy.sum(diff == 0)
    return diff


def sortDegs(n, feature1, D1, feature2, D2, permu):
    D1 = numpy.reshape(D1, (n, 1))
    D2 = numpy.reshape(D2, (n, 1))
    permut = numpy.reshape(permu, (n, 1))
    g1 = numpy.array(numpy.hstack((D1, permut, feature1)))
    g1 = numpy.array(sorted(g1, key=lambda x: x[0]))
    g2 = numpy.array(numpy.hstack((D2, numpy.reshape(range(n), (n, 1)), feature2)))
    g2 = numpy.array(sorted(g2, key=lambda x: x[0]))
    # print("checking permuts:", permut, "\n", range(n))
    perm2T = numpy.array(g2[:, 1]).astype(int)
    perm1T = numpy.array(g1[:, 1]).astype(int)
    # print(perm2T)
    newPerm = numpy.zeros(n)
    for i in range(n):
        newPerm[perm2T[i]] = i
    # print("new perm", newPerm)
    # print("perm1T", perm1T)
    permNew = numpy.array([newPerm[perm1T[i]] for i in range(n)])
    # print("new permu:",permNew)
    # print("the new perm: ",permNew)

    return g1[:, 2:], g2[:, 2:], permNew


def initialGraph(n, p):
    edges = numpy.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i < j:
                ran = random.uniform(0, 1)
                if ran < p:
                    edges[i, j] = 1
                    edges[j, i] = 1
    # print(edges)
    return edges


def testFullMatching(n, m, p, s, k, weights):
    stats = numpy.zeros(m)
    for i in range(m):
        print("\n", i)
        edges = initialGraph(n, p, k)
        colInd, permutation = produceAndMatch(n, edges, p, s, k, weights)
        diff = numpy.array(colInd - permutation)
        # print(diff)
        correct_matches = numpy.sum((diff == 0))
        print(correct_matches)
        stats[i] = correct_matches / n
    return stats


def testHighDegMatching(n, m, p, s, k, weights, highDegNum):
    stats = numpy.zeros(m)
    x = 0
    for i in range(m):
        edges = initialGraph(n, p, k)
        diff, max = produceAndMatchWithDegs(n, edges, s, k, weights, highDegNum)
        if max == 0:
            max = 1
        stats[i] = numpy.sum(diff == 0) / highDegNum
        # stats[i] = numpy.sum(diff == 0) / highDegNum
        # print("corrects: ", numpy.sum(diff == 0), " out of ", int(max))
    return stats


n = int(input("graph size : "))
m = int(input("trials : "))
p = (3 * math.log(n, math.e)) / n
print("p = ", p)
k = int(input("k = "))
s = float(input("s = "))
highDegMax = int(input("number of high degs up to: "))
for i in range(highDegMax):
    weights = numpy.array([1] * k).reshape(1, k)
    numpy.set_printoptions(precision=3)
    stats = testHighDegMatching(n, m, p, s, k, weights, i + 1)
    # stats = testFullMatching(n,m,p,s,k,weights)
    hist, bin = numpy.histogram(stats)
    print(n, " nodes with s = ", s, "\n", " high degs : ", i + 1)
    print("hist and bin\n", hist, "\n", bin)
    print("average matches = ",numpy.average(stats))
    print("100%:", numpy.sum(stats == 1), "out of ", m, " attempts")
