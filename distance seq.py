import math

import networkx
import numpy
import random
import time
import scipy.optimize
import matplotlib.pyplot as plt
import stringdist
import zss


def frange(start, stop, dist):
    i = start
    while i >= stop:
        yield i
        i += dist

def greedy_match_n3(costMatrix):
    n = len(costMatrix)
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
    return numpy.array(range(n)), matchingPerm


def initial_graph(n, p):
    edges = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                ran = random.uniform(0, 1)
                if ran < p:
                    edges[i, j] = 1
                    edges[j, i] = 1
    return edges


def produce_g1_g2(n, edges, s):
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


def sortDegs(n, g1, g2, permu):
    D1 = numpy.reshape(numpy.array(numpy.sum(g1, axis=1)), (n, 1))
    D2 = numpy.reshape(numpy.array(numpy.sum(g2, axis=1)), (n, 1))

    D12 = [(D1[i], permu[i], i) for i in range(n)]
    D12.sort(key=lambda x: x[0])
    # print("D12:", D12)
    sortedD1, permuNew, permu1 = zip(*D12)
    permMatrix = numpy.zeros((n, n))
    for i in range(n):
        permMatrix[permu1[i]][i] = 1
    g12 = numpy.array(numpy.matmul(permMatrix.transpose(), numpy.matmul(g1, permMatrix)))
    # print("g12: ", g12)
    D22 = [(D2[i], i) for i in range(n)]
    D22.sort(key=lambda x: x[0])
    sortedD2, permu2 = zip(*D22)
    permMatrix2 = numpy.zeros((n, n))
    for i in range(n):
        permMatrix2[permu2[i]][i] = 1
    g22 = numpy.array(numpy.matmul(permMatrix2.transpose(), numpy.matmul(g2, permMatrix2)))

    perm1T = numpy.array(permuNew).astype(int)
    perm2T = numpy.array(permu2).astype(int)
    newPerm = numpy.zeros(n)
    for i in range(n):
        newPerm[perm2T[i]] = i
    permNew = numpy.array([newPerm[perm1T[i]] for i in range(n)])
    return g12, g22, permNew


def sub_graph(adj_mat, v, take_zeros=False):
    # print("-- making sub_graph")
    # print("degree of selected one: ", numpy.sum(adj_mat[v,:]))
    sub_ind = []
    for i in range(len(adj_mat)):
        if adj_mat[v, i] == 1:
            sub_ind.append(i)
    sub_ind = numpy.array(sub_ind)
    sub_adj_with_zeros = numpy.array(adj_mat[numpy.ix_(sub_ind, sub_ind)])
    if take_zeros == True:
        sub_degs = numpy.sum(sub_adj_with_zeros, axis=0)
        select_non_zero = numpy.array([i for i in range(len(sub_degs)) if sub_degs[i] != 0])
        sub_adj = sub_adj_with_zeros[numpy.ix_(select_non_zero, select_non_zero)]
    else:
        sub_adj = sub_adj_with_zeros
    # print("-- sub_graph size = ", len(select_non_zero))
    # networkx.draw(networkx.from_numpy_matrix(sub_adj))
    # plt.show()
    # time.sleep(1)
    return sub_adj, len(sub_adj_with_zeros), len(sub_adj)


# def edit_distance(g1, g2, max_iter):
#     # return edit_distance_exact(g1, g2)
#     # return edit_distance_iterative(g1, g2, max_iter)
#     # return ed_assignment_approx(g1, g2)
#     # return ed_spectral_approx(g1, g2)
#     return my_ed_approx(g1, g2)


def bfs_ed(full_g1 : numpy.ndarray, D1, v1, full_g2: numpy.ndarray, D2, v2):
    v1_neighs = numpy.array(range(len(full_g1)))[numpy.ix_(full_g1[v1, :] == 1)]
    v2_neighs = numpy.array(range(len(full_g2)))[numpy.ix_(full_g2[v2, :] == 1)]
    cost_in = inside_cost(full_g1, v1_neighs, full_g2, v2_neighs)
    cost_out = out_cost(D1, v1_neighs, D2, v2_neighs)
    return cost_in, cost_out


def out_cost(D1, v1_neighs, D2, v2_neighs):
    if len(v1_neighs) < len(v2_neighs):
        degs1 = D2[numpy.ix_(v2_neighs)] - 1
        degs2 = D1[numpy.ix_(v1_neighs)] - 1
    else:
        degs1 = D1[numpy.ix_(v1_neighs)] - 1
        degs2 = D2[numpy.ix_(v2_neighs)] - 1
    degs2_new = numpy.zeros(len(degs1))
    degs2_new[:len(degs2)] = degs2
    cost_mat = numpy.array([[abs(degs1[i] - degs2_new[j]) for j in range(len(degs2_new))] for i in range(len(degs1))])
    row_ind, col_ind = solve_cost_matrix(cost_mat,False)
    cost = numpy.sum(cost_mat[row_ind, col_ind])
    # print(cost)
    return cost


def inside_cost(full_g1: numpy.ndarray, v1_neighs, full_g2: numpy.ndarray, v2_neighs):
    if len(v1_neighs) < len(v2_neighs):
        inside_degs1 = numpy.sum(full_g2[numpy.ix_(v2_neighs, v2_neighs)], axis=0)
        inside_degs2 = numpy.sum(full_g1[numpy.ix_(v1_neighs, v1_neighs)], axis=0)
    else:
        inside_degs1 = numpy.sum(full_g1[numpy.ix_(v1_neighs, v1_neighs)], axis=0)
        inside_degs2 = numpy.sum(full_g2[numpy.ix_(v2_neighs, v2_neighs)], axis=0)
    inside_degs2_new = numpy.zeros(len(inside_degs1))
    inside_degs2_new[:len(inside_degs2)] = inside_degs2
    cost_mat = numpy.array([[abs(inside_degs1[i] - inside_degs2_new[j]) for j in range(len(inside_degs2_new))] for i in range(len(inside_degs1))])
    row_ind, col_ind = solve_cost_matrix(cost_mat, False)
    return numpy.sum(cost_mat[row_ind, col_ind])


def edit_distance_exact(g1, g2):
    return networkx.graph_edit_distance(networkx.from_numpy_matrix(g1), networkx.from_numpy_matrix(g2))


def compute_cost_matrix(sorted_g1, sorted_g2, k):
    cost_mat = numpy.zeros((k, k))
    vectors1 = [deg_seq_vector(sorted_g1, i) for i in range(len(sorted_g1) - k, len(sorted_g1))]
    vectors2 = [deg_seq_vector(sorted_g2, i) for i in range(len(sorted_g1) - k, len(sorted_g2))]

    for i in range(k):
        for j in range(k):
            cost_mat[k - 1 - i, k - 1 - j] = vector_distance(vectors1[-(i+1)], vectors2[-(j+1)])
    return cost_mat


def vector_distance(v, u):
    if len(v) < len(u):
        t = u
        u = v
        v = t
    sum2 = 0
    for i in range(len(u)):
        sum2 += (v[i]-u[i])**2
    return math.sqrt(sum2)


def degree_seq_distance(full_g1 : numpy.ndarray, v1, full_g2: numpy.ndarray, v2):
    vec1 = deg_seq_vector(full_g1, v1)
    # print(vec1)
    vec2 = deg_seq_vector(full_g2, v2)
    # print(vec2)
    return vector_distance(vec1, vec2)


def deg_seq_vector(g : numpy.ndarray, v, max_level = 3):
    vector = [1]
    marked = numpy.zeros(len(g))
    bfs_q = [(v,0)]
    marked[v] = 1
    last_depth = 0
    flag = False
    while len(bfs_q) > 0:
        if flag:
            break
        node = bfs_q[0]
        bfs_q = bfs_q[1:]
        if node[1] > last_depth:
            last_depth = node[1]
            vector.append(len(bfs_q) + 1)
        for i in range(len(g)):
            if marked[i] == 0 and g[node[0],i] == 1:
                marked[i] = 1
                if node[1] == max_level:
                    flag = True
                    vector.append(len(bfs_q)+1)
                    break
                bfs_q.append((i, node[1] + 1))
    # print(vector)
    return vector


def produce_and_match_with_ed(n, edges, s, k):
    print("producing g1 g2")
    g1, g2 = produce_g1_g2(n, edges, s)
    print("sorting degs: ")
    g1_sorted, g2_sorted, correct_perm = sortDegs(n, g1, g2, numpy.array(range(n)))
    print("---------------------------------")
    print("computing cost matrix...")
    #TODO
    cost_matrix = compute_cost_matrix(g1_sorted, g2_sorted, k)
    print("solving cost matrix: ")
    rowInd, colInd = solve_cost_matrix(cost_matrix, True)
    colInd = numpy.array(colInd + (n - k))
    diff = colInd - correct_perm[-k:]
    max_matchable = int(numpy.sum(correct_perm[-k:] >= n - k))
    return diff, max_matchable


def barPlotDiffs(data_pairs):
    k = len(data_pairs)
    maxes = numpy.array(range(k)) / k
    matches = [data_pairs[i][0] / data_pairs[i][1] for i in range(k)]
    plt.bar(numpy.array(len(matches)), numpy.array(matches))
    plt.bar(numpy.array(len(maxes)), numpy.array(maxes))
    plt.show()
    return matches, maxes


def test_ed_with_high_degs(n, m, p, s, k):
    stats0 = numpy.zeros(m)
    stats1 = numpy.zeros(m)
    from_max_stats = [[0, 0]] * (k + 1)
    # print(from_max_stats)
    for i in range(m):
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("trial # ", i)
        edges = initial_graph(n, p)
        diff1, max = produce_and_match_with_ed(n, edges, s, k)
        match_percentage1 = numpy.sum(diff1 == 0) / k
        stats0[i] = match_percentage1
        stats1[i] = max / k
        print("match1        = ", match_percentage1)
        print("max matchable = ", max / k)
        print(max)
        from_max_stats[max][0] = from_max_stats[max][0] + match_percentage1
        from_max_stats[max][1] = from_max_stats[max][1] + 1
    # matches, maxes = barPlotDiffs(from_max_stats)
    return stats0, stats1


def solve_cost_matrix(cost_mat, greedyHungarianNot):
    if greedyHungarianNot:
        return greedy_match_n3(cost_mat)
    else:
        return scipy.optimize.linear_sum_assignment(cost_mat)


def ds_match_test():
    n = int(input("graph size : "))
    m = int(input("trials : "))
    p = (2 * math.log(n, math.e)) / n
    print("p = ", p)
    l = float(input("multiply p (p = 2*log(n)/n) by: "))
    print(p * l)
    p *= l
    s = float(input("s = "))
    k = int(input("number of high degs up to: "))
    # have_zeros = True if int(input("have isolated nodes?(1,0)")) == 1 else False
    stats = [[],[]]
    for s in frange(1,0.9,-0.01):
        if(s != 1):
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print("s = ", s)
            match_stats1, max_matchable_stats = test_ed_with_high_degs(n, m, p, s, k)
            stats[0].append(numpy.average(match_stats1))
            stats[1].append(numpy.average(max_matchable_stats))
        else:
            stats[0].append(1)
            stats[1].append(1)
    stats[0].reverse()
    stats[1].reverse()
    print(stats[0])
    print(stats[1])
    l1, = plt.plot(numpy.linspace(0.9,1,10), stats[0], 'b' , label='deg seq')
    l2, = plt.plot(numpy.linspace(0.9,1,10), stats[1], 'r', label='max matchable')
    plt.legend(handles=[l1,l2])
    plt.show()

numpy.set_printoptions(precision=3)
ds_match_test()
