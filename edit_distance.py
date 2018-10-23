import math

import networkx
import numpy
import random
import time
import scipy.optimize
import matplotlib.pyplot as plt
import stringdist
import zss


def greedy_match_n3(n, costMatrix):
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


def edit_distance(g1, g2, max_iter):
    # return edit_distance_exact(g1, g2)
    # return edit_distance_iterative(g1, g2, max_iter)
    # return ed_assignment_approx(g1, g2)
    # return ed_spectral_approx(g1, g2)
    return my_ed_approx(g1, g2)


def bfs_ed(full_g1 : numpy.ndarray, D1, v1, full_g2: numpy.ndarray, D2, v2):
    v1_neighs = numpy.array(range(len(full_g1)))[numpy.ix_(full_g1[v1, :] == 1)]
    v2_neighs = numpy.array(range(len(full_g2)))[numpy.ix_(full_g2[v2, :] == 1)]
    cost_in = inside_cost(full_g1, v1_neighs, full_g2, v2_neighs)
    cost_out = out_cost(D1, v1_neighs, D2, v2_neighs)
    return cost_in + cost_out + abs(D1[v1] - D2[v2])


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
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)
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
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)
    return numpy.sum(cost_mat[row_ind, col_ind])


def my_ed_approx(g1, g2):
    graph1 = networkx.from_numpy_matrix(g1)
    graph2 = networkx.from_numpy_matrix(g2)
    c_e1 = []
    c_e2 = []
    for c in networkx.connected_components(graph1):
        c_e1.append((len(c), len(networkx.edges(graph1, c))))
    for c in networkx.connected_components(graph2):
        c_e2.append((len(c), len(networkx.edges(graph2, c))))
    return my_component_distance(c_e1, c_e2)


def my_component_distance(c_e1, c_e2):
    c_e1, c_e2 = iso_size_and_sort(c_e1, c_e2)
    dist = 0
    for i in range(len(c_e1)):
        dist += abs(c_e1[i][0] - c_e2[i][0]) + abs(c_e1[i][1] - c_e2[i][1])
    return dist


def iso_size_and_sort(c_e1, c_e2):
    if len(c_e1) < len(c_e2):
        x = c_e2.copy()
        c_e2 = c_e1
        c_e1 = x
    for i in range(len(c_e1) - len(c_e2)):
        c_e2.append((0, 0))
    c_e1 = sorted(c_e1, key=lambda x: (x[0], x[1]))
    c_e2 = sorted(c_e2, key=lambda x: (x[0], x[1]))
    return c_e1, c_e2


def edit_distance_exact(g1, g2):
    return networkx.graph_edit_distance(networkx.from_numpy_matrix(g1), networkx.from_numpy_matrix(g2))


def edit_distance_iterative(g1, g2, max_iter):
    # print("-- in edit distance\n")
    graph_x_1 = networkx.from_numpy_matrix(g1)
    graph_x_2 = networkx.from_numpy_matrix(g2)
    gen = networkx.optimize_graph_edit_distance(graph_x_1, graph_x_2)
    i = 0
    min_ed = 0
    last_val = 0
    t1 = 0
    t2 = 0
    for val in gen:
        if i >= max_iter:
            break
        # print("new val: ", val)
        t1 = time.time()
        if i > 0:
            if abs(t2 - t1) >= 0.02:
                break
        t2 = t1
        min_ed = val
        i += 1
    return min_ed


def ed_assignment_approx(g1, g2):
    n1 = len(g1)
    n2 = len(g2)
    # print("size g1 g2: ", n1," ", n2)
    if n1 == 0:
        return n2 + (numpy.sum(g2) / 2)
    elif n2 == 0:
        return n1 + (numpy.sum(g1) / 2)
    else:
        cols = numpy.zeros(n1 + n2)
        cols[:n1] = range(n1)
        cols[n1:] -= 1
        rows = numpy.zeros(n1 + n2)
        rows[:n2] = range(n2)
        rows[n2:] -= 1
        cost_matrix = numpy.array([[node_map_cost_approx(i, j, g1, g2) for j in range(n2)] for i in range(n1)])
        rowInd, colInd = scipy.optimize.linear_sum_assignment(cost_matrix)
        cost_sum = numpy.sum(cost_matrix[numpy.ix_(rowInd, colInd)])
    return cost_sum


def node_map_cost_approx(u, v, g1, g2):
    if (u == -1) & (v == -1):
        return 0
    elif u == -1:
        return 1 + (numpy.sum(g2[:, v]) / 2)
    elif v == -1:
        return 1 + (numpy.sum(g1[:, u]) / 2)
    else:
        return 1 + abs(numpy.sum(g1[:, u]) - numpy.sum(g2[:, v])) / 2


def ed_spectral_approx(g1, g2):
    D1 = numpy.diag(numpy.sum(g1, axis=1))
    D2 = numpy.diag(numpy.sum(g2, axis=1))
    # print(g1)
    # print(g2)
    laplacian_1 = D1 - g1
    laplacian_2 = D2 - g2
    w1, v1 = numpy.linalg.eig(laplacian_1)
    w2, v2 = numpy.linalg.eig(laplacian_2)
    w1 = numpy.array(w1)
    w2 = numpy.array(w2)
    w1 = numpy.array(sorted(w1, reverse=True))
    w2 = numpy.array(sorted(w2, reverse=True))
    sum1 = numpy.sum(w1)
    sum2 = numpy.sum(w2)
    s1 = 0
    for i in range(len(w1)):
        s1 += w1[i]
        if s1 >= 0.9 * sum1:
            s1 = i
            break
    s2 = 0
    for j in range(len(w2)):
        s2 += w2[j]
        if s2 >= 0.9 * sum2:
            s2 = j
            break
    k = min(s1, s2)
    # print(k)
    diff = numpy.sum((w1[:k] - w2[:k]) * (w1[:k] - w2[:k]))
    return diff


def compute_cost_matrix(sorted_g1, sorted_g2, k, max_iter, bfs_approach, have_zeros=False):
    cost_mat = numpy.zeros((k, k))
    for i in range(k):
        for j in range(k):
            # print("i, j = ",i,",",j)
            if not bfs_approach:
                sub_g1, full_size_1, final_size_1 = sub_graph(sorted_g1, -(i + 1), have_zeros)
                sub_g2, full_size_2, final_size_2 = sub_graph(sorted_g2, -(j + 1), have_zeros)
                cost = edit_distance(sub_g1, sub_g2, max_iter) + (
                    (abs(full_size_1 - full_size_2) - abs(final_size_1 - final_size_2)) if (
                                abs(full_size_1 - full_size_2) > abs(final_size_1 - final_size_2)) else 0)
                cost_mat[k - 1 - i, k - 1 - j] = cost
            else:
                cost_mat[k - 1 - i, k - 1 - j] = bfs_ed(sorted_g1, numpy.sum(sorted_g1, axis=0), -(i+1), sorted_g2, numpy.sum(sorted_g2, axis=0), -(j+1))
    return cost_mat


def produce_and_match_with_ed(n, edges, s, k, max_iter, bfs_approach, have_zeros=False):
    g1, g2 = produce_g1_g2(n, edges, s)
    # print("sorting degs: ")
    g1_sorted, g2_sorted, correct_perm = sortDegs(n, g1, g2, numpy.array(range(n)))
    print("computing cost matrix...")
    cost_matrix = compute_cost_matrix(g1_sorted, g2_sorted, k, max_iter, bfs_approach, have_zeros)
    # print("solving cost matrix: ")
    rowInd, colInd = scipy.optimize.linear_sum_assignment(cost_matrix)
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


def test_ed_with_high_degs(n, m, p, s, k, max_iter, bfs_approach, have_zeros=False):
    stats1 = numpy.zeros(m)
    stats2 = numpy.zeros(m)
    from_max_stats = [[0, 0]] * (k + 1)
    # print(from_max_stats)
    for i in range(m):
        print("trial # ", i)
        edges = initial_graph(n, p)
        diff, max = produce_and_match_with_ed(n, edges, s, k, max_iter, bfs_approach, have_zeros)
        match_percentage = numpy.sum(diff == 0) / k
        stats2[i] = numpy.sum(diff == 0) / max
        stats1[i] = match_percentage
        print("match          = ", match_percentage)
        print("max matchable = ", max / k)
        print(max)
        from_max_stats[max][0] = from_max_stats[max][0] + match_percentage
        from_max_stats[max][1] = from_max_stats[max][1] + 1
    matches, maxes = barPlotDiffs(from_max_stats)
    return stats1, stats2


def ed_match_test():
    n = int(input("graph size : "))
    m = int(input("trials : "))
    MAX_ITER = int(input("maximum ED iterations: "))
    p = (2 * math.log(n, math.e)) / n
    print("p = ", p)
    l = float(input("multiply p (    p = (2 * math.log(n, math.e)) / n) by: "))
    print(p * l)
    p *= l
    s = float(input("s = "))
    k = int(input("number of high degs up to: "))
    bfs_approach = bool(int(input("go bfs approach?(0 \ 1)")))
    # have_zeros = True if int(input("have isolated nodes?(1,0)")) == 1 else False
    stats1, stats2 = test_ed_with_high_degs(n, m, p, s, k, MAX_ITER, bfs_approach)
    hist, bin = numpy.histogram(stats1)
    print("hist and bin\n", hist, "\n", bin)
    print("average matches = ", numpy.average(stats1))


numpy.set_printoptions(precision=3)
ed_match_test()
