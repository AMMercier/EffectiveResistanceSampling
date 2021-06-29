import numpy as np
from matplotlib import pyplot as plt
from HeapObject import heappush, heappop
import scipy.stats as st
import Spielman_Sparse as spl
import pandas as pd
import multiprocessing as mp
from functools import partial


# # Find neighbors of a given node
# # Par:
# ## A; adj matrix
# ## node; node to find neighbors
# def find_neighbors(A, node):
#     incident_row = A[node, :]
#     neighbors = [i for i, e in enumerate(incident_row) if e > 0]
#     edges = [(node, x, A[node, x]) for x in neighbors]
#     return edges
#
#
# # Fast SIR continuous-time, gillespie algorithm
# # Par:
# ## network; adj matrix for corresponding matrix
# ## beta; infection rate - corresponds to DiffEQ beta
# ## gamma; recovery rate
# ## pzs; list of patient zeros
# ## t_max; max time to take
# ## seed = None; set seed for reproducibility
# ### Change so can have multiple pz... [DONE]
# ### In the event that no change happens, the event is still added to the events list...
# ### A duplicate row is also added to SIR_t
# ### t goes above t_max do to loop structure
# ### change for weighted as well! (for edge events prb.exponential(1/(beta*weight)))? [done]
# def SIR_fast(network, infec_prob, recov_prob, pzs, t_max, seed=None):
#     rng = np.random.default_rng(seed)
#
#     n = len(network)
#     times, S, I, R = [0], [1] * n, [0] * n, [0] * n
#     for pz in pzs:
#         I[pz], S[pz] = 1, 0
#     S_t, I_t, R_t = [sum(S)], [sum(I)], [sum(R)]
#     t = 0
#     events = []
#
#     Q = []
#     for pz in pzs:
#         neighbors = find_neighbors(network, pz)
#         first_events_edges = [(rng.exponential(1 / (infec_prob * edge[2])), edge[0], edge[1]) for edge in neighbors]
#         first_events_nodes = (rng.exponential(1 / recov_prob), pz)
#         heappush(Q, first_events_nodes)
#         for x in first_events_edges:
#             heappush(Q, x)
#
#     while t < t_max:
#         if len(Q) != 0:
#             S, I, R, Q, t, event, rng = SIR_step(network, S, I, R, infec_prob, recov_prob, Q, t, rng)
#             if sum(I) == 0:
#                 times.append(t_max)
#                 S_t.append(sum(S))
#                 I_t.append(sum(I))
#                 R_t.append(sum(R))
#                 SIR_t = np.array([times, S_t, I_t, R_t]).T
#                 size = n - sum(S)
#                 return SIR_t, size, events
#         else:
#             times.append(t_max)
#             S_t.append(sum(S))
#             I_t.append(sum(I))
#             R_t.append(sum(R))
#             SIR_t = np.array([times, S_t, I_t, R_t]).T
#             size = n - sum(S)
#             return SIR_t, size, events
#         if isinstance(event, tuple):
#             times.append(t)
#             events.append(event)
#             S_t.append(sum(S))
#             I_t.append(sum(I))
#             R_t.append(sum(R))
#     SIR_t = np.array([times, S_t, I_t, R_t]).T
#     size = n - sum(S)
#     return SIR_t, size, events
#
#
# # Move to next event in Fast_SIR
# # Par:
# ## A; adj matrix
# ## S; 1xn dim array to tag nodes susceptible
# ## I; 1xn dim array to tag nodes infected
# ## R; 1xn dim array to tag nodes recovered
# ## beta; infection rate - scaled (see above)
# ## gamma; recovery rate
# ## Q; heap of events
# ### Q is structured (time, edge0, edge1) for edge infection events where edge0 is source
# ### Q is structured (time, node) for node recovery events
# ## t; current time of sim
# ## rng; probability generating function (see above) - not current
# def SIR_step(A, S, I, R, infec_prob, recov_prob, Q, t, rng):
#     rng = rng
#     event = heappop(Q)
#     t = event[0]
#     if len(event) == 3:
#         u, v = event[1], event[2]
#         if I[u] == 1 and S[v] == 1:
#             S[v] = 0
#             I[v] = 1
#             neighbors = find_neighbors(A, v)
#             events_edges = [(t + rng.exponential(1 / (infec_prob * edge[2])), edge[0], edge[1]) for edge in neighbors if
#                             S[edge[1]] == 1]
#             event_node = (t + rng.exponential(1 / recov_prob),
#                           v)  # Can I just generate a big list of n exponential events then pull from them? It is faster.
#             heappush(Q, event_node)
#             for x in events_edges:
#                 heappush(Q, x)
#         else:
#             event = False
#             return S, I, R, Q, t, event, rng
#     else:
#         if I[event[1]] == 1:
#             u = event[1]
#             R[u], S[u], I[u] = 1, 0, 0
#         else:
#             event = False
#     return S, I, R, Q, t, event, rng


# Faster code for neighbors...
# test = [x for x in nx.neighbors(G,350)]
# edges = [(350, x, G[350][x]['weight']) for x in test]


def SIR_fast2(network, beta, gamma, pzs, t_max, neighbor_dict, seed=None):
    rng = np.random.default_rng(seed)

    n = network.shape[0]
    times, S, I, R = [0], [1] * n, [0] * n, [0] * n
    for pz in pzs:
        I[pz], S[pz] = 1, 0
    S_t, I_t, R_t = [sum(S)], [sum(I)], [sum(R)]
    t = 0
    events = []

    Q = []
    for pz in pzs:
        neighbors = neighbor_dict[pz]
        first_events_edges = [(rng.exponential(1 / (beta * edge[2])), edge[0], edge[1]) for edge in neighbors]
        first_events_nodes = (rng.exponential(1 / gamma), pz)
        heappush(Q, first_events_nodes)
        for x in first_events_edges:
            heappush(Q, x)

    while t < t_max:
        if len(Q) != 0:
            S, I, R, Q, t, event, rng = SIR_step2(neighbor_dict, S, I, R, beta, gamma, Q, t, rng)
            if sum(I) == 0:
                times.append(t_max)
                S_t.append(sum(S))
                I_t.append(sum(I))
                R_t.append(sum(R))
                SIR_t = np.array([times, S_t, I_t, R_t]).T
                size = n - sum(S)
                return SIR_t, size, events
        else:
            times.append(t_max)
            S_t.append(sum(S))
            I_t.append(sum(I))
            R_t.append(sum(R))
            SIR_t = np.array([times, S_t, I_t, R_t]).T
            size = n - sum(S)
            return SIR_t, size, events
        if isinstance(event, tuple):
            times.append(t)
            events.append(event)
            S_t.append(sum(S))
            I_t.append(sum(I))
            R_t.append(sum(R))
    SIR_t = np.array([times, S_t, I_t, R_t]).T
    size = n - sum(S)
    return SIR_t, size, events


# Move to next event in Fast_SIR
# Par:
## A; adj matrix
## S; 1xn dim array to tag nodes susceptible
## I; 1xn dim array to tag nodes infected
## R; 1xn dim array to tag nodes recovered
## beta; infection rate - scaled (see above)
## gamma; recovery rate
## Q; heap of events
### Q is structured (time, edge0, edge1) for edge infection events where edge0 is source
### Q is structured (time, node) for node recovery events
## t; current time of sim
## rng; probability generating function (see above) - not current
def SIR_step2(neighbor_dict, S, I, R, beta, gamma, Q, t, rng):
    rng = rng
    event = heappop(Q)
    t = event[0]
    if len(event) == 3:
        u, v = event[1], event[2]
        if I[u] == 1 and S[v] == 1:
            S[v] = 0;
            I[v] = 1
            neighbors = neighbor_dict[v]
            events_edges = [(t + rng.exponential(1 / (beta * edge[2])), edge[0], edge[1]) for edge in neighbors if
                            S[edge[1]] == 1]
            event_node = (t + rng.exponential(1 / gamma), v)
            heappush(Q, event_node)
            for x in events_edges:
                heappush(Q, x)
        else:
            event = False
            return S, I, R, Q, t, event, rng
    else:
        if I[event[1]] == 1:
            u = event[1]
            R[u], S[u], I[u] = 1, 0, 0
        else:
            event = False
    return S, I, R, Q, t, event, rng


# Fast SI continuous-time, gillespie algorithm
# Par:
## network; adj matrix for corresponding matrix
## beta; infection rate - corresponds to DiffEQ beta
## pzs; list of patient zeros
## t_max; max time to take
## seed = None; set seed for reproducibility
# def SI_fast(network, infec_prob, pzs, t_max, seed=None):
#     rng = np.random.default_rng(seed)
#
#     n = len(network)
#     times, S, I = [0], [1] * n, [0] * n
#     for pz in pzs: I[pz] = 1; S[pz] = 0
#     S_t, I_t = [sum(S)], [sum(I)]
#     t = 0
#     events = []
#
#     Q = []
#     for pz in pzs:
#         neighbors = find_neighbors(network, pz)
#         first_events_edges = [(rng.exponential(1 / infec_prob), edge[0], edge[1]) for edge in neighbors]
#         for x in first_events_edges:
#             heappush(Q, x)
#
#     while t < t_max:
#         if len(Q) != 0:
#             S, I, Q, t, event, rng = SI_step(network, S, I, infec_prob, Q, t, rng)
#         else:
#             SIR_t = np.array([times, S_t, I_t]).T
#             return SIR_t, S, I, times, events
#
#         times.append(t)
#         events.append(event)
#         S_t.append(sum(S))
#         I_t.append(sum(I))
#     SIR_t = np.array([times, S_t, I_t]).T
#     return SIR_t, S, I, times, events
#
#
# # Move to next event in Fast_SIR
# # Par:
# ## A; adj matrix
# ## S; 1xn dim array to tag nodes susceptible
# ## I; 1xn dim array to tag nodes infected
# ## R; 1xn dim array to tag nodes recovered
# ## beta; infection rate - scaled (see above)
# ## gamma; recovery rate
# ## Q; heap of events
# ### Q is structured (time, edge0, edge1) for edge infection events where edge0 is source
# ### Q is structured (time, node) for node recovery events
# ## t; current time of sim
# ## rng; probability generating function (see above) - not current
# def SI_step(A, S, I, infec_prob, Q, t, rng):
#     rng = rng
#     event = heappop(Q)
#     t = event[0]
#     u = event[1]
#     v = event[2]
#     if I[u] == 1 and S[v] == 1:
#         S[v] = 0
#         I[v] = 1
#         neighbors = find_neighbors(A, v)
#         events_edges = [(t + rng.exponential(1 / infec_prob), edge[0], edge[1]) for edge in neighbors if
#                         S[edge[1]] == 1]
#         for x in events_edges:
#             heappush(Q, x)
#     else:
#         return S, I, Q, t, event, rng
#     return S, I, Q, t, event, rng


# Find the confidence intervals and mean of a set of data
# Par:
## data; the set of data to find the confidence interval from
## conf; the confidence interval to set
# Output:
## lmh; the lower bound, mean, and upper bound given by the ci
# def confidence_interval(data, conf=0.95):
#     n1, n2 = np.shape(data)
#     lmh = np.zeros((n1, 3))
#     for i in range(n1):
#         avg, serr = np.mean(data[i, :]), st.sem(data[i, :])
#         int = serr * st.t.ppf((1 + conf) / 2., n2 - 1)
#         lmh[i, :] = [avg - int, avg, avg + int]
#     return lmh


# Find the average of a set of SIR sims
# Par:
## res; number of intervals to take average
## num; number of simulations to take average across
## network; network to use <network object>
## beta; infection rate
## gamma; recovery rate
## pzs; patient zeros
## t_max; max amount of time to run simulation
### Note: add a seed list  for reproducible results
### Odd R dropoff caused by termination of sim when heap ends?
# def AvgSIR(res, num, network, beta, gamma, pzs, t_max):
#     res, diff = np.linspace(0, t_max, res, retstep=True)
#     S, I, R = np.zeros((len(res), num)), np.zeros((len(res), num)), np.zeros((len(res), num))
#     time = res.reshape((len(res), 1))
#
#     for i in range(num):
#         sim = network.SIR(beta, gamma, pzs, t_max)[0]
#         tick = 0
#         for r in res:
#             events = sim[sim[:, 0] <= r]
#             events = events[events[:, 0] > r - diff]
#             if len(events) == 0:
#                 events = [r, 0, 0, 0]
#             else:
#                 event = events[-1, :]
#                 S[tick, i], I[tick, i], R[tick, i] = event[1], event[2], event[3]
#                 tick += 1
#     S = confidence_interval(S)
#     I = confidence_interval(I)
#     R = confidence_interval(R)
#     return time, S, I, R


# def SizeSIR(num, network, beta, gamma, pzs, t_max):
#     sizes = []
#     n = len(network.adj())
#
#     for i in range(num):
#         sim = network.SIR(beta, gamma, pzs, t_max)[0]
#         e_size = n - sim[-1, 1]
#         sizes.append(e_size)
#
#     return sizes


# def KSTest(num, res, pts, network, beta, gamma, pzs, t_max):
#     qs = [10**x for x in range(pts + 1) if x > 0]
#     effR = network.effR(0.1, 'spl')
#     S = np.zeros((len(qs),3))
#     I = np.zeros((len(qs), 3))
#     R = np.zeros((len(qs), 3))
#     results = np.zeros((len(qs),4))
#     pedge = []
#
#     i = 0
#     for q in qs:
#         spl_net = network.spl(q, effR)
#
#         time, S_org, I_org, R_org = network.AvgSIR(res, num, beta, gamma, pzs, t_max)
#         time, S_spl, I_spl, R_spl = spl_net.AvgSIR(res, num, beta, gamma, pzs, t_max)
#
#         S[i,:] = st.kstest(S_spl[:,0].T, S_org[:,0].T)[0], st.kstest(S_spl[:,1].T, S_org[:,1].T)[0], st.kstest(S_spl[:,2].T, S_org[:,0].T)[0]
#         I[i,:] = st.kstest(I_spl[:,0].T, I_org[:,0].T)[0], st.kstest(I_spl[:,1].T, I_org[:,1].T)[0], st.kstest(I_spl[:,2].T, I_org[:,0].T)[0]
#         R[i,:] = st.kstest(R_spl[:,0].T, R_org[:,0].T)[0], st.kstest(R_spl[:,1].T, R_org[:,1].T)[0], st.kstest(R_spl[:,2].T, R_org[:,0].T)[0]
#
#         i += 1
#         pedge.append(spl_net.edgenum()/network.edgenum() * 100)
#     return qs, S, I, R, pedge

# def gettimes(sim):
#     events = sim[5]
#     times = []
#     for e in events:
#         if len(e) == 3:
#             times.append(e[0])
#
#     return times


# def ArivalTime(num, network, beta, gamma, pzs, t_max):
#     total_times = []
# 
#     for i in range(num):
#         sim = network.SIR(beta, gamma, pzs, t_max)
#         times = gettimes(sim)
#         total_times.extend(times)
# 
#     return total_times
# 
# def KSArival(num, pts, network, beta, gamma, pzs, t_max):
#     qs = [2 ** x for x in range(pts + 1) if x > 0]
#     effR = network.effR(0.1, 'spl')
#     uni, wts, spl = np.zeros((1,pts)), np.zeros((1,pts)), np.zeros((1,pts))
#     org_tm = network.Arival(num, beta, gamma, pzs, t_max)
# 
#     i = 0
#     for q in qs:
#         uni_net = network.uni(q)
#         wts_net = network.wts(q)
#         spl_net = network.spl(q, effR)
# 
#         uni_tm = uni_net.Arival(num, beta, gamma, pzs, t_max)
#         wts_tm = uni_net.Arival(num, beta, gamma, pzs, t_max)
#         spl_tm = uni_net.Arival(num, beta, gamma, pzs, t_max)
# 
#         if len(uni_tm) == 0:
#             uni[0,i] = 1
#         else:
#             uni[0,i] = st.kstest(uni_tm, org_tm)[0]
#         if len(wts_tm) == 0:
#             wts[0,i] = 1
#         else:
#             wts[0,i] = st.kstest(wts_tm, org_tm)[0]
#         if len(spl_tm) == 0:
#             spl[0,i] = 1
#         else:
#             spl[0,i] = st.kstest(spl_tm, org_tm)[0]
# 
#         i += 1
#     return qs, uni, wts, spl

# def Arrivals(num, network, beta, gamma, pzs, t_max):
#     n = len(network.adj())
#     arivals = {}
#     for i in range(n):
#         arivals[i] = []
#
#     for n in range(num):
#         events = network.SIR(beta, gamma, pzs, t_max)[5]
#         events = [e for e in events if len(e) == 3]
#         print(n)
#         for e in events:
#             t, infec_node = e[0], e[2]
#             arivals[infec_node].append(t)
#
#     return arivals


def IST(network, beta, pzs, t_max):
    n = len(network.adj())
    events = network.SI(beta, pzs, t_max)[5]  # check [5]
    events = [e for e in events if len(e) == 3]
    H = np.zeros((n, n))
    for e in events:
        n1, n2 = e[1], e[2]
        H[n1, n2], H[n2, n1] = 1, 1
    return H


def EEI(num, network, beta, pzs, t_max):
    n = len(network.adj())
    H = np.zeros((n, n))
    for i in range(num):
        H += IST(network, beta, pzs, t_max)
    return H / num


def IT(network, beta, gamma, pzs, t_max, seed):
    n = len(network.adj())
    events = network.SIR(beta, gamma, pzs, t_max, seed)[2]
    events = [e for e in events if len(e) == 3]
    H = np.zeros((n, n))
    for e in events:
        n1, n2 = e[1], e[2]
        H[n1, n2], H[n2, n1] = 1, 1
    return H


def EEI_SIR(num, network, gamma, beta, pzs, t_max, seed):
    rng = np.random.default_rng(seed)
    sds = rng.integers(0, 800000, num)
    n = len(network.adj())
    H = np.zeros((n, n))
    for i in range(num):
        H += IT(network, beta, gamma, pzs, t_max, sds[i])
        print(i)
    return H / num


def Thresh(E_list, weights, per):
    m = int(np.ceil(per * len(weights)))
    n_weights = [0] * len(weights)
    weights = list(enumerate(weights))
    weights = sorted(weights, key=lambda tup: tup[1], reverse=True)
    weights = weights[0:m]
    for i in weights:
        n_weights[i[0]] = i[1]
    return E_list, n_weights


def getSIRAvg(SIR, res, diff, n, S, I, R):
    i = 0
    for r in res:
        t_c = SIR[SIR[:, 0] <= r]
        t_c = t_c[t_c[:, 0] > r - diff]
        if len(t_c) == 0:
            t_c = [r, 0, 0, 0]
        else:
            t = t_c[-1, :]
        S[i, n], I[i, n], R[i, n] = t[1], t[2], t[3]
        i += 1
    return S, I, R


def confidence_interval(data, conf=0.95, column='S'):
    n1, n2 = np.shape(data)
    lmh = np.zeros((n1, 3))
    for i in range(n1):
        avg, s_err = np.mean(data[i, :]), st.sem(data[i, :])
        interval = s_err * st.t.ppf((1 + conf) / 2., n2 - 1)
        lmh[i, :] = [avg - interval, avg, avg + interval]
    return pd.DataFrame.from_dict({'{}_low'.format(column): lmh[:, 0],
                                   '{}_avg'.format(column): lmh[:, 1],
                                   '{}_high'.format(column): lmh[:, 2]})


def getArrivals(events, ari):
    events = [e for e in events if len(e) == 3]
    for e in events:
        t, inf_node = e[0], e[2]
        ari[inf_node].append(t)
    return ari


def simulations(num, res, network, beta, gamma, pzs, t_max, seed):
    # Set list of seeds
    rng = np.random.default_rng(seed)
    sd = rng.integers(0, 80000, num)
    # Set arrivals dictionary
    n = len(network.adj())
    ari = {}
    for i in range(n):
        ari[i] = []
    # Set resolutions for averages
    res, diff = np.linspace(0, t_max, res, retstep=True)
    S, I, R = np.zeros((len(res), num)), np.zeros((len(res), num)), np.zeros((len(res), num))
    time = pd.DataFrame.from_dict({'t': res})
    # Set sizes
    sizes = np.zeros((1, num))
    # Run all sims
    for n in range(num):
        # Compute sim
        sim = network.SIR(beta, gamma, pzs, t_max, sd[n])
        # Find arrival times for each node
        events = sim[2]
        ari = getArrivals(events, ari)
        # Compute averages
        tSIR = sim[0]
        S, I, R = getSIRAvg(tSIR, res, diff, n, S, I, R)
        # Assign sizes
        sizes[0, n] = sim[1]
        print(n)
    S = confidence_interval(S, column='S')
    I = confidence_interval(I, column='I')
    R = confidence_interval(R, column='R')
    avg_tSIR = pd.concat([time, S, I, R], axis=1)
    return ari, avg_tSIR, sizes


def SIR(seed, network, beta, gamma, pzs, t_max, neighbor_dict):
    return SIR_fast2(network, beta, gamma, pzs, t_max, neighbor_dict, seed)


def par_SIR(num, network, beta, gamma, pzs, t_max, seed, workers):
    partial_func = partial(SIR, network=network.graph, beta=beta, gamma=gamma, pzs=pzs, t_max=t_max,
                           neighbor_dict=network.neighbors)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10 ** 16, num).tolist()
    with mp.Pool(processes=workers) as pool:
        multiple_results = pool.map(partial_func, seeds, chunksize=num//workers)

    #

    return multiple_results


if __name__ == '__main__':
    import Network as nt

    A = nt.Network.NCCom()
    test = par_SIR(100, A, 5 / 2183, 1, [350], 10, 100, 20)
    print(len(list(test)))

#     pass
#     #     E_list, weights = er.Mtrx_Elist(rg.ER_gen(500, 1))
#     #     network = nt.Network(E_list, weights)
#     #     del E_list, weights
#     #
#     # def AvgSIR(res, num, network, beta, gamma, pzs, t_max):
#     #     res, diff = np.linspace(0,t_max,res+1, retstep=True)
#     #     avgSIR = np.zeros((len(res),4))
#     #     avgSIR[:,0] = res.T
#     #
#     #     for n in range(num):
#     #         sim =  network.SIR(beta, gamma, pzs, t_max)[0]
#     #         tick = 0
#     #         for r in res:
#     #             events = sim[sim[:,0] <= r]
#     #             events = events[events[:,0] > r-diff]
#     #             if len(events)==0:
#     #                 event = [r, 0, 0, 0]
#     #             else:
#     #                 event = events[-1,:]
#     #                 avgSIR[tick,1:4] = avgSIR[tick,1:4] + event[1:4]
#     #                 print(r,event)
#     #                 tick += 1
#     #
#     #     return avgSIR
#
#     # A = rg.ER_gen(1000, 1)
#     # E_list, weights = er.Mtrx_Elist(A)
#     # network = nt.Network(E_list, weights)
#     #
#     # sim = SIR_fast(network.Adj(), 3, 1, 1, 100)
#     #
#     # SIR_t = sim[0]
#     #
#     # SIR_t = sim[0]
#     # SIR_t2 = sim2[0]
#     #     plt.plot(SIR[:, 0], SIR[:, 1] / 500, color="black")
#     #     plt.plot(SIR[:, 0], SIR[:, 2] / 500, color="black")
#     #     plt.plot(SIR[:, 0], SIR[:, 3] / 500, color="black")
#
#     plt.plot(SIR_t[:, 0], SIR_t[:, 1] / 500, label="Susceptible", color="yellow")
#     plt.plot(SIR_t[:, 0], SIR_t[:, 2] / 500, label="Infected", color="orange")
#     plt.plot(SIR_t[:, 0], SIR_t[:, 3] / 500, label="Recovered", color="purple")
#
#     # Rescale
#     S = S / 1470
#     I = I / 1470
#     R = R / 1470
#
#     fig, ax = plt.subplots()
#
#     ax.plot(time, S[:, 1], label="Susceptible", color="green")
#     ax.fill_between(time[:, 0], S[:, 0], S[:, 2], color="green", alpha=0.1)  # ugliness because of fill_between issues
#
#     ax.plot(time, I[:, 1], label="Infected", color="red")
#     ax.fill_between(time[:, 0], I[:, 0], I[:, 2], color="red", alpha=0.1)
#
#     ax.plot(time, R[:, 1], label="Recovered", color="blue")
#     ax.fill_between(time[:, 0], R[:, 0], R[:, 2], color="blue", alpha=0.1)
#
#     ax.legend()  # Orignal
#
#     weights = np.ones_like(test) / len(test)
#     plt.hist(test2, bins=250, weights=weights, color='blue', label="Original", alpha=0.5)
#     plt.hist(test, bins=250, weights=weights, color='red', label="Sparse", alpha=0.5)
#     plt.xlabel("Epidemic Size", size=12)
#     plt.ylabel("Fraction of Sims", size=12)
#     plt.legend(loc="upper left")
#     plt.title("Massachusetts Epidemic Size\n Original vs. 20% Sparse Network")
#
#     S = sim_org[1] / 1470
#     I = sim_org[2] / 1470
#     R = sim_org[3] / 1470
#     time = sim_org[0]
#
#     fig, ax = plt.subplots()
#
#     ax.plot(time[0:24], S[0:24, 1], label="Original", color="black", linestyle='--')
#     ax.fill_between(time[0:24, 0], S[0:24, 0], S[0:24, 2], color="black",
#                     alpha=0.1)  # ugliness because of fill_between issues
#
#     ax.plot(time[0:24], I[0:24, 1], color="black", linestyle='--')
#     ax.fill_between(time[0:24, 0], I[0:24, 0], I[0:24, 2], color="black", alpha=0.1)
#
#     ax.plot(time[0:24], R[0:24, 1], color="black", linestyle='--')
#     ax.fill_between(time[0:24, 0], R[0:24, 0], R[0:24, 2], color="black", alpha=0.1)
#
#     S = sim_spl[1] / 1470
#     I = sim_spl[2] / 1470
#     R = sim_spl[3] / 1470
#     time = sim_spl[0]
#
#     ax.plot(time[0:24], S[0:24, 1], label="Susceptible", color="green")
#     ax.fill_between(time[0:24, 0], S[0:24, 0], S[0:24, 2], color="green",
#                     alpha=0.1)  # ugliness because of fill_between issues
#
#     ax.plot(time[0:24], I[0:24, 1], label="Infected", color="red")
#     ax.fill_between(time[0:24, 0], I[0:24, 0], I[0:24, 2], color="red", alpha=0.1)
#
#     ax.plot(time[0:24], R[0:24, 1], label="Recovered", color="blue")
#     ax.fill_between(time[0:24, 0], R[0:24, 0], R[0:24, 2], color="blue", alpha=0.1)
#
#     plt.legend()
#     plt.title("Massachusetts Network SIR Averrage\n Original vs. 20% Sparse Network")
#     plt.xlabel("Time", size=12)
#     plt.ylabel("Fraction of Nodes", size=12)
#     plt.figtext(0.5, 0.01, r"1000 simulations, $\beta=5/1470$, $\gamma=1$", ha="center")
#
#     plt.hist(ari_org[237], bins=100, weights=[1 / 1000] * len(ari_org[237]), color='blue', label="Original", alpha=0.5)
#     plt.hist(ari_spl[237], bins=100, weights=[1 / 1000] * len(ari_spl[237]), color='red', label="Sparse", alpha=0.5)
#     plt.xlabel("Time of Infection", size=12)
#     plt.ylabel("Fraction of Sims", size=12)
#     plt.legend(loc="upper left")
#     plt.title("NC Raleigh to Charlotte Time to Infection\n Original vs. 10% Sparse Network")
#     x = "Wasserstein Distance: {:.3f}".format(st.wasserstein_distance(ari_org[237], ari_spl[237]))
#     plt.figtext(0.6, 0.75, x, ha="center", size=12)
#     plt.figtext(0.5, 0.01, r"1000 simulations, $\beta=5/1470$, $\gamma=1$, 37163050100$\rightarrow$37119000100",
#                 ha="center", size=9)
#
#     S = sim_org[1] / 1470
#     I = sim_org[2] / 1470
#     R = sim_org[3] / 1470
#     time = sim_org[0]
#
#     fig, ax = plt.subplots()
#     plt.style.use('seaborn-whitegrid')
#
#     ax.plot(time[0:21], S[0:21, 1], label="Original", color="black", linestyle='--')
#     ax.fill_between(time[0:21, 0], S[0:21, 0], S[0:21, 2], color="black",
#                     alpha=0.1)  # ugliness because of fill_between issues
#
#     ax.plot(time[0:21], I[0:21, 1], color="black", linestyle='--')
#     ax.fill_between(time[0:21, 0], I[0:21, 0], I[0:21, 2], color="black", alpha=0.1)
#
#     ax.plot(time[0:21], R[0:21, 1], color="black", linestyle='--')
#     ax.fill_between(time[0:21, 0], R[0:21, 0], R[0:21, 2], color="black", alpha=0.1)
#
#     S = sim_spl[1] / 1470
#     I = sim_spl[2] / 1470
#     R = sim_spl[3] / 1470
#     time = sim_spl[0]
#
#     ax.plot(time[0:21], S[0:21, 1], label="Susceptible", color="green")
#     ax.fill_between(time[0:21, 0], S[0:21, 0], S[0:21, 2], color="green",
#                     alpha=0.1)  # ugliness because of fill_between issues
#
#     ax.plot(time[0:21], I[0:21, 1], label="Infected", color="red")
#     ax.fill_between(time[0:21, 0], I[0:21, 0], I[0:21, 2], color="red", alpha=0.1)
#
#     ax.plot(time[0:21], R[0:21, 1], label="Recovered", color="blue")
#     ax.fill_between(time[0:21, 0], R[0:21, 0], R[0:21, 2], color="blue", alpha=0.1)
#
#     plt.legend(fancybox=True, framealpha=1, shadow=False, borderpad=0.75)
#     plt.title("Massachusetts Network SIR Averrage\n Original vs. 5.5% Sparse Network")
#     plt.xlabel("Time", size=12)
#     plt.ylabel("Fraction of Nodes", size=12)
#     plt.figtext(0.5, 0.01, r"1000 simulations, $\beta=5/1470$, $\gamma=1$", ha="center")
#
#     qs = KSTest[0]
#     KS_uni = KSTest[1]
#     KS_wts = KSTest[2]
#     KS_spl = KSTest[3]
#
#     fig, ax = plt.subplots()
#     plt.style.use('seaborn-whitegrid')
#
#     ax.plot(qs, KS_uni, 'o', color="darkgreen", label="Uniform")
#     ax.plot(qs, KS_wts, '^', color="mediumvioletred", label="Weights")
#     ax.plot(qs, KS_spl, 'v', color="tomato", label="EffR")
#
#     plt.xscale('log')
#
#     ax.xlabel("Number of Samples")
#     ax.ylabel("D")
#     ax.ylim([0, 1])
#
#     # plt.plot(SIR_t2[:,0],SIR_t2[:,1]/500,label="Susceptible2",color="green")
#     # plt.plot(SIR_t2[:,0],SIR_t2[:,2]/500,label="Infected2",color="red")
#     # plt.plot(SIR_t2[:,0],SIR_t2[:,3]/500,label="Recovered2",color="blue")
#     #
#     # plt.legend()
#     # plt.show()
#
#     # #Make SIR plots
#     # sim = SIR_fast(A, 2, 1, 0, 1000)
#     # SIR_t = sim[0]
#     # plt.plot(SIR_t[:,0],SIR_t[:,1]/500,label="Susceptible",color="yellow")
#     # plt.plot(SIR_t[:,0],SIR_t[:,2]/500,label="Infected",color="orange")
#     # plt.plot(SIR_t[:,0],SIR_t[:,3]/500,label="Recovered",color="purple")
#     # plt.legend()
#     #
#     # #Make SIR recovered histograms
#     # A = rg.ER_gen(1000,1)
#     # R_num = []
#     # for i in range(1,1000):
#     #     #ran_int = np.ceil(np.random.uniform(0,1)*1000)
#     #     sim = SIR_fast(A, 0.01, 0.2, 0, 1000)
#     #     R_num.append(sum(sim[3]))
#     # print(R_num)
#     #
#     # n, bins, patches = plt.hist(x=R_num, bins='auto', color='#0504aa',
#     #                              alpha=0.7, rwidth=0.85)
#     # plt.grid(axis='y', alpha=0.75)
#     # plt.xlabel('Recovered')
#     # plt.ylabel('Frequency')
#     # plt.title('Test 4 of Recovered Fraction')
#     # plt.text(23, 45, r'$\beta=0.01, \gamma=0.2$, Trials=1000')
#     #
#
#     S = sim_org[1] / 2183
#     I = sim_org[2] / 2183
#     R = sim_org[3] / 2183
#     time = sim_org[0]
#
#     fig, axs = plt.subplots(3)
#
#     axs[0].plot(time[0:21], S[0:21, 1], label="Original", color="black", linestyle='--')
#     axs[0].fill_between(time[0:21, 0], S[0:21, 0], S[0:21, 2], color="black",
#                         alpha=0.1)  # ugliness because of fill_between issues
#
#     axs[0].plot(time[0:21], I[0:21, 1], color="black", linestyle='--')
#     axs[0].fill_between(time[0:21, 0], I[0:21, 0], I[0:21, 2], color="black", alpha=0.1)
#
#     axs[0].plot(time[0:21], R[0:21, 1], color="black", linestyle='--')
#     axs[0].fill_between(time[0:21, 0], R[0:21, 0], R[0:21, 2], color="black", alpha=0.1)
#
#     axs[1].plot(time[0:21], S[0:21, 1], label="Original", color="black", linestyle='--')
#     axs[1].fill_between(time[0:21, 0], S[0:21, 0], S[0:21, 2], color="black",
#                         alpha=0.1)  # ugliness because of fill_between issues
#
#     axs[1].plot(time[0:21], I[0:21, 1], color="black", linestyle='--')
#     axs[1].fill_between(time[0:21, 0], I[0:21, 0], I[0:21, 2], color="black", alpha=0.1)
#
#     axs[1].plot(time[0:21], R[0:21, 1], color="black", linestyle='--')
#     axs[1].fill_between(time[0:21, 0], R[0:21, 0], R[0:21, 2], color="black", alpha=0.1)
#
#     axs[2].plot(time[0:21], S[0:21, 1], label="Original", color="black", linestyle='--')
#     axs[2].fill_between(time[0:21, 0], S[0:21, 0], S[0:21, 2], color="black",
#                         alpha=0.1)  # ugliness because of fill_between issues
#
#     axs[2].plot(time[0:21], I[0:21, 1], color="black", linestyle='--')
#     axs[2].fill_between(time[0:21, 0], I[0:21, 0], I[0:21, 2], color="black", alpha=0.1)
#
#     axs[2].plot(time[0:21], R[0:21, 1], color="black", linestyle='--')
#     axs[2].fill_between(time[0:21, 0], R[0:21, 0], R[0:21, 2], color="black", alpha=0.1)
#
#     S = sim_spl[1] / 2183
#     I = sim_spl[2] / 2183
#     R = sim_spl[3] / 2183
#
#     axs[0].plot(time[0:21], S[0:21, 1], label="Susceptible", color="green")
#     axs[0].fill_between(time[0:21, 0], S[0:21, 0], S[0:21, 2], color="green",
#                         alpha=0.1)  # ugliness because of fill_between issues
#
#     axs[0].plot(time[0:21], I[0:21, 1], label="Infected", color="red")
#     axs[0].fill_between(time[0:21, 0], I[0:21, 0], I[0:21, 2], color="red", alpha=0.1)
#
#     axs[0].plot(time[0:21], R[0:21, 1], label="Recovered", color="blue")
#     axs[0].fill_between(time[0:21, 0], R[0:21, 0], R[0:21, 2], color="blue", alpha=0.1)
#
#     S = sim_uni[1] / 2183
#     I = sim_uni[2] / 2183
#     R = sim_uni[3] / 2183
#
#     axs[1].plot(time[0:21], S[0:21, 1], label="Susceptible", color="green")
#     axs[1].fill_between(time[0:21, 0], S[0:21, 0], S[0:21, 2], color="green",
#                         alpha=0.1)  # ugliness because of fill_between issues
#
#     axs[1].plot(time[0:21], I[0:21, 1], label="Infected", color="red")
#     axs[1].fill_between(time[0:21, 0], I[0:21, 0], I[0:21, 2], color="red", alpha=0.1)
#
#     axs[1].plot(time[0:21], R[0:21, 1], label="Recovered", color="blue")
#     axs[1].fill_between(time[0:21, 0], R[0:21, 0], R[0:21, 2], color="blue", alpha=0.1)
#
#     S = sim_wts[1] / 2183
#     I = sim_wts[2] / 2183
#     R = sim_wts[3] / 2183
#
#     axs[2].plot(time[0:21], S[0:21, 1], label="Susceptible", color="green")
#     axs[2].fill_between(time[0:21, 0], S[0:21, 0], S[0:21, 2], color="green",
#                         alpha=0.1)  # ugliness because of fill_between issues
#
#     axs[2].plot(time[0:21], I[0:21, 1], label="Infected", color="red")
#     axs[2].fill_between(time[0:21, 0], I[0:21, 0], I[0:21, 2], color="red", alpha=0.1)
#
#     axs[2].plot(time[0:21], R[0:21, 1], label="Recovered", color="blue")
#     axs[2].fill_between(time[0:21, 0], R[0:21, 0], R[0:21, 2], color="blue", alpha=0.1)
#
#     axs[0].set_title("EffR", size=14, loc="left")
#     axs[1].set_title("Uniform", size=14, loc="left")
#     axs[2].set_title("Weights", size=14, loc="left")
#
#     axs[0].legend(loc="upper center", ncol=4, prop={'size': 8}, bbox_to_anchor=(0.5, 1.15))
#
#     plt.figtext(0.5, 0.01, r"1000 simulations, $\beta=5/1470$, $\gamma=1$", ha="center")
#     plt.style.use('seaborn-whitegrid')
#     fig.text(0.5, 0.04, 'Time', ha='center')
#     fig.text(0.04, 0.5, 'Fraction of Nodes', va='center', rotation='vertical')
#     plt.suptitle("NC Network, 10% Sparsifiers")
#
#     fig, axs = plt.subplots(1, 3)
#     axs[0].hist(spl.normprobs(ari_org[237]), bins=100, weights=[1 / 1000] * len(ari_org[237]), color='blue',
#                 label="Original",
#                 alpha=0.5)
#     axs[1].hist(spl.normprobs(ari_org[237]), bins=100, weights=[1 / 1000] * len(ari_org[237]), color='blue',
#                 label="Original",
#                 alpha=0.5)
#     axs[2].hist(spl.normprobs(ari_org[237]), bins=100, weights=[1 / 1000] * len(ari_org[237]), color='blue',
#                 label="Original",
#                 alpha=0.5)
#     axs[0].hist(spl.normprobs(ari_spl[237]), bins=100, weights=[1 / 1000] * len(ari_spl[237]), color='red',
#                 label="effR", alpha=0.5)
#     axs[1].hist(spl.normprobs(ari_wts[237]), bins=100, weights=[1 / 1000] * len(ari_wts[237]), color='yellow',
#                 label="Weights",
#                 alpha=0.5)
#     axs[2].hist(spl.normprobs(ari_uni[237]), bins=100, weights=[1 / 1000] * len(ari_uni[237]), color='orange',
#                 label="Uniform",
#                 alpha=0.5)
#     fig.text(0.5, 0.04, 'Norm Arival Time', ha='center')
#     fig.text(0.04, 0.5, 'Fraction of Sims', va='center', rotation='vertical')
#     axs[0].legend(loc="upper left")
#     axs[1].legend(loc="upper left")
#     axs[2].legend(loc="upper left")
#     plt.suptitle("NC Raleigh to Charlotte Time to Infection\n Original vs. 10% Sparse Network")
#     x = "Wasserstein Dist: {:.3f}".format(
#         st.wasserstein_distance(spl.normprobs(ari_org[237]), spl.normprobs(ari_spl[237])))
#     y = "Wasserstein Dist: {:.3f}".format(
#         st.wasserstein_distance(spl.normprobs(ari_org[237]), spl.normprobs(ari_wts[237])))
#     z = "Wasserstein Dist: {:.3f}".format(
#         st.wasserstein_distance(spl.normprobs(ari_org[237]), spl.normprobs(ari_uni[237])))
#     plt.figtext(0.3, 0.75, x, ha="center", size=8)
#     plt.figtext(0.55, 0.75, y, ha="center", size=8)
#     plt.figtext(0.8, 0.75, z, ha="center", size=8)
#     plt.figtext(0.5, 0.01, r"1000 simulations, $\beta=5/2183$, $\gamma=1$, 37163050100$\rightarrow$37119000100",
#                 ha="center", size=9)
