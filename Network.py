import EffRApprox as er
import Spielman_Sparse as spl
import RanGraphGen as rg
import FastSims as fs
import numpy as np
from scipy import sparse
import networkx as nx


class Network:
    def __init__(self, E_list, weights, *args):
        if len(args) != 0:
            for arg in args:
                if arg.size() > 800000:
                    A = nx.adjacency_matrix(arg)
                    if not self._csr_allclose(a=A, b=A.T):
                        A.setdiag(0)
                        A = (A + A.T) / 2
                    self.graph = A

                    self._getIDs(arg)
                    self.data = arg.nodes.data()
                    # self.pos = self._getpos(arg)
                    del arg
                    G = nx.from_scipy_sparse_matrix(A)
                    self.neighbors = self._findneighbors(G)
                    self._getedgelist(A)

                else:
                    A = nx.adjacency_matrix(arg).toarray()
                    if not np.allclose(A, A.T):
                        np.fill_diagonal(A, 0)
                        A = (A + A.T) / 2
                    self.E_list, self.weights = er.Mtrx_Elist(A)
                    self.neighbors = self._findneighbors(A)
                    self.graph = A

        else:
            self.E_list = E_list
            self.weights = weights
            self.IDs = None
            self.data = None
            self.neighbors = self._findneighbors(er.Elist_Mtrx(E_list, weights))

    def _getIDs(self, G):
        nodes = [i for i in G.nodes]
        IDs = {}
        for i in range(len(nodes)):
            IDs[nodes[i]] = i
        self.IDs = IDs

    def _getpos(self, G):
        pos = {}
        nodes = {}
        long = nx.get_node_attributes(G, 'longitude')
        lat = nx.get_node_attributes(G, 'latitude')
        ids = [x for x in self.IDs.keys()]
        for i in range(len(self.IDs)):
            nodes[i] = ids[i]
        for i in range(G.number_of_nodes()):
            pos[nodes[i]] = (long[nodes[i]], lat[nodes[i]])
        return pos

    @staticmethod
    def _csr_allclose(a, b, rtol=1e-5, atol=1e-8):
        c = np.abs(np.abs(a - b) - rtol * np.abs(b))
        return c.max() <= atol

    @staticmethod
    def _findneighbors(G):
        neighbors = {}
        if isinstance(G, nx.classes.graph.Graph):
            for n in range(G.number_of_nodes()):
                test = [x for x in nx.neighbors(G,n)]
                neighbors[n] = [(n, x, G[n][x]['weight']) for x in test]
        elif isinstance(G, np.ndarray):
            for n in range(len(G)):
                incident_row = G[n, :]
                edges = [i for i, e in enumerate(incident_row) if e > 0]
                neighbors[n] = [(n, x, G[n, x]) for x in edges]
        return neighbors

    def _getedgelist(self, A):
        E = sparse.triu(A)
        edges = E.nonzero()
        E_list = np.zeros((len(edges[1]), 2))
        weights = []
        i = 0
        for e1, e2 in zip(edges[0], edges[1]):
            E_list[i, :] = e1, e2
            weights.append(A[e1, e2])
            i += 1
        self.E_list = E_list.astype('int')
        self.weights = weights

    def adj(self):
        return er.Elist_Mtrx(self.E_list, self.weights)

    def edgenum(self):
        return len(self.weights)

    def effR(self, epsilon, method, tol=1e-10, precon=False):
        return er.EffR(self.E_list, self.weights, epsilon, method, tol=tol, precon=precon)

    def spl(self, q, effR, seed=None):
        spl_net = spl.Spl_EffRSparse(n=self.graph.shape[0], E_list=self.E_list, weights=self.weights, q=q, effR=effR, seed=seed)
        E_list, weights = er.Mtrx_Elist(spl_net)
        return Network(E_list, weights)

    def uni(self, q, seed=None):
        uni_net = spl.UniSampleSparse(n=self.graph.shape[0], E_list=self.E_list, weights=self.weights, q=q, seed=None)
        E_list, weights = er.Mtrx_Elist(uni_net)
        return Network(E_list, weights)

    def wts(self, q):
        wts_net = spl.WeightSparse(n=self.graph.shape[0], E_list=self.E_list, weights=self.weights, q=q, seed=None)
        E_list, weights = er.Mtrx_Elist(wts_net)
        return Network(E_list, weights)

    def thr(self, per):
        E_list, weights = fs.Thresh(self.E_list, self.weights, per)
        return Network(E_list, weights)

    def SIR(self, beta, gamma, pzs, t_max, seed=None):
        return fs.SIR_fast2(self.graph, beta, gamma, pzs, t_max, self.neighbors, seed=seed)

    # def AvgSIR(self, res, num, beta, gamma, pzs, t_max):
    #     return fs.AvgSIR(res, num, self, beta, gamma, pzs, t_max)
    #
    # def SizeSIR(self, num, beta, gamma, pzs, t_max):
    #     return fs.SizeSIR(num, self, beta, gamma, pzs, t_max)
    #
    # def Arrivals(self, num, beta, gamma, pzs, t_max):
    #     return fs.Arrivals(num, self, beta, gamma, pzs, t_max)
    #
    # def SI(self, beta, pzs, t_max, seed=None):
    #     return fs.SI_fast(self.adj(), beta, pzs, t_max, seed=seed)

    def sims(self, num, res, beta, gamma, pzs, t_max, seed=None):
        return fs.simulations(num, res, self, beta, gamma, pzs, t_max, seed)

    @classmethod
    def tri(cls):
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        E_list, weights = er.Mtrx_Elist(A)
        return Network(E_list, weights)

    @classmethod
    def complete(cls, n):
        A = rg.ER_gen(n, 1)
        E_list, weights = er.Mtrx_Elist(A)
        return Network(E_list, weights)

    @classmethod
    def MassCom(cls):
        A = np.load('C://Users//henry//PycharmProjects//Summer2021Research//mass_commute_2017.npy')
        np.fill_diagonal(A, 0)
        E_list, weights = er.Mtrx_Elist(A)
        return Network(E_list, weights)

    @classmethod
    def NCCom(cls):
        G = nx.read_graphml('tract_commuter_flows.graphml')
        return Network(None, None, G)

