import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


# Transform adj matrix to edge list
# Par:
# adj; adj matrix
### Optimize!
def Mtrx_Elist(A):
    j, i = np.nonzero(np.triu(A))  # Find edges
    elist = np.vstack((i, j))
    weights = A[np.triu(A) != 0]  # Find weights

    return elist.transpose(), weights.tolist()


# Legacy code
# def Mtrx_Elist(adj):
#     n = len(adj)
#     elist = []
#     weights = []
#     u_adj = np.triu(adj)
#     for j in range(n):
#         for i in range(n):
#             if u_adj[i][j] > 0:
#                 elist.append([j, i])
#                 weights.append(u_adj[i, j])
#     return np.array(elist), weights


# Transform edge list to adj matrix
# Par:
## E_list; edge list
## weights; edge weights
### Make sparse adj matrix for future?
def Elist_Mtrx(E_list, weights):
    n = np.max(E_list) + 1  # +1 for Python 0-index
    A = np.zeros(shape=(n, n))

    for i in range(np.shape(E_list)[0]):
        n1, n2 = E_list[i, :]
        w = weights[i]
        A[n1, n2], A[n2, n1] = w, w

    return A


# Compute Laplacian, L
# Par:
## A; adj matrix
def Lap(A):
    L = np.diag(np.sum(abs(A), 1)) - A
    return (L)


# Compute signed-edge vertex incidence matrix, B
# Par:
## E_list; edge list
def sVIM(E_list):
    m = np.shape(E_list)[0]  # number of edges
    E_list = E_list.transpose()  # make rows edge list

    data = [1] * m + [-1] * m  # arbitrary tails and heads
    i = list(range(0, m)) + list(range(0, m))  # i-th positions
    j = E_list[0, :].tolist() + E_list[1, :].tolist()  # j-th positions

    B = sparse.csr_matrix((data, (i, j)))  # Using sparse row matrix format for later use

    return B


# Compute weights matrix, W
# Par:
## weights; edge weights
def WDiag(weights):
    m = len(weights)

    weights_sqrt = np.sqrt(weights)  # element-wise sqrt of weights for later use
    W = sparse.dia_matrix((weights_sqrt, [0]), shape=(m, m))  # Use more efficent dia sparse matrix

    return W


# EffR Aproximation
# method from Koutis et al.
# Par:
## E_list; edge list
## weights; list of weights
## epsilon; controls accuracy of approximation, increases computation time
## type; type of calculation for EffR
##
#### 'ext', exact calculation
#### 'ssa', original Spielman-Srivastava algorithm
#### 'kts', Koutis et. al
##### Implement preconditioner M for cg solver? cg(A,b,tol,M=None) - use spilu function or another from scipy.sparse.linalg? https://stackoverflow.com/questions/32865832/preconditioned-conjugate-gradient-and-linearoperator-in-python
##### !Warning! For very small networks, a preconditioner is advised!
def EffR(E_list, weights, epsilon, type, tol=1e-10, precon=False):
    # Find number of edges and number of nodes
    m = np.shape(E_list)[0]
    n = np.max(E_list) + 1

    # Obtain necessary matrcies from edge list and edge weights
    A = Elist_Mtrx(E_list, weights)  # adj matrix - go to sparse?
    L = Lap(A)  # Laplacian (array) - go to sparse? Also needed for solver
    B = sVIM(E_list)  # vertex indicies matrix (crs)
    W = WDiag(weights)  # Diagonal weight matrix (dia)
    scale = np.ceil(np.log2(n)) / epsilon  # set scale/resolution for Johnson-Lindenstrauss projection

    # Find preconditioner for L if precon is True
    if precon:
        M_inverse = sparse.linalg.spilu(L)
        M = sparse.linalg.LinearOperator((n, n), M_inverse.solve)

    # Ignore preconditioner if precon is False
    elif not (precon):
        M = None

    # If preconditioner is passed, set M to precon
    else:
        M = precon

    # Exact effR values
    if type == 'ext':
        effR = np.zeros(shape=(1, m))
        if M == None:  # If no preconditioner
            for i in range(m):
                Br = B[i, :].toarray()
                Z = cg(L, Br.transpose(), tol=tol)[0]
                R_eff = Br @ Z
                effR[:, i] = R_eff[0]
        else:  # If preconditioner
            for i in range(m):
                Br = B[i, :].toarray()
                Z = cg(L, Br.transpose(), tol=tol, M=M)[0]
                R_eff = Br @ Z
                effR[:, i] = R_eff[0]

        effR = effR[0]
        return effR.tolist()

    # Orignal Spielman-Srivastava algorithm
    if type == 'spl':

        # Define Q in type coo sparse matrix
        Q1 = sparse.random(int(scale), m, 1, format='csr') > 0.5
        Q2 = sparse.random(int(scale), m, 1, format='csr') > 0
        Q_not = Q1 - Q2  # need this to pass by invalid 'not' operator
        Q = Q1 + (-1 * Q_not)  # create Q matrix of 1s and -1s
        Q = Q / np.sqrt(scale)

        SYS = Q @ W @ B  # create system for Johnson-Lindenstrauss projection
        Z = np.zeros(shape=(int(scale), n))  # Create Z matrix to solve smaller dim SYS for effR

        if M == None:  # If no preconditioner
            for i in range(int(scale)):
                SYSr = SYS[i, :].toarray()
                Z[i, :] = cg(L, SYSr.transpose(), tol=tol)[0]
        else:  # If preconditioner
            for i in range(int(scale)):
                SYSr = SYS[i, :].toarray()
                Z[i, :] = cg(L, SYSr.transpose(), tol=tol, M=M)[0]

        effR = np.sum(np.square(Z[:, E_list[:, 0]] - Z[:, E_list[:, 1]]),
                      axis=0)  # Calculate distance between poitns for effR
        return effR.tolist()

    # Koutis et al. algorithm
    if type == 'kts':
        effR_res = np.zeros(shape=(1, m))

        if M == None:
            for i in range(int(scale)):
                ons1 = sparse.random(1, m, 1, format='csr') > 0.5
                ons2 = sparse.random(1, m, 1, format='csr') > 0
                ons_not = ons1 - ons2  # need this to pass by invalid 'not' operator
                ons = ons1 + (-1 * ons_not)  # create Q matrix of 1s and -1s
                ons = ons / np.sqrt(scale)

                b = ons @ W @ B
                b = b.toarray()

                Z = sparse.linalg.cg(L, b.transpose(), tol=tol)[0]
                Z = Z.transpose()

                effR_res = effR_res + np.abs(np.square(Z[E_list[:, 0]] - Z[E_list[:, 1]]))

        else:
            for i in range(int(scale)):
                # Create memory saving vectors
                ons1 = sparse.random(1, m, 1, format='csr') > 0.5
                ons2 = sparse.random(1, m, 1, format='csr') > 0
                ons_not = ons1 - ons2  # need this to pass by invalid 'not' operator
                ons = ons1 + (-1 * ons_not)  # create Q matrix of 1s and -1s
                ons = ons / np.sqrt(scale)

                b = ons @ W @ B

                Z = sparse.linalg.cg(L, b.transpose, tol=tol, M=M)[0]
                Z = Z.transpose()

                effR_res = effR_res + np.abs(np.square(Z[E_list[:, 0]] - Z[E_list[:, 1]]))

        effR = effR_res[0]
        return effR.tolist()
