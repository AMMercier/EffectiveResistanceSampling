import matplotlib.pyplot as plt
import Network as nt
import numpy as np
import pickle
import geopandas as gp
import pandas as pd
import matplotlib.colors as cl
from scipy.stats import wasserstein_distance
from scipy import sparse
import networkx as nx
from Spielman_Sparse import normprobs


def IDs_Nodes(IDs):
    ids = [x for x in IDs.keys()]
    nodes = {}
    for i in range(len(ids)):
        nodes[i] = ids[i]
    return nodes


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = cl.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plotPZ(sf, pzs, IDs, axs):
    for pz in pzs:
        geoID = IDs[pz]
        pz_df = sf.loc[sf['GEOID'] == geoID]
        if (pz_df.STATEFP == '15').all():
            pz_df.plot(ax=axs[0], color='red')
        else:
            pz_df.plot(ax=axs[1], color='red')


def Map_Plot_Arr(sf, pzs, color='Spectral', size=(15, 20), I_cmap=(0, 1)):
    nonan = sf.dropna()

    min = np.min(
        [np.min(sf['Wts_Arr']), np.min(sf['Uni_Arr']), np.min(sf['Spl_Arr']), np.min(sf['Thr_Arr'])]
    )
    max = np.max(
        [np.max(sf['Wts_Arr']), np.max(sf['Uni_Arr']), np.max(sf['Spl_Arr']), np.max(sf['Thr_Arr'])]
    )
    cmap = plt.get_cmap(color)
    color = truncate_colormap(cmap, minval=I_cmap[0], maxval=I_cmap[1], n=500)
    sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=min, vmax=max))
    sm._A = []

    legend_args = {'label': 'Avg. Arrival Time', 'orientation': 'horizontal'}
    missing_args = {'color': 'white', 'edgecolor': 'red', 'hatch': "///", "label": 'Missing Values'}

    fig, axs = plt.subplots(2, 2)
    cbax = fig.add_axes([0.75, 0.3, 0.03, 0.39])
    cbax.set_title('Arrival')
    fig.colorbar(sm, cax=cbax, format="%.2f")

    # sf.plot(column='Org_{}'.format(eval), ax=ax[0], cmap=color, legend=False, vmin=min, vmax=max,
    #         legend_kwds=legend_args, missing_kwds=missing_args)


def SIRnorm(SIR1, n):
    # Normalize
    SIR1['t'] = SIR1['t'] / np.max(SIR1['t'])

    SIR = ['S_low', 'S_avg', 'S_high', 'I_low', 'I_avg', 'I_high', 'R_low', 'R_avg', 'R_high']
    SIR1[SIR] = SIR1[SIR] / n

    return SIR1


def SIRplot(SIR1, SIR2, title, ax):
    # Plot Original
    ax.plot(SIR1['t'], SIR1['I_avg'], label="Original", color="black", linestyle='--')
    ax.plot(SIR1['t'], SIR1['R_avg'], color="black", linestyle='--')

    ax.fill_between(SIR1['t'], SIR1['I_low'], SIR1['I_high'], color="black", alpha=0.3)
    ax.fill_between(SIR1['t'], SIR1['R_low'], SIR1['R_high'], color="black", alpha=0.3)

    # Plot Sparse
    ax.plot(SIR2['t'], SIR2['I_avg'], label="Infected", color="red")
    ax.plot(SIR2['t'], SIR2['R_avg'], label="Recovered", color="blue")

    ax.fill_between(SIR2['t'], SIR2['I_low'], SIR2['I_high'], color="red", alpha=0.3)
    ax.fill_between(SIR2['t'], SIR2['R_low'], SIR2['R_high'], color="blue", alpha=0.3)

    # Set plot
    # ax.title.set_text(title)


def WDMeans(Org_Arrivals, Sps_Arrivals, simnum):
    arrivals = []
    for i in range(len(Org_Arrivals)):
        arrival_org = [x / simnum for x in Org_Arrivals[i]]
        arrival_spl = [x / simnum for x in Sps_Arrivals[i]]
        if arrival_org == [] or arrival_spl == []:
            if arrival_org == [] and arrival_spl != []:
                arrivals.append(1)
            elif arrival_org != [] and arrival_spl == []:
                arrivals.append(1)
            elif arrival_org == [] and arrival_spl == []:
                arrivals.append(0)
        else:
            arrivals.append(wasserstein_distance(arrival_org, arrival_spl))

    return np.mean(arrivals)


def WDArrivals(simnum, edgenum, qs):
    num = qs.shape[0]

    with open('USNet_Results_Org/US_Arrivals_Org.pkl', 'rb') as f:
        Org_Arrivals = pickle.load(f)

    Spl = np.zeros((2, num))
    Uni = np.zeros((2, num))
    Wts = np.zeros((2, num))
    Thr = np.zeros((2, num))

    for i in range(num):
        # Spl Sparsifiers
        with open('USNet_Results_Spl/US_Arrivals_Spl_{}.pkl'.format(qs[i]), 'rb') as f:
            Spl_Arrivals = pickle.load(f)

        Spl[0, i] = WDMeans(Org_Arrivals, Spl_Arrivals, simnum)
        G = nx.from_scipy_sparse_matrix(sparse.load_npz('USNet_Results_Spl/Spl_Nets/SplNet_{}.npz'.format(qs[i])))
        SplNet = nt.Network(None, None, G)
        Spl[1, i] = SplNet.edgenum() / edgenum

        # Uni Sparsifiers
        with open('USNet_Results_Uni/US_Arrivals_Uni_{}.pkl'.format(qs[i]), 'rb') as f:
            Uni_Arrivals = pickle.load(f)

        Uni[0, i] = WDMeans(Org_Arrivals, Uni_Arrivals, simnum)
        G = nx.from_scipy_sparse_matrix(sparse.load_npz('USNet_Results_Uni/Uni_Nets/UniNet_{}.npz'.format(qs[i])))
        UniNet = nt.Network(None, None, G)
        Uni[1, i] = UniNet.edgenum() / edgenum

        # Wts Sparsifiers
        with open('USNet_Results_Wts/US_Arrivals_Wts_{}.pkl'.format(qs[i]), 'rb') as f:
            Wts_Arrivals = pickle.load(f)

        Wts[0, i] = WDMeans(Org_Arrivals, Wts_Arrivals, simnum)
        G = nx.from_scipy_sparse_matrix(sparse.load_npz('USNet_Results_Wts/Wts_Nets/WtsNet_{}.npz'.format(qs[i])))
        WtsNet = nt.Network(None, None, G)
        Wts[1, i] = WtsNet.edgenum() / edgenum

        # Thr Sparsifiers
        with open('USNet_Results_Thr/US_Arrivals_Thr_{}.pkl'.format(qs[i]), 'rb') as f:
            Thr_Arrivals = pickle.load(f)

        Thr[0, i] = WDMeans(Org_Arrivals, Thr_Arrivals, simnum)
        G = nx.from_scipy_sparse_matrix(sparse.load_npz('USNet_Results_Thr/Thr_Nets/ThrNet_{}.npz'.format(qs[i])))
        ThrNet = nt.Network(None, None, G)
        Thr[1, i] = ThrNet.edgenum() / edgenum

        print(i)

    return Spl, Uni, Wts, Thr


def InfecProb(Arrivals, simnum, nodenum):
    infec = np.zeros((1, nodenum))
    for i in range(nodenum):
        infec[0, i] = len(Arrivals[i]) / simnum
    return infec


def InfecProb_dict(Arrivals, simnum):
    nodenum = len(Arrivals)
    infecprobs = {}
    for i in range(nodenum):
        infecprobs[i] = len(Arrivals[i]) / simnum
    return infecprobs


def R_squared(x, y):
    corr_matrix = np.corrcoef(x, y)
    corr_xy = corr_matrix[0, 1]
    return corr_xy ** 2


def HeapSizes(qs):
    Org_Heap = int(np.mean(np.load('USNet_Results_Org/USNet_heaps.npy')))
    results = np.zeros((4, qs.shape[0]))

    for i in range(qs.shape[0]):
        Spl_Heap = int(np.mean(np.load('USNet_Results_Spl/US_Heap_Spl_{}.npy'.format(qs[i]))))
        Uni_Heap = int(np.mean(np.load('USNet_Results_Uni/US_Heap_Uni_{}.npy'.format(qs[i]))))
        Wts_Heap = int(np.mean(np.load('USNet_Results_Wts/US_Heap_Wts_{}.npy'.format(qs[i]))))
        Thr_Heap = int(np.mean(np.load('USNet_Results_Thr/US_Heap_Thr_{}.npy'.format(qs[i]))))

        results[0, i] = Spl_Heap / Org_Heap
        results[1, i] = Uni_Heap / Org_Heap
        results[2, i] = Wts_Heap / Org_Heap
        results[3, i] = Thr_Heap / Org_Heap

    return results


def CPUTime(qs):
    Org_Time = int(np.mean(np.load('USNet_Results_Org/USNet_times.npy')))
    results = np.zeros((4, qs.shape[0]))

    for i in range(qs.shape[0]):
        Spl_Time = np.mean(np.load('USNet_Results_Spl/US_Times_Spl_{}.npy'.format(qs[i])))
        Uni_Time = np.mean(np.load('USNet_Results_Uni/US_Times_Uni_{}.npy'.format(qs[i])))
        Wts_Time = np.mean(np.load('USNet_Results_Wts/US_Times_Wts_{}.npy'.format(qs[i])))
        Thr_Time = np.mean(np.load('USNet_Results_Thr/US_Times_Thr_{}.npy'.format(qs[i])))

        results[0, i] = Spl_Time / Org_Time
        results[1, i] = Uni_Time / Org_Time
        results[2, i] = Wts_Time / Org_Time
        results[3, i] = Thr_Time / Org_Time

    return results


def AvgArrival(Arrivals):
    avg_arrivals = {}
    for i in range(len(Arrivals)):
        if len(Arrivals[i]) == 0:
            avg_arrivals[i] = []
        else:
            avg_arrivals[i] = np.mean(Arrivals[i])
    return avg_arrivals


def ConCat_Results(sf, dic, IDs, title):
    tag = '{}'.format(title)
    GEOID = []
    results = []
    for i in range(len(dic)):
        GEOID.append(IDs[i])
        if not dic[i]:
            results.append(None)
        else:
            results.append(np.mean(dic[i]))
    df = pd.DataFrame.from_dict({'GEOID': GEOID, tag: results})
    sf = sf.merge(df, on='GEOID')
    return sf


def ConCat_Map():
    states = [('AL', '01'), ('AZ', '04'), ('AR', '05'), ('CA', '06'), ('CO', '08'), ('CT', '09'), ('DE', '10'),
              ('DC', '11'),
              ('FL', '12'), ('GA', '13'), ('HI', '15'), ('ID', '16'), ('IL', '17'), ('IN', '18'), ('IA', '19'),
              ('KS', '20'), ('KY', '21'),
              ('LA', '22'), ('ME', '23'), ('MD', '24'), ('MA', '25'), ('MI', '26'), ('MN', '27'), ('MS', '28'),
              ('MO', '29'), ('MT', '30'),
              ('NE', '31'), ('NV', '32'), ('NH', '33'), ('NJ', '34'), ('NM', '35'), ('NY', '36'), ('NC', '37'),
              ('ND', '38'), ('OH', '39'),
              ('OK', '40'), ('OR', '41'), ('PA', '42'), ('RI', '44'), ('SC', '45'), ('SD', '46'), ('TN', '47'),
              ('TX', '48'), ('UT', '49'),
              ('VT', '50'), ('VA', '51'), ('WA', '53'), ('WV', '54'), ('WI', '55'), ('WY', '56')]

    path = 'US Shape File 2016/{}/cb_2016_{}_tract_500k.shp'.format(states[0][0], states[0][1])
    map = gp.read_file(path)
    for i in range(1, 50):
        # if states[i][1] == '26':
        #     path = 'US Shape File 2016/MI_2015/cb_2015_26_tract_500k.shp'
        # else:
        path = 'US Shape File 2016/{}/cb_2016_{}_tract_500k.shp'.format(states[i][0], states[i][1])
        join = gp.read_file(path)
        map = map.append(join)
    return map


def GenericUSMap(sf, eval, save, pzs, bounds, color='Spectral'):
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
    cmap = plt.get_cmap(color)
    # HI
    sf[sf.STATEFP == '15'].plot(color='lightgrey', ax=axs[0])
    sf[sf.STATEFP == '15'].plot(column=eval, ax=axs[0], cmap=cmap)
    # Contiguous US
    sf[sf.STATEFP != '15'].plot(color='lightgrey', ax=axs[1])
    sf[sf.STATEFP != '15'].plot(column=eval, ax=axs[1], cmap=cmap)
    # Patient zeros
    plotPZ(sf, pzs, USNet_IDs, axs)
    # Crop
    axs[0].set_xlim(-160.5, -154.75)
    axs[0].set_ylim(18.75, 22.5)
    axs[0].axis('off')
    axs[1].axis('off')
    plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, wspace=0.05, hspace=0.1)
    # Colorbar
    min = bounds[0]
    max = bounds[1]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min, vmax=max))
    sm._A = []
    fig.colorbar(sm, format="%.1f", orientation='horizontal', shrink=0.7, pad=0.01)
    # Save
    plt.savefig('USNetMaps/{}.png'.format(save), dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    qs = np.linspace(0, 1, 10, endpoint=False) + 0.1
    num = qs.shape[0]

    # Node number: 72721
    # Edge number: 26319308

    ######################
    ######################
    ##                  ##
    ##   Code Chunk 1   ##
    ##                  ##
    ######################
    ######################

    # SIR curves

    # nodenum = 72721
    # edgenum = 26319308
    #
    # OrgNet_SIR = SIRnorm(pd.read_csv('USNet_Results_Org/US_SIR_Org.csv'), nodenum)
    #
    # for i in range(10):
    #     SplNet_SIR = SIRnorm(pd.read_csv('USNet_Results_Spl/US_SIR_Spl_{}.csv'.format(qs[i])), nodenum)
    #     UniNet_SIR = SIRnorm(pd.read_csv('USNet_Results_Uni/US_SIR_Uni_{}.csv'.format(qs[i])), nodenum)
    #     WtsNet_SIR = SIRnorm(pd.read_csv('USNet_Results_Wts/US_SIR_Wts_{}.csv'.format(qs[i])), nodenum)
    #     ThrNet_SIR = SIRnorm(pd.read_csv('USNet_Results_Thr/US_SIR_Thr_{}.csv'.format(qs[i])), nodenum)
    #
    #     plt.style.use('seaborn-whitegrid')
    #     fig, axs = plt.subplots(4, figsize=(10,8))
    #     plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.35)
    #
    #     SIRplot(OrgNet_SIR, SplNet_SIR, 'EffR', axs[0])
    #     SIRplot(OrgNet_SIR, UniNet_SIR, 'Uniform', axs[1])
    #     SIRplot(OrgNet_SIR, WtsNet_SIR, 'Weights', axs[2])
    #     SIRplot(OrgNet_SIR, ThrNet_SIR, 'Thresholding', axs[3])
    #
    #     axs[0].set_title("EffR", size=14, loc="left")
    #     axs[1].set_title("Uniform", size=14, loc="left")
    #     axs[2].set_title("Weights", size=14, loc="left")
    #     axs[3].set_title("Thresholding", size=14, loc="left")
    #
    #     axs[0].legend(loc="upper center", ncol=3, prop={'size': 12}, bbox_to_anchor=(0.5, 1.25))
    #
    #     # plt.figtext(0.5, 0.01, r"1000 simulations, $\beta=10/2183$, $\gamma=1$", ha="center")
    #     plt.style.use('seaborn-whitegrid')
    #     fig.text(0.5, 0.02, 'Fraction of Time', size=14, ha='center')
    #     fig.text(0.02, 0.5, 'Fraction of Nodes', size=14, va='center', rotation='vertical')
    #     plt.suptitle("US Network\n q={:.1f} Sparsifiers".format(qs[i]), size=16)
    #
    #     plt.savefig('USNetFigs/SIRFigs/USNet_SIR_avg_{:.1f}.png'.format(qs[i]), dpi=300)

    ######################
    ######################
    ##                  ##
    ##   Code Chunk 2   ##
    ##                  ##
    ######################
    ######################

    # Wasserstein distance arrival times

    # nodenum = 72721
    # edgenum = 26319308
    # simnum = 1000
    # Spl, Uni, Wts, Thr = WDArrivals(simnum, edgenum, qs)  # confidence intervals?
    # print([Spl, Uni, Wts, Thr])
    # np.save('SplAvg_Arrivals.npy', Spl)
    # np.save('UniAvg_Arrivals.npy', Uni)
    # np.save('WtsAvg_Arrivals.npy', Wts)
    # np.save('ThrAvg_Arrivals.npy', Thr)

    # Spl = np.load('SplAvg_Arrivals.npy')
    # Uni = np.load('UniAvg_Arrivals.npy')
    # Wts = np.load('WtsAvg_Arrivals.npy')
    # Thr = np.load('ThrAvg_Arrivals.npy')
    # plt.style.use('seaborn-whitegrid')
    # plt.plot(Spl[1, :], Spl[0, :], '-o', label="EffR", color="red")
    # plt.plot(Uni[1, :], Uni[0, :], '-^', label="Uni", color="blue")
    # plt.plot(Wts[1, :], Wts[0, :], '-v', label="Wts", color="green")
    # plt.plot(Thr[1, :], Thr[0, :], '-d', label="Thr", color="purple")
    # plt.title("US Network\n Wasserstein Metric of Arrival Times")
    # plt.xlabel("Fraction of Edges")
    # plt.ylabel("Wasserstein Distance")
    # plt.legend(loc="upper right", prop={'size': 12})
    # plt.savefig('USNetFigs/Arrival Figs/USNet_Arrivals1.png', dpi=300)

    ######################
    ######################
    ##                  ##
    ##   Code Chunk 3   ##
    ##                  ##
    ######################
    ######################

    # Node infec probs

    nodenum = 72721
    edgenum = 26319308
    simnum = 1000

    # with open('USNet_Results_Org/US_Arrivals_Org.pkl', 'rb') as f:
    #     Org_Arrivals = pickle.load(f)
    # Org_Infec = InfecProb(Org_Arrivals, simnum, nodenum)
    # np.save('InfecProbs/Org_Infec.npy', Org_Infec)
    #
    # for i in range(num):
    #     # Spl Sparsifiers
    #     with open('USNet_Results_Spl/US_Arrivals_Spl_{}.pkl'.format(qs[i]), 'rb') as f:
    #         Spl_Arrivals = pickle.load(f)
    #     Spl_Infec = InfecProb(Spl_Arrivals, simnum, nodenum)
    #     np.save('InfecProbs/Spl_Infec_{:2f}.npy'.format(qs[i]), Spl_Infec)
    #
    #     # Uni Sparsifiers
    #     with open('USNet_Results_Uni/US_Arrivals_Uni_{}.pkl'.format(qs[i]), 'rb') as f:
    #         Uni_Arrivals = pickle.load(f)
    #     Uni_Infec = InfecProb(Uni_Arrivals, simnum, nodenum)
    #     np.save('InfecProbs/Uni_Infec_{:2f}.npy'.format(qs[i]), Uni_Infec)
    #
    #     # Wts Sparsifiers
    #     with open('USNet_Results_Wts/US_Arrivals_Wts_{}.pkl'.format(qs[i]), 'rb') as f:
    #         Wts_Arrivals = pickle.load(f)
    #     Wts_Infec = InfecProb(Wts_Arrivals, simnum, nodenum)
    #     np.save('InfecProbs/Wts_Infec_{:2f}.npy'.format(qs[i]), Wts_Infec)
    #
    #     # Thr Sparsifiers
    #     with open('USNet_Results_Thr/US_Arrivals_Thr_{}.pkl'.format(qs[i]), 'rb') as f:
    #         Thr_Arrivals = pickle.load(f)
    #     Thr_Infec = InfecProb(Thr_Arrivals, simnum, nodenum)
    #     np.save('InfecProbs/Thr_Infec_{:2f}.npy'.format(qs[i]), Thr_Infec)
    #
    #     print(i)

    # Org_Infec = np.load('InfecProbs/Org_Infec.npy')
    #
    # plt.style.use('seaborn-whitegrid')
    # fig, axs = plt.subplots(2,2)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.35)
    #
    #
    # # Lowest Spl Sparsifier
    # Spl_Infec = np.load('InfecProbs/Spl_Infec_0.100000.npy')
    # R2 = R_squared(Org_Infec, Spl_Infec)
    # axs[0,0].plot(Org_Infec, Spl_Infec, 'o', color='black', markersize=2)
    # axs[0,0].text(0.2, 0.80, r'$R^2=$' + '{:.2f}'.format(R2), size=14, ha='center')
    #
    # # Lowest Uni Sparsifier
    # Uni_Infec = np.load('InfecProbs/Uni_Infec_0.100000.npy')
    # R2 = R_squared(Org_Infec, Uni_Infec)
    # axs[0,1].plot(Org_Infec, Uni_Infec, 'o', color='black', markersize=2)
    # axs[0,1].text(0.2, 0.80, r'$R^2=$' + '{:.2f}'.format(R2), size=14, ha='center')
    #
    # # Lowest Wts Sparsifier
    # Wts_Infec = np.load('InfecProbs/Wts_Infec_0.100000.npy')
    # R2 = R_squared(Org_Infec, Wts_Infec)
    # axs[1,0].plot(Org_Infec, Wts_Infec, 'o', color='black', markersize=2)
    # axs[1,0].text(0.2, 0.80, r'$R^2=$' + '{:.2f}'.format(R2), size=14, ha='center')
    #
    # # Lowest Thr Sparsifier
    # Thr_Infec = np.load('InfecProbs/Thr_Infec_0.100000.npy')
    # R2 = R_squared(Org_Infec, Thr_Infec)
    # axs[1,1].plot(Org_Infec, Thr_Infec, 'o', color='black', markersize=2)
    # axs[1,1].text(0.2, 0.80, r'$R^2=$' + '{:.2f}'.format(R2), size=14, ha='center')
    #
    # # plt.suptitle("US Network, q=0.1\n Infection Probability QQ Plot", size=16)
    # axs[0,0].set_title("EffR", size=14, loc="left")
    # axs[0,1].set_title("Uniform", size=14, loc="left")
    # axs[1,0].set_title("Weights", size=14, loc="left")
    # axs[1,1].set_title("Thresholding", size=14, loc="left")
    # fig.text(0.5, 0.02, 'Original Node Infection Probability', size=12, ha='center')
    # fig.text(0.02, 0.5, 'Sparsifier Node Infection Probability', size=12, va='center', rotation='vertical')
    #
    # plt.savefig('USNetFigs/InfecProbFigs/USNet_InfecProb_0.1.png', dpi=300)

    ######################
    ######################
    ##                  ##
    ##   Code Chunk 4   ##
    ##                  ##
    ######################
    ######################

    # # Heap Size
    # HeapComp = HeapSizes(qs)
    # plt.plot(qs, HeapComp[0, :],'-o', label='EffR', color='red')
    # plt.plot(qs, HeapComp[1, :],'-^', label="Uni", color="blue")
    # plt.plot(qs, HeapComp[2, :],'-v', label="Wts", color="green")
    # plt.plot(qs, HeapComp[3, :],'-d', label="Thr", color="purple")
    # plt.legend(loc="upper center", ncol=4, prop={'size': 12}, bbox_to_anchor=(0.5, 1.01))
    # plt.xlabel('q')
    # plt.ylabel('Fraction of Maximum Heap Size')
    # plt.title("US Network\n Fraction of Original Maximum Heap Size")
    # plt.savefig('USNetFigs/HeapFigs/USNet_HeapFig.png', dpi=300)

    ######################
    ######################
    ##                  ##
    ##   Code Chunk 5   ##
    ##                  ##
    ######################
    ######################

    # CPU Time (min)
    # TimeComp = CPUTime(qs)
    # plt.plot(qs, TimeComp[0, :], '-o', label='EffR', color='red')
    # plt.plot(qs, TimeComp[1, :], '-^', label="Uni", color="blue")
    # plt.plot(qs, TimeComp[2, :], '-v', label="Wts", color="green")
    # plt.plot(qs, TimeComp[3, :], '-d', label="Thr", color="purple")
    # plt.legend(loc="upper center", ncol=4, prop={'size': 12}, bbox_to_anchor=(0.5, 1.01))
    # plt.xlabel('q')
    # plt.ylabel('Fraction of CPU Time')
    # plt.title("US Network\n Fraction of Original CPU Time")
    # plt.savefig('USNetFigs/CPUTimeFigs/USNet_CPUFig.png', dpi=300)

    ######################
    ######################
    ##                  ##
    ##   Code Chunk 6   ##
    ##                  ##
    ######################
    ######################

    # Find and save node IDs
    # USNet = nt.Network.USNet()
    # USNet_IDs = USNet.IDs
    # USNet_IDs = IDs_Nodes(USNet_IDs)
    # with open('USNet_Results_Org/US_NodeIDs.pkl', 'wb') as f:
    #     pickle.dump(USNet_IDs, f)

    with open('USNet_Results_Org/US_NodeIDs.pkl', 'rb') as f:
        USNet_IDs = pickle.load(f)

    # # Import US census tract shapefile
    # USMap = ConCat_Map()

    # Load Patient Zero locations
    #  = np.load('USNet_Pzs.npy')
    # fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
    # USMap[USMap.STATEFP != '15'].plot(ax=axs[1], color='grey')
    # USMap[USMap.STATEFP == '15'].plot(ax=axs[0], color='grey')
    # plotPZ(USMap, pzs, USNet_IDs, axs)
    # axs[0].set_xlim(-160.5, -154.75)
    # axs[0].set_ylim(18.75, 22.5)
    # axs[0].axis('off')
    # axs[1].axis('off')
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.01, hspace=0.35)
    # plt.savefig('USNetMaps/PatientZeros/USNet_1%PZ.png', dpi=600, bbox_inches='tight', pad_inches=0)

    # # Load Arrivals
    # with open('USNet_Results_Org/US_Arrivals_Org.pkl', 'rb') as f:
    #     USNet_Arrivals = pickle.load(f)
    #
    # AvgArr = AvgArrival(USNet_Arrivals)
    # print(AvgArr)
    # USMap = ConCat_Results(USMap, AvgArr, USNet_IDs, 'Org_Arr')
    # print(USMap)
    # missing_args = {'color': 'grey', 'edgecolor': 'white', "label": 'Missing Values'}
    # cmap = plt.get_cmap('Spectral')
    # fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
    # USMap[USMap.STATEFP != '15'].plot(column='Org_Arr', ax=axs[1], cmap=cmap, missing_kwds=missing_args)
    # USMap[USMap.STATEFP == '15'].plot(column='Org_Arr', ax=axs[0], cmap=cmap, missing_kwds=missing_args)
    # plotPZ(USMap, pzs, USNet_IDs, axs)
    # axs[0].set_xlim(-160.5, -154.75)
    # axs[0].set_ylim(18.75, 22.5)
    # axs[0].axis('off')
    # axs[1].axis('off')
    # plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, wspace=0.05, hspace=0.1)
    # # cbax = fig.add_axes([0.5, 0.1, 0.75, 0.25])
    # min = np.min(USMap['Org_Arr'])
    # max = np.max(USMap['Org_Arr'])
    # sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=min, vmax=max))
    # sm._A = []
    # fig.colorbar(sm, format="%.1f", orientation='horizontal', shrink=0.7, pad=0.1)
    # # plt.show()
    # plt.savefig('USNetMaps/Arrivals/USNet_1%PZ_Arrivals.png', dpi=600, bbox_inches='tight', pad_inches=0)
    # pzs = [42840]
    # for i in range(1,8):
    #     USMap = ConCat_Map()
    #     with open('USNet_Results_Org/LAX Tests/US_Arrivals_OrgLAX{}.pkl'.format(i), 'rb') as f:
    #         USNet_Arrivals = pickle.load(f)
    #
    #     AvgArr = AvgArrival(USNet_Arrivals)
    #     print(AvgArr)
    #     USMap = ConCat_Results(USMap, AvgArr, USNet_IDs, 'Org_Arr')
    #     print(USMap)
    #     missing_args = {'color': 'grey', 'edgecolor': 'white', "label": 'Missing Values'}
    #     cmap = plt.get_cmap('Spectral')
    #     fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
    #     USMap[USMap.STATEFP == '15'].plot(color='lightgrey', ax=axs[0])
    #     USMap[USMap.STATEFP != '15'].plot(color='lightgrey', ax=axs[1])
    #     # USMap = USMap.dropna()
    #     USMap[USMap.STATEFP == '15'].plot(column='Org_Arr', ax=axs[0], cmap=cmap)
    #     USMap[USMap.STATEFP != '15'].plot(column='Org_Arr', ax=axs[1], cmap=cmap)
    #     plotPZ(USMap, pzs, USNet_IDs, axs)
    #     axs[0].set_xlim(-160.5, -154.75)
    #     axs[0].set_ylim(18.75, 22.5)
    #     axs[0].axis('off')
    #     axs[1].axis('off')
    #     plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, wspace=0.05, hspace=0.1)
    #     # cbax = fig.add_axes([0.5, 0.1, 0.75, 0.25])
    #     min = np.min(USMap['Org_Arr'])
    #     max = np.max(USMap['Org_Arr'])
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min, vmax=max))
    #     sm._A = []
    #     fig.colorbar(sm, format="%.1f", orientation='horizontal', shrink=0.7, pad=0.01)
    #     plt.savefig('USNetMaps/Arrivals/USNet_LAX{}_Arrivals.png'.format(i), dpi=600, bbox_inches='tight', pad_inches=0)
    #     # plt.show()

    # pzs = [72589]
    # for i in range(1,8):
    #     USMap = ConCat_Map()
    #     with open('USNet_Results_Org/JFK Tests/US_Arrivals_OrgJFK{}.pkl'.format(i), 'rb') as f:
    #         USNet_Arrivals = pickle.load(f)
    #
    #     AvgArr = AvgArrival(USNet_Arrivals)
    #     print(AvgArr)
    #     USMap = ConCat_Results(USMap, AvgArr, USNet_IDs, 'Org_Arr')
    #     print(USMap)
    #     missing_args = {'color': 'grey', 'edgecolor': 'white', "label": 'Missing Values'}
    #     cmap = plt.get_cmap('Spectral')
    #     fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
    #     USMap[USMap.STATEFP == '15'].plot(color='lightgrey', ax=axs[0])
    #     USMap[USMap.STATEFP != '15'].plot(color='lightgrey', ax=axs[1])
    #     # USMap = USMap.dropna()
    #     USMap[USMap.STATEFP == '15'].plot(column='Org_Arr', ax=axs[0], cmap=cmap)
    #     USMap[USMap.STATEFP != '15'].plot(column='Org_Arr', ax=axs[1], cmap=cmap)
    #     plotPZ(USMap, pzs, USNet_IDs, axs)
    #     axs[0].set_xlim(-160.5, -154.75)
    #     axs[0].set_ylim(18.75, 22.5)
    #     axs[0].axis('off')
    #     axs[1].axis('off')
    #     plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, wspace=0.05, hspace=0.1)
    #     # cbax = fig.add_axes([0.5, 0.1, 0.75, 0.25])
    #     min = np.min(USMap['Org_Arr'])
    #     max = np.max(USMap['Org_Arr'])
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min, vmax=max))
    #     sm._A = []
    #     fig.colorbar(sm, format="%.1f", orientation='horizontal', shrink=0.7, pad=0.01)
    #     plt.savefig('USNetMaps/Arrivals/USNet_JFK{}_Arrivals.png'.format(i), dpi=600, bbox_inches='tight', pad_inches=0)
    #     # plt.show()

    USMap = ConCat_Map()
    with open('USNet_Results_Org/JFK Tests/US_Arrivals_OrgJFK7.pkl', 'rb') as f:
        USNet_Arrivals = pickle.load(f)
    infecprobs = InfecProb_dict(USNet_Arrivals, 100)
    # print(infecprobs)
    USMap = ConCat_Results(USMap, infecprobs, USNet_IDs, 'Org_IfP')
    save = 'InfecProb/USNet_InfecProb_OrgJFK7'
    pzs = [72589]
    bounds = (np.min(USMap['Org_IfP']),np.max(USMap['Org_IfP']))
    print(bounds)
    GenericUSMap(USMap, 'Org_IfP', save, pzs, bounds, color='Spectral_r')
