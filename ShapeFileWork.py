import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import geopandas as gp
import Spielman_Sparse as spl
import scipy.stats as st
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

path = "NCShapeFile/tl_2016_37_tract.shp"
map = gp.read_file(path)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = cl.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def map_plot(sf, pzs, thr, eval, IDS, color='terrain', size=(15, 20), I_cmap=(0, 1)):
    min = np.min(
        [np.min(sf['Wts_{}'.format(eval)]), np.min(sf['Uni_{}'.format(eval)]), np.min(sf['Spl_{}'.format(eval)]),
         np.min(sf['Thr_{}'.format(eval)])])
    max = np.max(
        [np.max(sf['Wts_{}'.format(eval)]), np.max(sf['Uni_{}'.format(eval)]), np.max(sf['Spl_{}'.format(eval)]),
         np.max(sf['Thr_{}'.format(eval)])])

    cmap = plt.get_cmap(color)
    color = truncate_colormap(cmap, minval=I_cmap[0], maxval=I_cmap[1], n=500)

    legend_args = {'label': 'Wasserstein Distance', 'orientation': 'horizontal'}
    missing_args = {'color': 'white', 'edgecolor': 'red', 'hatch': "///", "label": 'Missing Values'}
    nonan = sf.dropna()

    if eval == 'Avg' or eval == 'prob':
        min = np.min(
            [np.min(sf['Wts_{}'.format(eval)]), np.min(sf['Uni_{}'.format(eval)]), np.min(sf['Spl_{}'.format(eval)]),
             np.max(sf['Org_{}'.format(eval)])])
        max = np.max(
            [np.max(sf['Wts_{}'.format(eval)]), np.max(sf['Uni_{}'.format(eval)]), np.max(sf['Spl_{}'.format(eval)]),
             np.max(sf['Org_{}'.format(eval)])])

        fig, ax = plt.subplots(5, 1)
        fig.set_figheight(size[0])
        fig.set_figwidth(size[1])
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        cbax = fig.add_axes([0.75, 0.3, 0.03, 0.39])
        if eval == 'prob':
            cbax.set_title(r'$p_{infected}$')
        else:
            cbax.set_title('{}'.format(eval))

        sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=min, vmax=max))
        sm._A = []
        fig.colorbar(sm, cax=cbax, format="%.2f")

        ax[0].axis('off'), ax[1].axis('off'), ax[2].axis('off'), ax[3].axis('off'), ax[4].axis('off')
        ax[0].set_title('Original Network\n Average: {:.4f}'.format(np.nanmean(nonan['Org_{}'.format(eval)])))
        sf.plot(column='Org_{}'.format(eval), ax=ax[0], cmap=color, legend=False, vmin=min, vmax=max,
                legend_kwds=legend_args, missing_kwds=missing_args)
        ax[1].set_title('Uniform Sampling\n Average: {:.4f}'.format(np.nanmean(nonan['Uni_{}'.format(eval)])))
        sf.plot(column='Uni_{}'.format(eval), ax=ax[1], cmap=color, legend=False, vmin=min, vmax=max,
                legend_kwds=legend_args, missing_kwds=missing_args)
        ax[2].set_title('EffR Sampling\n Average: {:.4f}'.format(np.nanmean(nonan['Spl_{}'.format(eval)])))
        sf.plot(column='Spl_{}'.format(eval), ax=ax[2], cmap=color, legend=False, vmin=min, vmax=max,
                legend_kwds=legend_args, missing_kwds=missing_args)
        ax[3].set_title('Weights Sampling\n Average: {:.4f}'.format(np.nanmean(nonan['Wts_{}'.format(eval)])))
        sf.plot(column='Wts_{}'.format(eval), ax=ax[3], cmap=color, legend=False, vmin=min, vmax=max,
                legend_kwds=legend_args, missing_kwds=missing_args)
        ax[4].set_title(r'Weight Threshold $\geq$' + '{}\n Average: {:.4f}'.format(thr,np.nanmean(nonan['Thr_{}'.format(eval)])))
        sf.plot(column='Thr_{}'.format(eval), ax=ax[4], cmap=color, legend=False, vmin=min, vmax=max,
                legend_kwds=legend_args, missing_kwds=missing_args)

        ids = [x for x in IDS.keys()]
        nodes = {}
        for i in range(len(ids)):
            nodes[i] = ids[i]

        for pz in pzs:
            geotag = nodes[pz]
            pz_df = sf.loc[sf['GEOID'] == geotag]
            pz_df.plot(ax=ax[0], color='red')
            pz_df.plot(ax=ax[1], color='red')
            pz_df.plot(ax=ax[2], color='red')
            pz_df.plot(ax=ax[3], color='red')
            pz_df.plot(ax=ax[4], color='red')

    else:
        fig, ax = plt.subplots(2, 2)
        fig.set_figheight(size[0])
        fig.set_figwidth(size[1])
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        cbax = fig.add_axes([0.9, 0.3, 0.03, 0.39])
        cbax.set_title('{}'.format(eval))

        sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=min, vmax=max))
        sm._A = []
        fig.colorbar(sm, cax=cbax, format="%.2f")

        ax[0, 0].axis('off'), ax[1, 0].axis('off'), ax[0, 1].axis('off'), ax[1, 1].axis('off')
        ax[0, 0].set_title('Uniform Sampling\n Average: {:.4f}'.format(np.nanmean(nonan['Uni_{}'.format(eval)])))
        sf.plot(column='Uni_{}'.format(eval), ax=ax[0, 0], cmap=color, legend=False, vmin=min, vmax=max,
                legend_kwds=legend_args, missing_kwds=missing_args)
        ax[0, 1].set_title('EffR Sampling\n Average: {:.4f}'.format(np.nanmean(nonan['Spl_{}'.format(eval)])))
        sf.plot(column='Spl_{}'.format(eval), ax=ax[0, 1], cmap=color, legend=False, vmin=min, vmax=max,
                legend_kwds=legend_args, missing_kwds=missing_args)
        ax[1, 0].set_title('Weights Sampling\n Average: {:.4f}'.format(np.nanmean(nonan['Wts_{}'.format(eval)])))
        sf.plot(column='Wts_{}'.format(eval), ax=ax[1, 0], cmap=color, legend=False, vmin=min, vmax=max,
                legend_kwds=legend_args, missing_kwds=missing_args)
        ax[1, 1].set_title(r'Weight Threshold $\geq$' + '{}\n Average: {:.4f}'.format(thr,np.nanmean(nonan['Thr_{}'.format(eval)])))
        sf.plot(column='Thr_{}'.format(eval), ax=ax[1, 1], cmap=color, legend=False, vmin=min, vmax=max,
                legend_kwds=legend_args, missing_kwds=missing_args)

        ids = [x for x in IDS.keys()]
        nodes = {}
        for i in range(len(ids)):
            nodes[i] = ids[i]

        for pz in pzs:
            geotag = nodes[pz]
            pz_df = sf.loc[sf['GEOID'] == geotag]
            pz_df.plot(ax=ax[0, 0], color='red')
            pz_df.plot(ax=ax[1, 0], color='red')
            pz_df.plot(ax=ax[0, 1], color='red')
            pz_df.plot(ax=ax[1, 1], color='red')

    if eval == 'prob':
        fig.suptitle("NC, Census Tract Network\n Comparison of" + r' $p_{infected}$ of Tracts', size=16)
    else:
        fig.suptitle("NC, Census Tract Network\n Comparison of {} Arrival Times".format(eval), size=16)

    return ax

    # for i in range(3):
    #     ax[i].annotate(
    #         'Raleigh',
    #         xy=(-78.6382, 35.7796), xycoords='data',
    #         xytext=(-70, 50), textcoords='offset points',
    #         size=12, color='red',
    #         arrowprops=dict(arrowstyle="->",
    #                         connectionstyle="arc,angleA=0,armA=50,rad=10", color='red'))
    #
    # ax[i].annotate(
    #     'Charlotte',
    #     xy=(-80.8431, 35.2271), xycoords='data',
    #     xytext=(-70, 100), textcoords='offset points',
    #     size=16,
    #     color='black',
    #     arrowprops=dict(arrowstyle="->",
    #                     connectionstyle="arc,angleA=0,armA=50,rad=10"))


# Census tract ID sf.records()[i][3]

def get_data(ari_mdl, ari_org, IDS, tag, sim_num=1000):
    tag1, tag2, tag3, tag4 = '{}_WD'.format(tag), '{}_KS'.format(tag), '{}_Avg'.format(tag), '{}_prob'.format(tag)
    tag5, tag6, tag7 = '{}_Diff'.format(tag), '{}_ProbD'.format(tag), '{}_log'.format(tag)
    ids = [x for x in IDS.keys()]
    nodes = {}
    for i in range(len(ids)):
        nodes[i] = ids[i]

    GEOID = []
    data_ks = []
    data_wd = []
    avgs = []
    probs = []
    dif_avg = []
    dif_probs = []
    neglog_wd = []

    for i in range(len(ari_org)):
        if len(ari_mdl[i]) == 0 or len(ari_org[i]) == 0:
            GEOID.append(nodes[i])
            data_ks.append(None)
            data_wd.append(None)
            avgs.append(None)
            probs.append(None)
            dif_probs.append(None)
            dif_avg.append(None)
            neglog_wd.append(None)
        else:
            GEOID.append(nodes[i])

            ks_dist = st.kstest(spl.normprobs(ari_mdl[i]), spl.normprobs(ari_org[i]))[0]
            data_ks.append(ks_dist)

            prob = len(ari_mdl[i]) / sim_num
            probs.append(prob)

            probs_org = len(ari_org[i]) / sim_num
            dif_probs.append(probs_org - prob)

            avg = np.mean(ari_mdl[i])
            avgs.append(avg)

            avg_org = np.mean(ari_org[i])
            dif_avg.append(avg_org - avg)

            ws_dist = st.wasserstein_distance(spl.normprobs(ari_mdl[i]), spl.normprobs(ari_org[i]))
            data_wd.append(ws_dist)
            neglog_wd.append(-1 * np.log(ws_dist))

    df = pd.DataFrame.from_dict({'GEOID': GEOID, tag1: data_wd,
                                 tag2: data_ks, tag3: avgs, tag4: probs,
                                 tag5: dif_avg,
                                 tag6: dif_probs,
                                 tag7: neglog_wd})

    return df

# S = nx.from_numpy_matrix(spl_net.adj())
# S = nx.relabel_nodes(S, nodes)
# S = nx.from_numpy_matrix(spl_net.adj())
# S = nx.relabel_nodes(S, nodes)
# U = nx.from_numpy_matrix(uni_net.adj())
# U = nx.relabel_nodes(U, nodes)
# W = nx.from_numpy_matrix(wts_net.adj())
# W = nx.relabel_nodes(W, nodes)
#
# min, max = np.min([np.min(spl_net.weight()), np.min(uni_net.weight()), np.min(wts_net.weight())]), np.max([np.max(spl_net.weight()), np.max(uni_net.weight()), np.max(wts_net.weight())])
# nx.draw_networkx_edges(U, pos=A.pos, width=0.2, edge_cmap='Greys', alpha=0.2, edge_vmin=min, edge_vmax=max, ax=ax[0])
# nx.draw_networkx_edges(S, pos=A.pos, width=0.2, edge_cmap='Greys', alpha=0.2, edge_vmin=min, edge_vmax=max, ax=ax[1])
# nx.draw_networkx_edges(W, pos=A.pos, width=0.2, edge_cmap='Greys', alpha=0.2, edge_vmin=min, edge_vmax=max, ax=ax[2])
# diffs = pd.DataFrame({'Spl_Dif':map_data['Spl_Avg']-map_data['Org_Avg'], 'Wts_Dif':map_data['Wts_Avg']-map_data['Org_Avg'], 'Uni_Dif':map_data['Uni_Avg']-map_data['Org_Avg'], 'GEOID':map_data['GEOID']})
# map_data = map_data.merge(diffs, on='GEOID')

# map_data = map
# for df in [org_df, spl_df, uni_df, wts_df]:
#     map_data = map_data.merge(df, on='GEOID')
