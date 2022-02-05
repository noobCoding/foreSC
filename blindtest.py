# -*- coding: utf-8 -*-
import numpy as np

import pandas as pd

from foresc import foresc
from helpers import clustering,measure, outliers_detection
from config import config
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


font = {'family': 'normal',
        'weight': 'bold',
        'size': 11}
mpl.rc('font', **font)
sns.set_style("ticks")
#
colors =['#e6194b', '#3cb44b', '#E0C800', '#4363d8', '#f58231', '#a9a9a9', '#911eb4',
         '#46f0f0','#469990', '#9a6324', '#800000', '#6fff9d', '#808000', '#000075', '#bcf60c']

# current_palette =sns.color_palette(colors)

sns.set_palette(palette=colors)
current_palette=sns.color_palette()

markers = ['.', '*', '+', '2', '.', '*', '+','2',  '.', '*', '+','2',
           '.', '*', '+', '2', '.', '*', '+', '2', '.', '*', '+', '2', ]

def sparsity_of_this(arr):
    if len(arr) == 0:
        return 0.

    r,c = arr.shape
    s = 0
    for a in arr:
        s = s + len(np.where(a==0.0)[0])
    return 1. * s / (r*c)

if __name__ == '__main__':

    config = {
        'epoch': 10000,
        'batch_size': 256,
        'latent': 2,
        'log': True,
        'patience': 50,
    }

    head = []
    label_set = []
    expr = []

    # DATASET = 'baron_57_human1'  # sys.argv[1]
    #
    # filename = DATASET + '.csv'
    # data = open(filename)
    # tmp = data.readline().rstrip().split(',')
    # # print len(genelist)
    #
    # for line in data:
    #     temp = line.rstrip().split(',')
    #
    #     # celltype
    #     c = temp[2]
    #     if c not in label_set:
    #         label_set.append(c)
    #
    #     # label list = head
    #     head.append(c)
    #
    #     # expression data of current sample
    #     del (temp[0])
    #     del (temp[0])
    #     del (temp[0])
    #     temp = [float(x) for x in temp]
    #     expr.append(temp)

    DATASET = 'baron_58_human2'  # sys.argv[1].65   .36 .78 .54 .63

    filename = DATASET + '.csv'
    data = open(filename)
    genelist = data.readline().rstrip().split(',') # level 2
    del(genelist[0])
    del (genelist[0])
    del (genelist[0])
    print len(genelist)

    for line in data:
        temp = line.rstrip().split(',')
        # print temp

        # celltype
        c = temp[2]
        if c not in label_set:
            label_set.append(c)

        # label list = head
        head.append(c)

        # expression data of current sample
        del (temp[0])
        del (temp[0])
        del (temp[0])
        temp = [float(x) for x in temp]
        expr.append(temp)

    # DATASET = 'baron_59_human3'  # sys.argv[1]
    #
    # filename = DATASET + '.csv'
    # data = open(filename)
    # tmp = data.readline().rstrip().split(',')
    # # print len(genelist)
    #
    # for line in data:
    #     temp = line.rstrip().split(',')
    #
    #     # celltype
    #     c = temp[2]
    #     if c not in label_set:
    #         label_set.append(c)
    #
    #     # label list = head
    #     head.append(c)
    #
    #     # expression data of current sample
    #     del (temp[0])
    #     del (temp[0])
    #     del (temp[0])
    #     temp = [float(x) for x in temp]
    #     expr.append(temp)

    # DATASET = 'baron_60_human4'  # alpha=1, beta=1.8
    #
    # filename = DATASET + '.csv'
    # data = open(filename)
    # tmp = data.readline().rstrip().split(',')
    # # print len(genelist)
    #
    # for line in data:
    #     temp = line.rstrip().split(',')
    #
    #     # celltype
    #     c = temp[2]
    #     if c not in label_set:
    #         label_set.append(c)
    #     head.append(c)
    #
    #     # expression data of current sample
    #     del (temp[0])
    #     del (temp[0])
    #     del (temp[0])
    #     temp = [float(x) for x in temp]
    #     expr.append(temp)

    expr_in = np.asarray(expr)  # NO transpose
    print(expr_in.shape)
    print(sparsity_of_this(expr_in))

    # print "label_set:"
    # print label_set

    name_map = {value: idx for idx, value in enumerate(label_set)}
    id_map = {idx: idx for idx, value in enumerate(label_set)}
    label = np.asarray([name_map[name] for name in head])

    print (len(np.unique(label)))
    print (np.unique(label))

    pop = {}
    sorted_pop = []
    sorted_label = []

    for i in range(0, len(np.unique(label))):
        tmp = np.where(label==i)[0]
        pop[i] = len(tmp)

    print (pop)
    sorted_pop = [(v,k) for k,v in pop.iteritems() ]
    # sorted_pop = [(v,k) for k,v in pop.items() ]  #pyt3
    sorted_pop.sort()
    print (sorted_pop)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    n_cell, _ = expr_in.shape
    if n_cell > 150:
        batch_size = 256
    else:
        batch_size = 32

    ### analysis results
    s = 40
    if n_cell <= 500:
        s = 80

    mm = 1.
    ss = 0.5
# highly variable genes ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    X = np.copy(expr_in)
    X = X.transpose()  # genes * cells

    # mean, var = materialize_as_ndarray(_get_mean_var(X))
    mean = X.mean(axis=0)
    # if issparse(X):
    #     mean_sq = X.multiply(X).mean(axis=0)
    #     mean = mean.A1
    #     mean_sq = mean_sq.A1
    # else:
    mean_sq = np.multiply(X, X).mean(axis=0)

    # enforece R convention (unbiased estimator) for variance
    var = (mean_sq - mean ** 2) * (X.shape[0] / (X.shape[0] - 1))

    # now actually compute the dispersion
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean

    # if flavor == 'seurat':  # logarithmized mean as in Seurat
    dispersion[dispersion == 0] = np.nan
    dispersion = np.log(dispersion)
    mean = np.log1p(mean)

    # all of the following quantities are "per-gene" here
    df = pd.DataFrame()
    df['mean'] = mean
    df['dispersion'] = dispersion

    n_bins = 20

    df['mean_bin'] = pd.cut(df['mean'], bins=n_bins)
    disp_grouped = df.groupby('mean_bin')['dispersion']
    disp_mean_bin = disp_grouped.mean()
    disp_std_bin = disp_grouped.std(ddof=1)
    # retrieve those genes that have nan std, these are the ones where
    # only a single gene fell in the bin and implicitly set them to have
    # a normalized disperion of 1
    one_gene_per_bin = disp_std_bin.isnull()
    gen_indices = np.where(one_gene_per_bin[df['mean_bin'].values])[0].tolist()
    # if len(gen_indices) > 0:
    #     logg.msg(
    #         'Gene indices {} fell into a single bin: their '
    #         'normalized dispersion was set to 1.\n    '
    #         'Decreasing `n_bins` will likely avoid this effect.'
    #             .format(gen_indices),
    #         v=4
    #     )
    # # Circumvent pandas 0.23 bug. Both sides of the assignment have dtype==float32,

    disp_std_bin[one_gene_per_bin.values] = disp_mean_bin[one_gene_per_bin.values].values
    disp_mean_bin[one_gene_per_bin.values] = 0

    # actually do the normalization
    df['dispersion_norm'] = (
            (
                    df['dispersion'].values  # use values here as index differs
                    - disp_mean_bin[df['mean_bin'].values].values
            ) / disp_std_bin[df['mean_bin'].values].values
    )

    ####  n%  additional information
    # additional_ratio = 0.3  ### 30%
    # selected_index = np.where(goodgene > (m * meangood + s * sdgood))[0]
    # selected_size = len(selected_index)
    # print 'selected_size:'
    # print selected_size
    #
    # new_selected_size = int(round((1.0 + additional_ratio) * selected_size, ndigits=0))
    # print 'new selected_size:'
    # print new_selected_size
    #
    # k_times = 2
    # n_top_genes = int(round(k_times * new_selected_size, ndigits=0))
    # print 'n_top_genes:'
    # print n_top_genes

    n_top_genes = 1000

    dispersion_norm = df['dispersion_norm'].values.astype('float32')

    print (len(dispersion_norm))
    print (dispersion_norm)

    dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
    dispersion_norm[::-1].sort()  # interestingly, np.argpartition is slightly slower
    disp_cut_off = dispersion_norm[n_top_genes - 1]
    gene_subset = np.nan_to_num(df['dispersion_norm'].values) >= disp_cut_off

    print ('n-top genes:::::')
    print (len(gene_subset))
    print (len(np.where(gene_subset == True)[0]))

    highly_variable_genes = np.where(gene_subset == True)[0] # 1,000 top genes

    # [x for x in item if x not in z]
    # might losing duplicates of non-unique elements):
    #  set(item) - set(z)
    # additional_genes = list(set(highly_variable_genes) - set(selected_index))
    # print ('additional genes')
    # print len(additional_genes)
    #
    # n = 0
    # while (n < (new_selected_size - selected_size)):
    #     selected_index = np.append(selected_index, additional_genes[n])
    #     n = n + 1
    #
    # print ('new index:')
    # print len(selected_index)

    # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    for j in range(6,0,-1):
        tmplabel = np.asarray([], dtype='int')
        for v in range(0,j):
            # ind = np.where(label==pop.keys()[pop.values().index(sorted_pop[v])]) # label of a sorted_pop
            ind = np.where(label==sorted_pop[v][1])
            tmplabel = np.append(tmplabel, ind)
        print (len(tmplabel))
        # print tmplabel

        train_expr = np.copy(expr_in)
        train_expr= np.delete(train_expr, tmplabel, axis=0)

        train_label = np.copy(label)
        train_label= np.delete(train_label, tmplabel, axis=0)

        test_expr = expr_in[tmplabel]
        test_label = label[tmplabel]

        print (train_expr.shape)
        print (train_label.shape)
        # print train_label
        #
        print (test_expr.shape)
        print (test_label.shape)
        # print test_label

        myforesc = foresc(train_expr,
                          latent=2,
                          batch_size=batch_size,
                          prefix=DATASET,
                          label=train_label,
                          patience=50,
                          fs=True, mm=mm, ss=ss,
                          log=2 # means log2
                          )
        res = myforesc['res']

        print ("foreSC:")
        cl, model = clustering(res, k=len(np.unique(train_label)), name=myforesc['method'])
        dm = measure(cl, train_label, scrprint=True)

        # fig = plt.figure()
        # fig = plt.figure(j, figsize=(15,5))
        # ax = plt.subplot(131)
        # plt.title('foreSC', fontsize=13)
        # for i in np.unique(train_label):
        #     ax.scatter(res[train_label == i, 0],
        #                res[train_label == i, 1],
        #                c=current_palette[i],
        #                label=label_set[i],
        #                s=s,
        #                marker=markers[i])
        # sns.despine()
        # plt.draw()
        # fig.savefig(DATASET + '-train-' + str(14-j) + '.png', bbox_inches='tight')

        from keras.models import load_model
        from keras.models import model_from_json

        # load json and create model
        json_file = open(DATASET + '-model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        dcvae = model_from_json(loaded_model_json)

        # load weights into new model
        dcvae.load_weights(DATASET + "model.h5")
        print("Loaded model from disk")

        # load gene ranking from file
        goodgene = np.loadtxt(DATASET + '_gene_inportance.txt')
        meangood = np.mean(goodgene[np.where(goodgene > 0.)[0]])
        sdgood = np.std(goodgene[np.where(goodgene > 0.)[0]])
        m = mm
        s = ss
        # print goodgene
        print (np.max(goodgene))
        print (meangood)
        print (sdgood)

        filtered_index = np.where(goodgene <= (m * meangood + s * sdgood))[0]
        # expr_mod = np.copy(test_expr)
        # expr_mod = np.delete(expr_mod, filtered_index, axis=1)
        # print expr_mod.shape

        #  set(item) - set(z)
        selected_index = np.where(goodgene > (m * meangood + s * sdgood))[0]
        additional_genes = list(set(highly_variable_genes) - set(selected_index))
        print ('additional genes')
        print (len(additional_genes))

        selected_size = len(selected_index)
        new_selected_size = int(round(selected_size * 1.3))
        if new_selected_size > 1000 :
             new_selected_size = 1000

        if (new_selected_size - selected_size) < len(additional_genes):
            if (new_selected_size > selected_size):
                selected_index = np.append(selected_index, additional_genes[0:(new_selected_size - selected_size - 1)])
            else:
                selected_index = np.append(selected_index, additional_genes[:])

        print ('new index:')
        print (len(selected_index))

    #############################################################################################################3
        from sklearn.decomposition import PCA

        test_expr[test_expr <= 0.] = 0.0
        test_expr= np.log2(1 + test_expr)
        texpr = test_expr[:, selected_index]

        ## coarse PCA
        tpca = PCA(n_components=2).fit_transform(texpr)
        tpca = tpca / 10.

        print("temp PCA")
        # cl1, _ = clustering(pca, k=len(np.unique(test_label)), name='kmeans')
        # measure(cl1, test_label, scrprint=True)

        km = KMeans(n_clusters=len(np.unique(test_label)), n_init=100).fit(tpca)
        tmpnewlabel = km.predict(tpca)  ######## we don't actually use this variable in calculation
        # print tmpnewlabel

        #####################################################################################################        #
        # km =  KMeans( n_clusters=len(np.unique(test_label)),n_init=100 ).fit(expr_mod)
        # tmpnewlabel = km.predict(expr_mod)
        # print tmpnewlabel

        ### refined
        blind_res = foresc(test_expr,
                          latent=2,
                          batch_size=batch_size,
                          prefix='blind',
                          label=tmpnewlabel,
                          patience=50,
                          fs=False,
                          log=0  # don't take log twice
                          )
        test_res = blind_res['res']
        print (test_res.shape)

        print ("blind foreSC:")
        clb, modelb = clustering(test_res, k=len(np.unique(test_label)), name='kmeans')
        dmb = measure(clb, test_label, scrprint=True)

        mix_data = np.concatenate((res, test_res))
        print (mix_data.shape)
        mix_label = np.concatenate((train_label, test_label))
        print (mix_label.shape)
        mix_clb, md = clustering(mix_data, k=len(np.unique(mix_label)), name='kmeans')
        mix_dmb = measure(mix_clb, mix_label, scrprint=True)

######################33 Other methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~````
        p0_thresh = .95
        expr_in[expr_in< 0.] = 0.0
        expr_log = np.log2(1 + expr_in)
        expr_filtered = expr_log[:, highly_variable_genes]
        print (expr_filtered.shape)

        # PCA   ///////////////////////////////////////
        # from sklearn.decomposition import PCA

        pca = PCA(n_components=2).fit_transform(expr_filtered)
        pca = pca / 10.

        print("PCA")
        cl1, _ = clustering(pca, k=len(np.unique(label)), name='kmeans')
        measure(cl1, label, scrprint=True)

        # t-SNE     /////////////////
        from sklearn.manifold import TSNE

        if expr_in.shape[1] > 500:
            expr_tsne = PCA(n_components=500).fit_transform(expr_filtered)
        else:
            expr_tsne = np.copy(expr_filtered)

        tsne = TSNE(n_components=2, perplexity=.2 * n_cell).fit_transform(expr_tsne)

        print("t-SNE")
        cl2, _ = clustering(tsne, k=len(np.unique(label)), name='kmeans')
        measure(cl2, label, scrprint=True)

        # SIMLR     ////////////////////////////////////////////////////////
        import SIMLR

        if expr_in.shape[1] > 500:
            expr_simlr = PCA(n_components=500).fit_transform(expr_filtered)
        else:
            expr_simlr = np.copy(expr_filtered)

        simlr = SIMLR.SIMLR_LARGE(len(np.unique(label)), 25, 0)  ###This is how we initialize an object for SIMLR. the first input is number of rank (clusters) and the second input is number of neighbors. The third one is an binary indicator whether to use memory-saving mode. you can turn it on when the number of cells are extremely large to save some memory but with the cost of efficiency.
        S, F, val, ind = simlr.fit(expr_simlr)

        simlr2d = TSNE(n_components=2, perplexity=.2 * len(np.unique(label))).fit_transform(F)

        print("SIMLR")
        cl4, _ = clustering(simlr2d, k=len(np.unique(label)), name='kmeans')
        measure(cl4, label, scrprint=True)

        # VASC      ///////////////////////////////////////////////////////
        from vasc import vasc

        vasc2d = vasc(expr_filtered, var=True,
        # vasc2d=vasc(expr_in, var=True,
                      latent=2,
                      annealing=True,
                      batch_size=batch_size,
                      prefix=DATASET,
                      label=label,
                      scale=True,
                      log=False,
                      patience=50)

        print("VASC")
        cl5, _ = clustering(vasc2d, k=len(np.unique(label)), name='kmeans')
        measure(cl5, label, scrprint=True)

        # ####ZIFA      /////////////////////////////////////////////////////////
        from ZIFA import block_ZIFA

        zifa, _ = block_ZIFA.fitModel(expr_log, 2,
                                      p0_thresh=p0_thresh)  # ZIFA has built-in filtering sys. so we just use expr_log

        print("ZIFA")
        cl3, _ = clustering(zifa, k=len(np.unique(label)), name='kmeans')
        dm = measure(cl3, label, scrprint=True)

        # print test_label
        # print len(np.unique(test_label))
        # print np.unique(test_label)
        #
        #
        # # fig = plt.figure(2, figsize=(5, 5))
        # ax2 = plt.subplot(132)
        # plt.title('blind', fontsize=13)
        # for i in np.unique(test_label):
        #     ax2.scatter(test_res[test_label == i, 0],
        #                test_res[test_label == i, 1],
        #                c=current_palette[i],
        #                label=label_set[i],
        #                s=s*128,
        #                marker=markers[i])
        #     # print i
        #     # print label_set[i]
        #     # print current_palette[i]
        #     # print markers[i]
        #
        #
        # sns.despine()
        # # plt.draw()
        # # fig.savefig(DATASET + '-blind-' + str(14-j) + '.png', bbox_inches='tight')
        #
        # # fig = plt.figure(3, figsize=(5, 5))
        # ax3 = plt.subplot(133)
        # plt.title('both', fontsize=13)
        # for i in np.unique(train_label):
        #     ax3.scatter(res[train_label == i, 0],
        #                res[train_label == i, 1],
        #                c=current_palette[i],
        #                label=label_set[i],
        #                s=s,
        #                marker=markers[i])
        #
        # for i in np.unique(test_label):
        #     ax3.scatter(test_res[test_label == i, 0],
        #                test_res[test_label == i, 1],
        #                c=current_palette[i],
        #                label=label_set[i],
        #                s=s*128,
        #                marker=markers[i])
        # sns.despine()
        #
        # # mng = plt.get_current_fig_manager()
        # # mng.window.showMaximized()
        # fig.tight_layout()
        #
        # plt.draw()
        # # plt.show()
        # fig.savefig(DATASET + '-both-only-foreSC-' + str(14-j) + '.png', bbox_inches='tight')