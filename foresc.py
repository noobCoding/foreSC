# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, Activation, Lambda, RepeatVector, merge, Reshape, Layer, Dropout, \
    BatchNormalization, Permute, Concatenate, Multiply
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from keras.models import Model
from helpers import measure, clustering, cart2polar, outliers_detection
from keras import regularizers
from keras.utils.layer_utils import print_summary
from keras.utils import plot_model
import numpy as np
from keras.optimizers import RMSprop, Adagrad, Adam
from keras import metrics
import h5py
import matplotlib.pyplot as plt


def sampling(args):
    epsilon_std = 1.0

    if len(args) == 2:
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean),
                                  mean=0.,
                                  stddev=epsilon_std)
        #
        return z_mean + K.exp(z_log_var / 2) * epsilon
    else:
        z_mean = args[0]
        epsilon = K.random_normal(shape=K.shape(z_mean),
                                  mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(1.0 / 2) * epsilon


def sampling_gumbel(shape, eps=1e-8):
    u = K.random_uniform(shape)
    return -K.log(-K.log(u + eps) + eps)


def compute_softmax(logits, temp):
    z = logits + sampling_gumbel(K.shape(logits))
    return K.softmax(z / temp)


def gumbel_softmax(args):
    logits, temp = args
    # return compute_softmax(logits, temp)
    # return  K.softmax( (logits + sampling_gumbel(K.shape(logits))) / temp)
    eps = 1e-8
    return K.softmax((logits + -K.log(-K.log(K.random_uniform(K.shape(logits))) + eps) + eps) / temp)

class BESC:
    def __init__(self, in_dim, attr_num=0, latent=2, var=False, fs=True, blind=False):
        self.in_dim = in_dim
        self.attr = attr_num
        self.vae = None
        self.ae = None
        self.aux = None
        self.latent = latent
        self.var = var
        self.fs = fs
        self.blind = blind

    def vaeBuild(self):
        var_ = self.var
        in_dim = self.in_dim
        cat_dim = self.attr
        expr_in = Input(shape=(self.in_dim,))
        cat_in = Input(shape=(self.attr,))

        layer_1 = 1024
        layer_2 = 256
        layer_3 = 64

        ##### The first part of model to recover the expr.
        # h0 = Concatenate(axis=-1)([expr_in, cat_in]) # CHANGED !!!
        # h0 = Dropout(0.5)(h0)  # ORDER
        h0 = Dropout(0.5)(expr_in)  # ORDER
        h0 = Concatenate(axis=-1)([h0, cat_in]) # CHANGED !!!
        ## Encoder layers
        h1 = Dense(units=in_dim / 10, kernel_regularizer=regularizers.l1(0.01))(h0)
        h2 = Dense(units=layer_1, name='encoder_2')(h1)
        h2_relu = Activation('relu')(h2)
        h2_relu = Dense(units=layer_2, name='encoder_22')(h2_relu)
        h2_relu = Activation('relu')(h2_relu)
        h3 = Dense(units=layer_3, name='encoder_3')(h2_relu)
        h3_relu = Activation('relu')(h3)

        z_mean = Dense(units=self.latent, name='z_mean')(h3_relu)
        # if self.var:
        z_log_var = Dense(units=self.latent, name='z_log_var')(h3_relu)
        z_log_var = Activation('softplus')(z_log_var)

        ## sampling new samples
        z = Lambda(sampling, output_shape=(self.latent,))([z_mean, z_log_var])
        # else:
        #     z = Lambda(sampling, output_shape=(self.latent,))([z_mean])

        ## Decoder layers
        z = Concatenate(axis=-1)([z, cat_in])
        decoder_h1 = Dense(units=layer_3, name='decoder_1')(z)
        decoder_h1_relu = Activation('relu')(decoder_h1)
        decoder_h2 = Dense(units=layer_2, name='decoder_2')(decoder_h1_relu)
        decoder_h2_relu = Activation('relu')(decoder_h2)
        decoder_h2_relu = Dense(units=layer_1, name='decoder_22')(decoder_h2_relu)
        decoder_h2_relu = Activation('relu')(decoder_h2_relu)
        decoder_h3 = Dense(units=in_dim / 10, name='decoder_3')(decoder_h2_relu)
        decoder_h3_relu = Activation('relu')(decoder_h3)
        # decoder_h3_relu = Concatenate(axis=-1)([decoder_h3_relu, cat_in])
        expr_x = Dense(units=self.in_dim, activation='sigmoid')(decoder_h3_relu)

        expr_x_drop = Lambda(lambda x: -x ** 2)(expr_x)
        expr_x_drop_p = Lambda(lambda x: K.exp(x))(expr_x_drop)
        expr_x_nondrop_p = Lambda(lambda x: 1 - x)(expr_x_drop_p)
        expr_x_nondrop_log = Lambda(lambda x: K.log(x + 1e-20))(expr_x_nondrop_p)
        expr_x_drop_log = Lambda(lambda x: K.log(x + 1e-20))(expr_x_drop_p)
        expr_x_drop_log = Reshape(target_shape=(self.in_dim, 1))(expr_x_drop_log)
        expr_x_nondrop_log = Reshape(target_shape=(self.in_dim, 1))(expr_x_nondrop_log)
        logits = Concatenate()([expr_x_drop_log, expr_x_nondrop_log])

        temp_in = Input(shape=(self.in_dim,))
        temp_ = RepeatVector(2)(temp_in)
        # print(temp_.shape)
        temp_ = Permute((2, 1))(temp_)
        samples = Lambda(gumbel_softmax, output_shape=(self.in_dim, 2,))([logits, temp_])
        samples = Lambda(lambda x: x[:, :, 1])(samples)
        samples = Reshape(target_shape=(self.in_dim,))(samples)
        ##        #print(samples.shape)
        out = Multiply()([expr_x, samples])

        ################################################3
        # Classifier
        h4 = LeakyReLU(0.01)(z_mean)
        cat = Dense(units=self.attr, activation='softmax')(h4)

        ########################
        class VariationalLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(VariationalLayer, self).__init__(**kwargs)

            def vae_loss(self, x, x_decoded_mean, cat, cat_in):
                # D = min(cat_dim, 8)
                # xent_loss = 1. / D *metrics.categorical_crossentropy(x, x_decoded_mean)
                xent_loss = metrics.categorical_crossentropy(x, x_decoded_mean)

                # if var_:
                # kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                kl_loss = 0.5 * K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=-1)
                # else:
                #     kl_loss = - 0.5 * K.sum(1 + 1 - K.square(z_mean) - K.exp(1.0), axis=-1)

                cat_loss = metrics.categorical_crossentropy(cat_in, cat)
                # cat_loss = metrics.mean_squared_error(cat, cat_in)

                return K.mean(xent_loss + kl_loss + 20. * cat_loss)
                # return K.mean(xent_loss + kl_loss)

            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean = inputs[1]
                cat = inputs[2]
                cat_in = inputs[3]
                loss = self.vae_loss(x, x_decoded_mean, cat, cat_in)
                self.add_loss(loss, inputs=inputs)
                # We won't actually use the output.
                return x

        class VariationalLayer2(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(VariationalLayer2, self).__init__(**kwargs)

            def vae_loss(self, x, x_decoded_mean, cat, cat_in):
                # D = min(cat_dim, 8)
                # xent_loss = 1. / D *metrics.categorical_crossentropy(x, x_decoded_mean)
                xent_loss = metrics.categorical_crossentropy(x, x_decoded_mean)

                # if var_:
                # kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                kl_loss = 0.5 * K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=-1)
                # else:
                #     kl_loss = - 0.5 * K.sum(1 + 1 - K.square(z_mean) - K.exp(1.0), axis=-1)

                cat_loss = metrics.categorical_crossentropy(cat_in, cat)
                # cat_loss = metrics.mean_squared_error(cat, cat_in)

                return K.mean(xent_loss + kl_loss + 100.0 * cat_loss)
                # return K.mean(xent_loss + kl_loss)

            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean = inputs[1]
                cat = inputs[2]
                cat_in = inputs[3]
                loss = self.vae_loss(x, x_decoded_mean, cat, cat_in)
                self.add_loss(loss, inputs=inputs)
                # We won't actually use the output.
                return x

        if self.fs==True and self.blind==False:
            y = VariationalLayer()([expr_in, out, cat, cat_in]) #
        else:
            y = VariationalLayer2()([expr_in, out, cat, cat_in])  #

        vae = Model(inputs=[expr_in, temp_in, cat_in], outputs=[y])
        # vae = Model( inputs= [expr_in,temp_in, cat_in],outputs=[y, dis] )

        opt = RMSprop(lr=0.001)
        vae.compile(optimizer=opt, loss=None)
        ae = Model(inputs=[expr_in, temp_in, cat_in], outputs=[h1, h2, h3, h2_relu, h3_relu,
                                                               z_mean, z, decoder_h1, decoder_h1_relu,
                                                               decoder_h2, decoder_h2_relu, decoder_h3, decoder_h3_relu,
                                                               h4,
                                                               samples, out, cat
                                                               ])
        aux = Model(inputs=[expr_in, temp_in, cat_in], outputs=[out, cat])

        self.vae = vae
        self.ae = ae
        self.aux = aux


def besc(expr,
          epoch=1000,
          latent=2,
          patience=50,
          min_stop=500,
          batch_size=32,
          var=False,
          prefix='test',
          label=None,
          log=10,
          annealing=False,
          tau0=1.0,
          min_tau=0.5,
          fs=True,
           mm=1.,
           ss=1.,
           blind=False):
    '''
    beSC: Random Forest featuring Variational Autoencoder for scRNA-seq datasets

    ============
    Parameters:
        expr: expression matrix (n_cells * n_features)
        epoch: maximum number of epochs, default 5000
        latent: dimension of latent variables, default 2
        patience: stop if loss showes insignificant decrease within *patience* epochs, default 50
        min_stop: minimum number of epochs, default 500
        batch_size: batch size for stochastic optimization, default 32
        var: whether to estimate the variance parameters, default False
        prefix: prefix to store the results, default 'test'
        label: numpy array of true labels, default None
        log: if log-transformation should be performed, default True
        scale: if scaling (making values within [0,1]) should be performed, default True
        annealing: if annealing should be performed for Gumbel approximation, default False
        tau0: initial temperature for annealing or temperature without annealing, default 1.0
        min_tau: minimal tau during annealing, default 0.5
        rep: not used

    =============
    Values:
        point: dimension-*latent* results
    '''
    expr[expr <= 0.] = 0.0

    if log==10:
        expr = np.log10(expr + 1.)
    else:
        if log==2:
            expr = np.log2(expr + 1.)
        else:
            expr = expr

    expr_train = expr

    if (fs):

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=689, max_features='auto' , max_depth=None, class_weight='balanced', random_state=None, criterion='gini' )
        clf.fit(expr_train, label)
        goodgene = clf.feature_importances_
        np.savetxt(prefix+'_gene_inportance.txt', goodgene, fmt='%.20f')

        # y_pred = clf.predict(expr_train)    #
        # from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        # print(confusion_matrix(label, y_pred))
        # print(classification_report(label, y_pred))
        # print(accuracy_score(label, y_pred))

        meangood = np.mean(goodgene[np.where(goodgene > 0.)[0]])
        sdgood = np.std(goodgene[np.where(goodgene > 0.)[0]])
        if meangood == np.nan:
            meangood = 0.
        if sdgood == np.nan:
            sdgood = 0.

        m = mm
        s = ss
        # print goodgene
        print (np.max(goodgene))
        print (meangood)
        print (sdgood)

        # filtered_index = np.where(goodgene < (alpha * sdgood))[0]
        filtered_index = np.where(goodgene <= (m*meangood + s*sdgood) )[0]
        selected_index = np.where(goodgene > (m*meangood + s*sdgood) )[0]
        # filtered_index = np.where(goodgene <= alpha*meangood)[0]
        # filtered_index = np.where(goodgene <= 0.0)[0]
        # print filtered_index
        # print np.max(filtered_index)
        # print expr_train.shape
        np.savetxt(prefix + '_gene_filtered_idx.txt', filtered_index, fmt='%d')
        np.savetxt(prefix + '_gene_selected_idx.txt', selected_index, fmt='%d')

        # filtered_data = expr_train
        # top_data = np.where(goodgene > (alpha*meangood+beta*sdgood))[0]
        # filtered_data = np.delete(filtered_data, top_data, axis=1)

        # plt.boxplot(filtered_data[0:20])
        # plt.savefig('filtereddata.png')
        expr_train = np.delete(expr_train, filtered_index, axis=1)

    print (expr_train.shape)

    # if rep > 0:
    #     expr_train = np.matlib.repmat(expr_train, rep, 1)
    # else:
    #     expr_train = np.copy(expr_train)

    label_map = {value:idx for idx, value in enumerate(np.unique(label))}

    # cat_in = np.zeros(shape=(len(label), 14), dtype='float32')
    cat_in = np.zeros(shape=(len(label), len(np.unique(label))), dtype='float32')
    # for i in label:
    #     cat_in[:, ] = 1.  # (i + 1.) / (np.max(label) + 1.)
    if not blind:
        for i in label_map:
            cat_in[:, label_map[i]] = 1.  # (i + 1.) / (np.max(label) + 1.)
    else:
            cat_in[:, :] = 0.  # (i + 1.) / (np.max(label) + 1.)

    vae_ = BESC(in_dim=expr_train.shape[1], attr_num=len(np.unique(label)), latent=latent, var=var, fs=fs, blind=blind)

    vae_.vaeBuild()

    bestSil = 0.
    bestNMI = 0.0
    bestARI = 0.0
    bestmethod = ''
    bestres=[]

    loss = []
    prev_loss = np.inf
    # tau0 = 1.
    tau = tau0
    # min_tau = 0.5
    anneal_rate = 0.0003

    ErrorCount = 0
    for e in range(epoch):
        cur_loss = prev_loss

        # mask = np.ones( expr_train.shape,dtype='float32' )
        # mask[ expr_train==0 ] = 0.0
        if e % 100 == 0 and annealing:
            tau = max(tau0 * np.exp(-anneal_rate * e), min_tau)
            # print(tau)

        tau_in = np.ones(expr_train.shape, dtype='float32') * tau

        loss_ = vae_.vae.fit([expr_train, tau_in, cat_in], expr_train, epochs=1, batch_size=batch_size,
                             shuffle=True, verbose=0
                             )

        train_loss = loss_.history['loss'][0]
        # cur_loss = min(train_loss, cur_loss)
        loss.append(train_loss)
        # val_loss = -loss.history['val_loss'][0]
        res = vae_.ae.predict([expr_train, tau_in, cat_in])

        myres = res[5]
        # print myres.shape

        if label is not None:
            k = len(np.unique(label))
            try:
                cl, silhoutte, _ = clustering(myres, k=k, name='kmeans')
                dm = measure(cl, label,scrprint=False)
                if (dm['ARI'] > bestARI):
                # if (silhoutte > bestSil):
                    bestARI = dm['ARI']
                    bestNMI = dm['NMI']
                    bestres = myres
                    bestSil = silhoutte
                    bestmethod = 'kmeans'
                    # vae_.ae.save('blind-model-' + prefix)

                    # serialize model to JSON
                    model_json = vae_.ae.to_json()
                    with open(prefix + "-model.json", "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    vae_.ae.save_weights(prefix + "model.h5")
                    # print("Saved model to disk")

                # cl1, _ = clustering(myres, k=k, name='meanshift')
                # dm1 = measure(cl1, label)
                # if (dm1['ARI'] > bestARI):
                #     bestARI = dm1['ARI']
                #     bestNMI = dm1['NMI']
                #     vae_.ae.save('bestmodel-' + prefix)
                #     bestres = myres
                #     bestmethod = 'meanshift'

                # cl2, _ = clustering(myres, k=k, name='gmm')
                # dm2 = measure(cl2, label)
                # if (dm2['ARI'] > bestARI):
                #     bestARI = dm2['ARI']
                #     bestNMI = dm2['NMI']
                #     vae_.ae.save('bestmodelARI')
                #     bestres = myres
                #     bestmethod = 'gmm'

                # cl3, _ = clustering(myres, k=k, name='dbscan')
                # dm3 = measure(cl3, label)
                # if (dm2['ARI'] > bestARI):
                #     bestARI = dm3['ARI']
                #     bestNMI = dm3['NMI']
                #     vae_.ae.save('bestmodelARI')
                #     bestres = myres
                #     bestmethod = 'dbscan'
            except:
                # print('Clustering error')
                ErrorCount = ErrorCount + 1

        # if e % patience == 1:
        #     # print("Epoch %d/%d" % (e + 1, epoch))
        #     # print("Loss:" + str(train_loss))
        #     if abs(cur_loss - prev_loss) < 1 and e > min_stop:
        #         break
        #     prev_loss = train_loss
        #     if label is not None:
        #         try:
        #             cl, _ = clustering(myres, k=k, name='kmeans')
        #             dm = measure(cl, label)
        #             if (dm['ARI'] > bestARI):
        #                 bestNMI = dm['NMI']
        #                 bestARI = dm['ARI']
        #                 vae_.ae.save('bestmodel-' + prefix)
        #         except:
        #             print('Clustering error')
        #             # ErrorCount = ErrorCount + 1

        # if ErrorCount > 20:
        #     exit(9)

    ### analysis results
    # cluster_res = np.asarray( cluster_res )
    # points = np.asarray(points)
    aux_res = h5py.File(prefix + '_' + str(latent) + '_res.h5', mode='w')
    # aux_res.create_dataset(name='POINTS', data=points)
    # aux_res.create_dataset(name='LOSS', data=loss)
    aux_res.create_dataset(name='METHOD', data=bestmethod)
    aux_res.create_dataset(name='NMI', data=bestNMI)
    aux_res.create_dataset(name='ARI', data=bestARI)
    aux_res.create_dataset(name='RES' , data=bestres)
    aux_res.create_dataset(name='SIL', data=bestSil)
    aux_res.create_dataset(name='FINRES', data=myres)
    aux_res.create_dataset(name='LABEL', data=label)
    # count = 0
    # for r in res:
    #     aux_res.create_dataset(name='RES' + str(count), data=r)
    #     count += 1
    # count = 0
    # for r in bestres:
    #     aux_res.create_dataset(name='BESTRES' + str(count), data=r)
    #     count += 1
    # aux_res.close()

    myforesc = {'NMI': bestNMI, 'ARI': bestARI, 'res': bestres, 'method': bestmethod, 'finres': res[5],
                'corrected': res[15],
                'SILHOUETTE': bestSil}
    return myforesc


