"""Structural Agnostic Model.

Author: Diviyan Kalainathan, Olivier Goudet
Date: 09/3/2018
"""

import math
import numpy as np
import pandas as pd
import torch as th
from torch.autograd import Variable
from torch.utils.data import DataLoader
from time import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale


class CNormalized_Linear(th.nn.Module):
    """Linear layer with column-wise normalized input matrix."""

    def __init__(self, in_features, out_features, bias=False):
        """Initialize the layer."""
        super(CNormalized_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = th.nn.Parameter(th.Tensor(out_features, in_features))
        if bias:
            self.bias = th.nn.Parameter(th.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return th.nn.functional.linear(input, self.weight.div(self.weight.pow(2).sum(0).sqrt()))

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


class SAM_discriminator(th.nn.Module):
    def __init__(self, sizes, zero_components=[], **kwargs):
        super(SAM_discriminator, self).__init__()
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        self.sht = kwargs.get('shortcut', False)
        activation_function = kwargs.get('activation_function', th.nn.ReLU)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        dropout = kwargs.get("dropout", 0.)

        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(th.nn.Linear(i, j))
            if batch_norm:
                layers.append(th.nn.BatchNorm1d(j))
            if dropout != 0.:
                layers.append(th.nn.Dropout(p=dropout))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))

        layers.append(th.nn.Linear(sizes[-2], sizes[-1]))
        self.layers = th.nn.Sequential(*layers)
        # print(self.layers)

    def forward(self, x):
        return self.layers(x)


class SAM_block(th.nn.Module):
    """ SAM-Block consisting of parents,
    a generative network and the output variable
    """

    def __init__(self, sizes, zero_components=[], **kwargs):
        super(SAM_block, self).__init__()
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        self.sht = kwargs.get('shortcut', False)
        activation_function = kwargs.get('activation_function', th.nn.ReLU)
        activation_argument = kwargs.get('activation_argument', None)
        initzero = kwargs.get('initzero', False)
        batch_norm = kwargs.get("batch_norm", False)
        dropout = kwargs.get("dropout", 0.)
        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(CNormalized_Linear(i, j))
            if batch_norm:
                layers.append(th.nn.BatchNorm1d(j))
            if dropout != 0.:
                layers.append(th.nn.Dropout(p=dropout))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))

        layers.append(th.nn.Linear(sizes[-2], sizes[-1]))

        self.layers = th.nn.Sequential(*layers)
        if self.sht:
            self.shortcut = th.nn.Linear(sizes[0], sizes[-1])

        # Filtering the unconnected nodes.
        self._filter = th.ones(1, sizes[0])

        for i in zero_components:
            self._filter[:, i].zero_()
            if self.sht:
                self.shortcut.weight[:, i].data.zero_()

        # self.layers[0].weight.data.normal_()
        self._filter = Variable(self._filter, requires_grad=False)
        self.fs_filter = th.nn.Parameter(self._filter.data)
        if initzero:
            self.fs_filter.data[self.fs_filter.data != 0] = 0.0001

        if gpu:
            self._filter = self._filter.cuda(gpu_no)
        # print(self.layers)

    def forward(self, x):

        if self.sht:
            return self.layers(x * (self._filter *
                                    self.fs_filter).expand_as(x)) + \
                self.shortcut(x * (self._filter *
                                   self.fs_filter).expand_as(x))
        else:
            return self.layers(x * (self._filter *
                                    self.fs_filter).expand_as(x))


class SAM_generators(th.nn.Module):
    def __init__(self, data_shape, zero_components, nh=200, dnh=200, batch_size=-1, **kwargs):
        super(SAM_generators, self).__init__()
        if batch_size == -1:
            batch_size = data_shape[0]
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        rows, self.cols = data_shape

        # building the computation graph

        self.noise = [Variable(th.FloatTensor(batch_size, 1))
                      for i in range(self.cols)]
        if gpu:
            self.noise = [i.cuda(gpu_no) for i in self.noise]
        self.blocks = th.nn.ModuleList()

        # Init all the blocks
        for i in range(self.cols):
            self.blocks.append(SAM_block(
                [self.cols + 1, nh, 1], zero_components[i], **kwargs))

    def forward(self, x):
        for i in self.noise:
            i.data.normal_()

        self.generated_variables = [self.blocks[i](
            th.cat([x, self.noise[i]], 1)) for i in range(self.cols)]
        return self.generated_variables


def run_SAM(df_data, skeleton=None, **kwargs):
    print(kwargs)
    gpu = kwargs.get('gpu', False)
    gpu_no = kwargs.get('gpu_no', 0)

    train_epochs = kwargs.get('train_epochs', 1000)
    test_epochs = kwargs.get('test_epochs', 1000)
    batch_size = kwargs.get('batch_size', -1)

    learning_rate = kwargs.get('learning_rate', 0.1)
    lr_disc = kwargs.get('lr_disc', learning_rate)
    verbose = kwargs.get('verbose', True)
    regul_param = kwargs.get('regul_param', 0.1)
    asymfactor = kwargs.get('asymfactor', 'inputweights')
    plot = kwargs.get("plot", False)
    nh = kwargs.get('nh', 200)
    dnh = kwargs.get('dnh', None)
    shortcut = kwargs.get('shortcut', False)
    d_str = "Epoch: {} -- Disc: {} -- Gen: {} -- L1: {} -- ASYM: {}"
    list_nodes = list(df_data.columns)
    df_data = (df_data[list_nodes]).as_matrix()
    data = df_data.astype('float32')
    data = th.from_numpy(data)
    if batch_size == -1:
        batch_size = data.shape[0]
    rows, cols = data.size()

    # Get the list of indexes to ignore
    if skeleton is not None:
        connections = []
        for idx, i in enumerate(list_nodes):
            connections.append([list_nodes.index(j)
                                for j in skeleton.dict_nw()[i]])

        zero_components = [
            [i for i in range(cols) if i not in j] for j in connections]

    else:
        zero_components = [[i] for i in range(cols)]

    sam = SAM_generators((rows, cols), zero_components, batch_norm=True, **kwargs)

    # UGLY
    activation_function = kwargs.get('activation_function', th.nn.ReLU)
    try:
        del kwargs["activation_function"]
    except KeyError:
        pass
    discriminator_sam = SAM_discriminator(
        [cols, dnh, dnh, 1], batch_norm=True,
        activation_function=th.nn.LeakyReLU,
        activation_argument=0.2, **kwargs)
    # discriminator_sam = SAM_block(
    #  [cols, 50, 50, 1], **kwargs)
    kwargs["activation_function"] = activation_function
    # END of UGLY

    if gpu:
        sam = sam.cuda(gpu_no)
        discriminator_sam = discriminator_sam.cuda(gpu_no)
        data = data.cuda(gpu_no)

    data_list = [data[:, [i]] for i in range(cols)]

    # Select parameters to optimize : ignore the non connected nodes
    criterion = th.nn.BCEWithLogitsLoss()
    g_optimizer = th.optim.Adam(sam.parameters(), lr=learning_rate)
    d_optimizer = th.optim.Adam(
        discriminator_sam.parameters(), lr=lr_disc)

    # Printout value
    gen = []
    true_variable = Variable(
        th.ones(batch_size, 1), requires_grad=False)
    false_variable = Variable(
        th.zeros(batch_size, 1), requires_grad=False)
    output_weights = th.zeros(data.shape[1], data.shape[1])

    def fill_nan(mat):
        mat[mat.ne(mat)] = 0
        return mat

    if gpu:
        true_variable = true_variable.cuda(gpu_no)
        false_variable = false_variable.cuda(gpu_no)
        output_weights = output_weights.cuda(gpu_no)

    # print('Init : ', time() - ac)
    data_iterator = DataLoader(data, batch_size=batch_size, shuffle=True)
    # for i in range(cols):
    #     vector_iterators[i].set_global(
    #         batch_size=batch_size, size=data.data.shape[0])

    # TRAIN
    for epoch in range(train_epochs + test_epochs):
        for i_batch, batch in enumerate(data_iterator):
            if asymfactor == 'grad' or asymfactor == 'gradloss':
                batch = Variable(batch, requires_grad=True)

            else:
                batch = Variable(batch)
            batch_vectors = [batch[:, [i]] for i in range(cols)]
            ac = time()

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # Train the discriminator

            generated_variables = sam(batch)
            if verbose and epoch % 200 == 0 and i_batch == 0:
                print("Init : {}".format(time() - ac))
                ac = time()
            disc_losses = []
            gen_losses = []

            for i in range(cols):
                # print([type(v) for c in [batch_vectors[:i], [
                #        generated_variables[i]],
                #        batch_vectors[i + 1: ]] for v in c])
                generator_output = th.cat([v for c in [batch_vectors[: i], [
                    generated_variables[i]],
                    batch_vectors[i + 1:]] for v in c], 1)
                # 1. Train discriminator on fake
                disc_output_detached = discriminator_sam(
                    generator_output.detach())
                disc_output = discriminator_sam(generator_output)
                disc_losses.append(
                    criterion(disc_output_detached, false_variable))

                # 2. Train the generator :

                gen_losses.append(criterion(disc_output, true_variable))

            if verbose and epoch % 200 == 0 and i_batch == 0:
                print('Gen/Disc loss computation : ', time() - ac)
                ac = time()

            true_output = discriminator_sam(batch)

            # adv_loss = sum(disc_losses) / cols + \
            #     criterion(true_output, true_variable)
            adv_loss = sum(disc_losses) + \
                criterion(true_output, true_variable) * cols

            gen_loss = sum(gen_losses)

            adv_loss.backward()
            d_optimizer.step()

            if epoch % 200 == 0 and i_batch == 0:
                print('Disc optimizer : ', time() - ac)
                ac = time()

            asymmetry_factors = th.stack(
                [i.fs_filter[0, :-1].abs() for i in sam.blocks], 1)

            l1_reg = regul_param * asymmetry_factors.sum()

            # l2_reg = input_regul_param * \
            #     sum([i.layers[0].weight.pow(2).sum() for i in sam.blocks])

            loss = gen_loss + l1_reg

            if verbose and epoch % 200 == 0 and i_batch == 0:

                print(str(i) + " " + d_str.format(epoch,
                                                  adv_loss.cpu().data[0],
                                                  gen_loss.cpu(
                                                  ).data[0] / cols,
                                                  l1_reg.cpu().data[0], 0))
            loss.backward()

            if epoch % 200 == 0 and i_batch == 0:
                print('Gen backward : ', time() - ac)
                ac = time()

            # STORE ASSYMETRY values for output
            if epoch > train_epochs:
                output_weights.add_(asymmetry_factors.data)

            g_optimizer.step()

            if epoch % 200 == 0 and i_batch == 0:
                print('Gen optimizer : ', time() - ac)
                ac = time()

            if plot and i_batch == 0:
                try:
                    ax.clear()
                    ax.plot(range(len(adv_plt)), adv_plt, "r-",
                            linewidth=1.5, markersize=4,
                            label="Discriminator")
                    ax.plot(range(len(adv_plt)), gen_plt, "g-", linewidth=1.5,
                            markersize=4, label="Generators")
                    ax.plot(range(len(adv_plt)), l1_plt, "b-",
                            linewidth=1.5, markersize=4,
                            label="L1-Regularization")
                    ax.plot(range(len(adv_plt)), asym_plt, "c-",
                            linewidth=1.5, markersize=4,
                            label="Assym penalization")

                    plt.legend()

                    adv_plt.append(adv_loss.cpu().data[0])
                    gen_plt.append(gen_loss.cpu().data[0] / cols)
                    l1_plt.append(l1_reg.cpu().data[0])
                    asym_plt.append(asymmetry_reg.cpu().data[0])
                    plt.pause(0.0001)

                except NameError:
                    plt.ion()
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    plt.xlabel("Epoch")
                    plt.ylabel("Losses")

                    plt.pause(0.0001)

                    adv_plt = [adv_loss.cpu().data[0]]
                    gen_plt = [gen_loss.cpu().data[0] / cols]
                    l1_plt = [l1_reg.cpu().data[0]]

            elif plot:
                adv_plt.append(adv_loss.cpu().data[0])
                gen_plt.append(gen_loss.cpu().data[0] / cols)
                l1_plt.append(l1_reg.cpu().data[0])

            # if epoch % 200 == 0:
            #     print('Plt : ', time() - ac)
            #     ac = time()
            #     if epoch % 200 == 0:
            #         if epoch ==0:
            #             plt.ion()
            #         # gen.append([gene.data for gene in generated_variables])
            #         to_print = [[0, 1]]  # , [1, 0]]  # [2, 3]]  # , [11, 17]]
            #         plt.clf()
            #         for (i, j) in to_print:
            #             #plt.figure()
            #             plt.scatter(generated_variables[i].data.cpu().numpy(
            #             ), batch.data.cpu().numpy()[:, j], label="Y -> X")
            #             plt.scatter(batch.data.cpu().numpy()[
            #                 :, i], generated_variables[j].data.cpu().numpy(), label="X -> Y")
            #
            #             plt.scatter(batch.data.cpu().numpy()[:, i], batch.data.cpu().numpy()[
            #                 :, j], label="original data")
            #             plt.legend()
            #
            #             # plt.savefig("test_SAM_adv/fig/epoch_" + str(epoch))
            #         plt.pause(0.01)
            #         #plt.show()
            #         #plt.close()

    # if gpu:
        # print(generated_variables)

        # ]
    return output_weights.div_(test_epochs).cpu().numpy()

    # else:
    #     return [[[l.numpy() for l in j] for j in gen],
    #     output_weights.div_(test_epochs).numpy()]


if __name__ == "__main__":
    import cdt_private
    from cdt_private.utils.metrics import precision_recall
    import pandas as pd
    import sys
    import numpy as np
    from itertools import product
    from joblib import Parallel, delayed
    import torch as th
    import argparse
    import datetime

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data', metavar='d', type=str, help='data')
    parser.add_argument('--skel', metavar='s', type=str,
                        help='skeleton', default=None)
    parser.add_argument('--target', metavar='t', type=str,
                        help='Target file for score computation', default=None)
    parser.add_argument('--train', metavar='tr', type=int,
                        help="num of train epochs", default=1000)
    parser.add_argument('--regul', metavar='r', type=int,
                        help="num of regul epochs", default=0)
    parser.add_argument('--test', metavar='te', type=int,
                        help="num of test epochs", default=1000)
    parser.add_argument('--nruns', metavar='n', type=int,
                        help="num of runs", default=1)
    parser.add_argument('--batchnorm', help="batchnorm", action='store_true')
    parser.add_argument('--dropout', type=float, help="dropout", default=0.)
    parser.add_argument('--batch', metavar='b', type=int,
                        help="batchsize", default=-1)
    parser.add_argument('--njobs', metavar='j', type=int,
                        help="num of jobs", default=-1)
    parser.add_argument('--gpu', help="Use gpu", action='store_true')
    parser.add_argument('--plot', help="Plot losses", action='store_true')
    parser.add_argument('--csv', help="CSV file", action='store_true')
    parser.add_argument('--nv', help="No verbose", action='store_false')
    parser.add_argument('--log', metavar='l', type=str,
                        help='Specify a custom log folder', default=".")

    args = parser.parse_args()
    print(args)
    if args.csv:
        sep = ","
    else:
        sep = "\t"
    data = pd.read_csv(args.data, sep=sep)

    if args.skel is not None:
        skel = pd.read_csv(args.skel, sep=sep)
        skel = cdt_private.UndirectedGraph(skel)
    else:
        skel = None

    if args.target is not None:
        target = pd.read_csv(args.target, sep=sep)
        target = cdt_private.DirectedGraph(target)
    else:
        target = None

    print(cdt_private.SETTINGS.GPU_LIST)
    # '1500', '2000', 'res', 'res2', 'res3'
    out_freq = ['res']  # '0', '500', '1000', '1500',
    cdt_private.SETTINGS.GPU = args.gpu
    if not cdt_private.SETTINGS.GPU:
        cdt_private.SETTINGS.GPU_LIST = [0]
        print("Forcing gpu list to 0")

    if args.njobs != -1:
        cdt_private.SETTINGS.NB_JOBS = args.njobs
    elif args.gpu:
        cdt_private.SETTINGS.NB_JOBS = len(cdt_private.SETTINGS.GPU_LIST)
    else:
        cdt_private.SETTINGS.NB_JOBS = 1
    # with torch cpu it is wiser to not parallel computation
    # it parallelizes automatically

    activation_function = th.nn.Tanh  # [th.nn.ReLU, th.nn.Tanh, th.nn.Sigmoid]
    learning_rate = 0.1  # , 0.005, 0.001]

    type_regul = "l1"
    regul = .05  # [0.005, 0.001, 0.0005, 0.0001, 0.00005]
    # [0.1, 0.05, 0.01, 0.005, 0.001, 'ct1', 'ct2', 'ct3']
    asymmetry = 0.1
    shortcut = False  # , True]
    bandwidth_mmd = 0
    n_h = 100  # [20, 30, 40, 50]
    # ['inputweights', 'grad', 'gradloss', 'hybrid_elm', 'elm']
    asymfactor = 'sqrt'
    # use_dif = True
    loss = "adv"

    mat = np.zeros((len(list(data.columns)), len(list(data.columns))))

    def exec_sam_instance(gpuno):
        return run_SAM(data, skeleton=skel, learning_rate=learning_rate,
                       type_regul=type_regul,
                       regul_param=regul, asymmetry_param=asymmetry,
                       shortcut=shortcut,
                       nh=n_h, gpu=args.gpu,
                       gpu_no=gpuno, train_epochs=args.train,
                       test_epochs=args.test,
                       asymfactor=asymfactor, batch_size=args.batch,
                       plot=args.plot, activation_function=activation_function,
                       verbose=args.nv, regul_epochs=args.regul, loss=loss, dropout=args.dropout)

    list_out = Parallel(n_jobs=cdt_private.SETTINGS.NB_JOBS)(delayed(exec_sam_instance)(
        idx % len(cdt_private.SETTINGS.GPU_LIST)) for idx in range(args.nruns))

    sys.stdout.flush()
    sys.stderr.flush()

    W = list_out[0]
    for w in list_out[1:]:
        W += w
    W /= args.nruns
    print(W)
    np.savetxt('{}/mat_SAM_{}_act-{}_lr-{}_tr-{}_l1-{}_l2-{}_alpha-{}_sht-{}_nh-{}_af-{}_e-{}.csv'.format(args.log, (args.data.split('/')[-1]).split('.')[0], str(activation_function).split(
        '.')[-1].split("'")[0], activation_function, learning_rate, type_regul, regul, asymmetry, bandwidth_mmd, shortcut, n_h, asymfactor, 'res'), W, delimiter=",")
    # W_max = W - W.transpose()
    # W_max[W_max < 0] = 0
    W_max = np.zeros((W.shape[0], W.shape[1]))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if(W[i][j] > W[j][i]):
                W_max[i][j] = W[i][j]
    # print(W_max)

    g = cdt_private.DirectedGraph(pd.DataFrame(
        W_max, columns=data.columns), adjacency_matrix=True)
    ll = g.list_edges()
    pd.DataFrame(ll, columns=['Cause', 'Effect', 'Weight']).to_csv(
        '{}/adv_SAM_{}_act-{}_lr-{}_tr-{}_l1-{}_l2-{}_alpha-{}_sht-{}_nh-{}_af-{}_e-{}.csv'.format(args.log, (args.data.split('/')[-1]).split('.')[0], str(activation_function).split('.')[-1].split("'")[0], activation_function, learning_rate, type_regul, regul, asymmetry, bandwidth_mmd, shortcut, n_h, asymfactor, 'res'), index=False)

    if target is not None:
        aupr_scores = [precision_recall(g, target)[0][0]]

        report = pd.DataFrame([[activation_function,
                                learning_rate,
                                type_regul,
                                regul,
                                asymmetry,
                                bandwidth_mmd,
                                shortcut,
                                n_h,
                                asymfactor]],
                              columns=["activation_function",
                                       "learning_rate",
                                       "type_regul",
                                       "regul",
                                       "asymmetry",
                                       "bandwidth_mmd",
                                       "shortcut",
                                       "n_h",
                                       "asymfactor"])

        report["AUPR"] = aupr_scores
        report.to_csv("{}/sam-report-{}-{}.csv".format(args.log, args.data.split('/')
                                                       [-1].split('.')[0], datetime.datetime.now().isoformat()), index=False)

        print(aupr_scores)
