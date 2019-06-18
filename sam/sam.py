"""Structural Agnostic Model.

Author: Diviyan Kalainathan, Olivier Goudet
Date: 09/3/2018
"""
import os
import numpy as np
import torch as th
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import scale
from .utils.linear3d import Linear3D
from .utils.graph import MatrixSampler, notears_constr
from .utils.batchnorm import ChannelBatchNorm1d
from .utils.parlib import parallel_run


class SAM_generators(th.nn.Module):
    """Ensemble of all the generators."""

    def __init__(self, data_shape, nh, skeleton=None, cat_sizes=None, linear=False):
        """Init the model."""
        super(SAM_generators, self).__init__()
        layers = []
        # Building skeleton
        self.sizes = cat_sizes
        self.linear = linear

        nb_vars = data_shape[1]
        self.nb_vars = nb_vars
        if skeleton is None:
            skeleton = 1 - th.eye(nb_vars + 1, nb_vars)  # 1 row for noise
        else:
            skeleton = th.cat([th.Tensor(skeleton), th.ones(1, nb_vars)], 1)
        if linear:
            self.input_layer = Linear3D((nb_vars, nb_vars + 1, 1))
        else:
            self.input_layer = Linear3D((nb_vars, nb_vars + 1, nh))
            layers.append(ChannelBatchNorm1d(nb_vars, nh))
            layers.append(th.nn.Tanh())
            self.output_layer = Linear3D((nb_vars, nh, 1))

        self.layers = th.nn.Sequential(*layers)
        self.register_buffer('skeleton', skeleton)

    def forward(self, data, noise, adj_matrix, drawn_neurons=None):
        """Forward through all the generators."""

        if self.linear:
            output = self.input_layer(data, noise, adj_matrix * self.skeleton)
        else:
            output = self.output_layer(self.layers(self.input_layer(data,
                                                                    noise,
                                                                    adj_matrix * self.skeleton)),
                                       drawn_neurons)

        return output.squeeze(2)

    def reset_parameters(self):
        if not self.linear:
            self.output_layer.reset_parameters()
            for layer in self.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        self.input_layer.reset_parameters()


class SAM_discriminator(th.nn.Module):
    """SAM discriminator."""

    def __init__(self, nfeatures, dnh, hlayers=2, mask=None):
        super(SAM_discriminator, self).__init__()
        self.nfeatures = nfeatures
        layers = []
        layers.append(th.nn.Linear(nfeatures, dnh))
        layers.append(th.nn.BatchNorm1d(dnh))
        layers.append(th.nn.LeakyReLU(.2))
        for i in range(hlayers-1):
            layers.append(th.nn.Linear(dnh, dnh))
            layers.append(th.nn.BatchNorm1d(dnh))
            layers.append(th.nn.LeakyReLU(.2))

        layers.append(th.nn.Linear(dnh, 1))
        self.layers = th.nn.Sequential(*layers)

        if mask is None:
            mask = th.eye(nfeatures, nfeatures)
        self.register_buffer("mask", mask.unsqueeze(0))

    def forward(self, input, obs_data=None):
        if obs_data is not None:
            return [self.layers(i) for i in th.unbind(obs_data.unsqueeze(1) * (1 - self.mask)
                                                      + input.unsqueeze(1) * self.mask, 1)]
        else:
            return self.layers(input)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


def run_SAM(in_data, skeleton=None, device="cpu",
            train=5000, test=1000,
            batch_size=-1, lr_gen=.001,
            lr_disc=.01, lambda1=0.001, lambda2=0.0000001, nh=None, dnh=None,
            verbose=True, losstype="fgan",
            dagstart=0, dagloss=False,
            dagpenalization=0.05, dagpenalization_increase=0.0,
            dag_threshold=0.5,
            linear=False, hlayers=2):

    list_nodes = list(in_data.columns)
    data = scale(in_data[list_nodes].values)
    nb_var = len(list_nodes)
    data = data.astype('float32')
    data = th.from_numpy(data).to(device)
    if batch_size == -1:
        batch_size = data.shape[0]
    rows, cols = data.size()
    # Get the list of indexes to ignore
    if skeleton is not None:
        skeleton = th.from_numpy(skeleton.astype('float32'))

    sam = SAM_generators((batch_size, cols), nh, skeleton=skeleton,
                         linear=linear).to(device)

    sam.reset_parameters()
    g_optimizer = th.optim.Adam(list(sam.parameters()), lr=lr_gen)

    if losstype != "mse":
        discriminator = SAM_discriminator(cols, dnh, hlayers).to(device)
        discriminator.reset_parameters()
        d_optimizer = th.optim.Adam(discriminator.parameters(), lr=lr_disc)
        criterion = th.nn.BCEWithLogitsLoss()
    else:
        criterion = th.nn.MSELoss()
        disc_loss = th.zeros(1)

    graph_sampler = MatrixSampler(nb_var, mask=skeleton,
                                  gumble=False).to(device)
    graph_sampler.weights.data.fill_(2)
    graph_optimizer = th.optim.Adam(graph_sampler.parameters(), lr=lr_gen)

    if not linear:
        neuron_sampler = MatrixSampler((nh, nb_var), mask=False,
                                       gumble=True).to(device)
        neuron_optimizer = th.optim.Adam(list(neuron_sampler.parameters()),
                                         lr=lr_gen)

    _true = th.ones(1).to(device)
    _false = th.zeros(1).to(device)
    output = th.zeros(nb_var, nb_var).to(device)

    noise = th.randn(batch_size, nb_var).to(device)
    noise_row = th.ones(1, nb_var).to(device)
    data_iterator = DataLoader(data, batch_size=batch_size,
                               shuffle=True, drop_last=True)

    # RUN
    if verbose:
        pbar = tqdm(range(train + test))
    else:
        pbar = range(train+test)
    for epoch in pbar:
        for i_batch, batch in enumerate(data_iterator):
            g_optimizer.zero_grad()
            graph_optimizer.zero_grad()

            if losstype != "mse":
                d_optimizer.zero_grad()

            if not linear:
                neuron_optimizer.zero_grad()

            # Train the discriminator

            if not epoch > train:
                drawn_graph = graph_sampler()
                if not linear:
                    drawn_neurons = neuron_sampler()
                else:
                    drawn_neurons = None
            noise.normal_()
            generated_variables = sam(batch, noise,
                                      th.cat([drawn_graph, noise_row], 0),
                                      drawn_neurons)

            if losstype == "mse":
                gen_loss = criterion(generated_variables, batch)
            else:
                disc_vars_d = discriminator(generated_variables.detach(), batch)
                disc_vars_g = discriminator(generated_variables, batch)
                true_vars_disc = discriminator(batch)

                if losstype == "gan":
                    disc_loss = sum([criterion(gen, _false.expand_as(gen)) for gen in disc_vars_d]) / nb_var \
                                     + criterion(true_vars_disc, _true.expand_as(true_vars_disc))
                    # Gen Losses per generator: multiply py the number of channels
                    gen_loss = sum([criterion(gen,
                                              _true.expand_as(gen))
                                    for gen in disc_vars_g])
                elif losstype == "fgan":

                    disc_loss = sum([th.mean(th.exp(gen - 1)) for gen in disc_vars_d]) / nb_var - th.mean(true_vars_disc)
                    gen_loss = -sum([th.mean(th.exp(gen - 1)) for gen in disc_vars_g])

                disc_loss.backward()
                d_optimizer.step()

            filters = graph_sampler.get_proba()

            struc_loss = lambda1*drawn_graph.sum()

            func_loss = 0 if linear else lambda2*drawn_neurons.sum()
            regul_loss = struc_loss + func_loss

            if dagloss and epoch > train * dagstart:
                dag_constraint = notears_constr(filters*filters)
                loss = gen_loss + regul_loss + (dagpenalization +
                                                (epoch - train * dagstart)
                                                * dagpenalization_increase) * dag_constraint
            else:
                loss = gen_loss + regul_loss
            if verbose and epoch % 20 == 0 and i_batch == 0:
                pbar.set_postfix(gen=gen_loss.item()/cols,
                                 disc=disc_loss.item(),
                                 regul_loss=regul_loss.item(),
                                 tot=loss.item())

            if epoch < train + test - 1:
                loss.backward(retain_graph=True)
            
            if epoch >= train:
                output.add_(filters.data)

            g_optimizer.step()
            graph_optimizer.step()
            if not linear:
                neuron_optimizer.step()

    return output.div_(test).cpu().numpy()


# def exec_sam_instance(data, skeleton=None, gpus=0,
#                       device='cpu', verbose=True, log=None,
#                       lr=0.001, dlr=0.01, lambda1=0.001, lambda2=0.0000001, nh=200, dnh=500,
#                       train=10000, test=1000, batchsize=-1,
#                       losstype="fgan", dagstart=0, dagloss=False,
#                       dagpenalization=0.001, dagpenalization_increase=0.0,
#                       dag_threshold=0.5, linear=False, hlayers=2):
#         out = run_SAM(data, skeleton=skeleton,
#                       device=device,lr_gen=lr, lr_disc=dlr,
#                       lambda1=lambda1, lambda2=lambda2,
#                       nh=nh, dnh=dnh,
#                       train=train,
#                       test=test, batch_size=batchsize,
#                       dagstart=dagstart,
#                       dagloss=dagloss,
#                       dagpenalization=dagpenalization,
#                       dagpenalization_increase=dagpenalization_increase,
#                       losstype=losstype,
#                       dag_threshold=dag_threshold,
#                       linear=linear,
#                       hlayers=hlayers
#                       )
#         if log is not None:
#             np.savetxt(log, out, delimiter=",")
#         return out


class SAM(object):
    """Structural Agnostic Model."""

    def __init__(self, lr=0.01, dlr=0.01, lambda1=0.01, lambda2=0.00001, nh=200, dnh=200,
                 train_epochs=10000, test_epochs=1000, batchsize=-1,
                 losstype="fgan", dagstart=0.5, dagloss=True, dagpenalization=0,
                 dagpenalization_increase=0.001, linear=False, hlayers=2):

        """Init and parametrize the SAM model.

        :param lr: Learning rate of the generators
        :param dlr: Learning rate of the discriminator
        :param l1: L1 penalization on the causal filters
        :param nh: Number of hidden units in the generators' hidden layers
           a
           ((cols,cols)
        :param dnh: Number of hidden units in the discriminator's hidden layer$
        :param train_epochs: Number of training epochs
        :param test_epochs: Number of test epochs (saving and averaging the causal filters)
        :param batchsize: Size of the batches to be fed to the SAM model.
        """
        super(SAM, self).__init__()
        self.lr = lr
        self.dlr = dlr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.nh = nh
        self.dnh = dnh
        self.train = train_epochs
        self.test = test_epochs
        self.batchsize = batchsize
        self.losstype = losstype
        self.dagstart = dagstart
        self.dagloss = dagloss
        self.dagpenalization = dagpenalization
        self.dagpenalization_increase = dagpenalization_increase
        self.linear = linear
        self.hlayers = hlayers

    def predict(self, data, skeleton=None, mixed_data=False, nruns=6, njobs=1,
                gpus=0, verbose=True, log=None):
        """Execute SAM on a dataset given a skeleton or not.

        :param data: Observational data for estimation of causal relationships by SAM
        :param skeleton: A priori knowledge about the causal relationships as an adjacency matrix.
                         Can be fed either directed or undirected links.
        :param nruns: Number of runs to be made for causal estimation.
                      Recommended: >5 for optimal performance.
        :param njobs: Numbers of jobs to be run in Parallel.
                      Recommended: 1 if no GPU available, 2*number of GPUs else.
        :param gpus: Number of available GPUs for the algorithm.
        :param verbose: verbose mode
        :param plot: Plot losses interactively. Not recommended if nruns>1
        :param plot_generated_pair: plots a generated pair interactively.  Not recommended if nruns>1
        :return: Adjacency matrix (A) of the graph estimated by SAM,
                A[i,j] is the term of the ith variable for the jth generator.
        """
        assert nruns > 0
        if nruns == 1:
            return run_SAM(data, skeleton=skeleton,
                           lr_gen=self.lr,
                           lr_disc=self.dlr,
                           verbose=verbose,
                           lambda1=self.lambda1, lambda2=self.lambda2,
                           nh=self.nh, dnh=self.dnh,
                           train=self.train,
                           test=self.test, batch_size=self.batchsize,
                           dagstart=self.dagstart,
                           dagloss=self.dagloss,
                           dagpenalization=self.dagpenalization,
                           dagpenalization_increase=self.dagpenalization_increase,
                           losstype=self.losstype,
                           dag_threshold=self.dag_threshold,
                           linear=self.linear,
                           hlayers=self.hlayers,
                           device='cuda:0' if gpus else 'cpu')
        else:
            list_out = []
            if log is not None:
                idx = 0
                while os.path.isfile(log + str(idx)):
                    list_out.append(np.loadtxt(log + str(idx), delimiter=","))
                    idx += 1
            results = parallel_run(run_SAM, data, skeleton=skeleton,
                                   nruns=nruns-len(list_out),
                                   njobs=njobs, gpus=gpus, lr_gen=self.lr,
                                   lr_disc=self.dlr,
                                   verbose=verbose,
                                   lambda1=self.lambda1, lambda2=self.lambda2,
                                   nh=self.nh, dnh=self.dnh,
                                   train=self.train,
                                   test=self.test, batch_size=self.batchsize,
                                   dagstart=self.dagstart,
                                   dagloss=self.dagloss,
                                   dagpenalization=self.dagpenalization,
                                   dagpenalization_increase=self.dagpenalization_increase,
                                   losstype=self.losstype,
                                   linear=self.linear,
                                   hlayers=self.hlayers)
            list_out.extend(results)
            list_out = [i for i in list_out if not np.isnan(i).any()]
            try:
                assert len(list_out) > 0
            except AssertionError as e:
                print("All solutions contain NaNs")
                raise(e)
            return sum(list_out)/len(list_out)
