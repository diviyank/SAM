"""Modification in the implementation of batchnorm.

In order to have the same comportment when 2D or 3D.
Author:Diviyan Kalainathan
"""

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class ChannelBatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """
    def __init__(self, num_channels, num_features, *args, **kwargs):
        super(ChannelBatchNorm1d, self).__init__(num_channels*num_features, *args, **kwargs)
        self.num_channels = num_channels
        self.num_features = num_features

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        _input = input.contiguous().view(-1, self.num_channels * self.num_features)
        output = super(ChannelBatchNorm1d, self).forward(_input)
        return output.view(-1, self.num_channels, self.num_features)
