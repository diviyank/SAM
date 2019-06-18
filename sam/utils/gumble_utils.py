"""Gumble softmax."""
import torch as th

import torch.distributions.relaxed_bernoulli as relaxed_bernoulli
from torch.distributions.transformed_distribution import  TransformedDistribution
from torch.distributions.transforms import SigmoidTransform,AffineTransform

from torch.distributions.uniform import Uniform

def _sample_gumbel(shape, eps=1e-10, out=None):
    """
    Implementation of pytorch.
    (https://github.com/pytorch/pytorch/blob/e4eee7c2cf43f4edba7a14687ad59d3ed61d9833/torch/nn/functional.py)
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else th.rand(shape)
    return - th.log(eps - th.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Implementation of pytorch.
    (https://github.com/pytorch/pytorch/blob/e4eee7c2cf43f4edba7a14687ad59d3ed61d9833/torch/nn/functional.py)
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = logits + gumbel_noise
    return th.softmax(y / tau, dims-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    Implementation of pytorch.
    (https://github.com/pytorch/pytorch/blob/e4eee7c2cf43f4edba7a14687ad59d3ed61d9833/torch/nn/functional.py)
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: `[batch_size, n_class]` unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if ``True``, take `argmax`, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.data + y_soft
    else:
        y = y_soft
    return y


def _sample_logistic(shape, out=None):

    U = out.resize_(shape).uniform_() if out is not None else th.rand(shape)
    #U2 = out.resize_(shape).uniform_() if out is not None else th.rand(shape)

    return th.log(U) - th.log(1-U)


def _sigmoid_sample(logits, tau=1):
    """
    Implementation of Bernouilli reparametrization based on Maddison et al. 2017
    """
    dims = logits.dim()
    logistic_noise = _sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return th.sigmoid(y / tau)


def gumbel_sigmoid(logits, ones_tensor, zeros_tensor, tau=1, hard=False):

    shape = logits.size()
 
    y_soft = _sigmoid_sample(logits, tau=tau)

    if hard:

        y_hard = th.where(y_soft > 0.5, ones_tensor, zeros_tensor)

        y = y_hard.data - y_soft.data + y_soft

    else:
    	y = y_soft

    return y



# "Utilisation de https://pytorch.org/docs/stable/_modules/torch/distributions/relaxed_bernoulli.html"
# def gumbel_sigmoid_v1(logits, tau=1, hard=False, eps=1e-10):
#
#     shape = logits.size()
#
#     y_soft = relaxed_bernoulli.RelaxedBernoulli(tau,logits=2*logits).sample()
#
#
#
#     if hard:
#         _, k = y_soft.data.max(-1)
#         # this bit is based on
#         # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
#         y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
#         # this cool bit of code achieves two things:
#         # - makes the output value exactly one-hot (since we add then
#         #   subtract y_soft value)
#         # - makes the gradient equal to y_soft gradient (since we strip
#         #   all other gradients)
#         y = y_hard - y_soft.data + y_soft
#     else:
#         y = y_soft
#     return y
#
#
# "Implementation d'après http://edwardlib.org/api/ed/models/RelaxedBernoulli"
# def gumbel_sigmoid_v2(logits, tau=1, hard=False, eps=1e-10):
#
#     shape = logits.size()
#
#     base_distribution = Uniform(0, 1)
#     transforms = [SigmoidTransform().inv, AffineTransform(loc=2*logits/tau, scale=1./tau)]
#     logistic = TransformedDistribution(base_distribution, transforms)
#
#     y_soft = th.sigmoid(logistic.sample())
#
#     if hard:
#         _, k = y_soft.data.max(-1)
#         # this bit is based on
#         # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
#         y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
#         # this cool bit of code achieves two things:
#         # - makes the output value exactly one-hot (since we add then
#         #   subtract y_soft value)
#         # - makes the gradient equal to y_soft gradient (since we strip
#         #   all other gradients)
#         y = y_hard - y_soft.data + y_soft
#     else:
#         y = y_soft
#     return y


if __name__ == "__main__":
    logits = th.tensor([0.1,0.2,0.3,2])
    stacked_logit = th.stack([logits, -logits], 1)

    probability = th.sigmoid(2*logits)

    print("probability")
    print(probability)
    print(th.softmax(stacked_logit, dim =1))


    print("Test gumble softmax")
    output_gumble_softmax = th.zeros(stacked_logit.shape)

    nbrun = 100000

    for i in range(0,nbrun):
        test_gumble_softmax = gumbel_softmax(stacked_logit,eps=0)
        output_gumble_softmax.add_(test_gumble_softmax)

    print(output_gumble_softmax.div_(nbrun))


    print("Test gumble sigmoid v1")
    output_gumble_sigmoid_v1 = th.zeros(logits.shape)

    for i in range(0,100000):
        test_gumble_sigmoid = gumbel_sigmoid_v1(logits)
        output_gumble_sigmoid_v1.add_(test_gumble_sigmoid)

    print(output_gumble_sigmoid_v1.div_(nbrun))



    print("Test gumble sigmoid v2")
    output_gumble_sigmoid_v2 = th.zeros(logits.shape)

    for i in range(0,100000):
        test_gumble_sigmoid = gumbel_sigmoid_v2(logits)
        output_gumble_sigmoid_v2.add_(test_gumble_sigmoid)

    print(output_gumble_sigmoid_v2.div_(nbrun))
