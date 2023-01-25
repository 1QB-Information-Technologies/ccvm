import numpy as np
import torch


class BoxQP(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.params = torch.nn.Parameter(c)

    def forward(self, q_mat, c_vector):
        """TODO: add docstrings
        :param q_mat: coefficients of the quadratic terms
        :type Tensor
        :param c_vector: coefficients of the linear terms
        :type Tensor
        :return:
        :rtype: Tensor
        """
        c_variables = self.params
        return func_post_torch(c_variables, q_mat, c_vector)


def func_post(c, *args):
    """TODO: add docstrings

    :param c:
    :type Tensor
    :return:
    :rtype: Tensor
    """
    q_mat = np.array(args[0].cpu())
    c_vector = np.array(args[1].cpu())
    energy1 = np.einsum("i, ij, j", c, q_mat, c)
    energy2 = np.einsum("i, i", c, c_vector)
    return 0.5 * energy1 + energy2


def func_post_jac(c, *args):
    """TODO: add docstrings

    :param c:
    :type Tensor
    :return:
    :rtype: Tensor
    """
    q_mat = np.array(args[0].cpu())
    c_vector = np.array(args[1].cpu())
    energy1_jac = np.einsum("ij,j->i", q_mat, c)
    energy2_jac = c_vector
    return energy1_jac + energy2_jac


def func_post_hess(c, *args):
    """TODO: add docstrings

    :param c:
    :type Tensor
    :return:
    :rtype: Tensor
    """
    q_mat = np.array(args[0].cpu())
    return 0.5 * q_mat


def func_post_LBFGS(c, q_mat, c_vector):
    """TODO: add docstrings

    :param c:
    :type Tensor
    :param q_mat: coefficients of the quadratic terms
    :type Tensor
    :param c_vector: coefficients of the linear terms
    :type Tensor
    :return:
    :rtype: Tensor
    """
    energy1 = torch.einsum("i, ij, j", c, q_mat, c)
    energy2 = torch.einsum("i, i", c, c_vector)
    return 0.5 * energy1 + energy2


def func_post_torch(c, q_mat, c_vector):
    """TODO: add docstrings

    :param c:
    :type Tensor
    :param q_mat: coefficients of the quadratic terms
    :type Tensor
    :param c_vector: coefficients of the linear terms
    :type Tensor
    :return:
    :rtype: Tensor
    """
    energy1 = torch.einsum("bi, ij, bj -> b", c, q_mat, c)
    energy2 = torch.einsum("bi, i -> b", c, c_vector)
    return 0.5 * energy1 + energy2
