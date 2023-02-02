import numpy as np
import torch
import torch.distributions as tdist
import time
from random import randint
import matplotlib.pyplot as plt
import pickle
import random
import tqdm


def get_w(i, iter_n):
    w1 = w_dist1.sample((N,)).transpose(0, 1)
    w2 = w_dist2.sample((N,)).transpose(0, 1)
    if i > 0.2 * iter_n:
        if i > 0.6 * iter_n:
            return w1 * 0
        else:
            return w1 * 1
    else:
        return w2


def calculate_grads(c, q_matrix, c_vector, p, rate):
    """we treat the SDE that simulates the CIM of NTT as gradient calculation.
    Original SDE considers only quadratic part of the objective function.
    Therefore, we need to modify and add linear part of the QP to the SDE.

    Args:
        c (torch.Tensor): amplitudes
        q_matrix (torch.Tensor): coefficients of the quadratic terms
        c_vector (torch.Tensor): coefficients of the linear terms
        my_al (torch.Tensor): lagrangean multiplier due to relaxation of box constraints

    Returns:
        torch.Tensor: grads
    """

    c_pow = torch.pow(c, 2)
    grad_1 = torch.einsum("bi,ij -> bj", c_pow, q_matrix)
    grad_2 = torch.einsum("cj,cj -> cj", -1 + (p * rate) - c_pow, c)
    grad_3 = torch.einsum("j,cj -> cj", c_vector, c)

    return -2 * grad_1 * c + grad_2 - 2 * grad_3


def compute_energy(confs, q_mat, c_vector, q_scale, c_scale, lamb):
    """
    Compute energy of configuration by xJx + hx formula
    Args:
    confs: Configurations for which to compute energy

    Returns: Energy of configurations

    """
    confs_pow = confs.pow(2)
    energy1 = torch.einsum("bi, ij, bj -> b", confs_pow, q_mat, confs_pow) * q_scale
    energy2 = torch.einsum("bi, i -> b", confs_pow, c_vector) * c_scale
    return 0.5 * energy1 + energy2
    # return energy1


def solve(
    n, q_mat, c_vector, q_scale, c_scale, p, pump_rate_flag, lr=0.09, n_iter=50000
):
    """
    the main loop is in this function
    this is basically a gradient descent algorithm
    one the main loop ends, we need to calculate the real values using satisfy_simplex()

    """
    c = torch.zeros((batch_size, n), dtype=torch.float).to(device)

    pump_rate = 1
    for i in range(n_iter):

        if pump_rate_flag:
            pump_rate = (i + 1) / n_iter

        grads = calculate_grads(c, q_mat, c_vector, p, pump_rate)
        Wt = get_w(i, n_iter)
        c = c + lr * (grads + Wt)

        c = torch.clamp(c, -1, 1)

    c_pow = c.pow(2)
    objval = compute_energy(c, q_mat, c_vector, q_scale, c_scale, 0)
    return c_pow, objval


def get_instance_info(instance_type, sizes, densities, num_instances):

    instances = []
    for size in sizes:
        for density in densities:
            for seed in range(num_instances):
                if instance_type == "test":
                    seed += 10
                instance_name = f"{instance_type}{size:03}-{density}-{seed}"
                instances += [instance_name]
    return instances


def load_problem(source, instance_type):
    """loading the box constraint problem

    Args:
        source ([String]): [the path of the problem]

    Returns:
        rval_q ([torch.Tensor]): [Q matrix of the QP problem]
        rval_c ([torch.Tensor]): [c vector of the QP problem]
    """

    rval_q = None
    rval_c = None
    N = None

    with open(source, "r") as stream:
        lines = stream.readlines()
        instance_info = lines[0].split("\n")[0].split("\t")
        if instance_type == "basic":
            N = int(instance_info[0])
            (optimality, optimal_sol, sol_time_gb) = (None, None, None)
            delimit = " "

        else:
            N = int(instance_info[0])
            optimal_sol = float(instance_info[1])
            if instance_info[2].lower() == "true":
                optimality = True
            else:
                optimality = False
            sol_time_gb = float(instance_info[3])
            delimit = "\t"

        rval_q = torch.zeros((N, N), dtype=torch.float).to(device)
        rval_c = torch.zeros((N,), dtype=torch.float).to(device)

        line_data_c = lines[1].split("\n")[0].split(delimit)
        for idx in range(0, N):
            rval_c[idx] = -torch.Tensor([float(line_data_c[idx])])

        for idx, line in enumerate(lines[2:]):
            line_data = line.split("\n")[0].split(delimit)
            for j, value in enumerate(line_data[:N]):
                rval_q[idx, j] = -torch.Tensor([float(value)])

    return N, optimal_sol, optimality, sol_time_gb, rval_q, rval_c


def scale_coefs(q, c, scaling_val=None):
    if scaling_val:
        q_scale = scaling_val
        c_scale = scaling_val
    else:
        q_scale = torch.sqrt(torch.sum(torch.abs(q))) * 0.05
        c_scale = q_scale
    return q / q_scale, c / c_scale, q_scale, c_scale


def get_result_stats(results, optimal_value):
    one_tensor = torch.ones(results.size()).to(device)
    zero_tensor = torch.zeros(results.size()).to(device)

    def count_with_condition(lhs, rhs):
        counter_tensor = torch.where(lhs <= rhs, one_tensor, zero_tensor)
        return round(counter_tensor.sum().item() / results.size()[0], 4)

    (
        optimal,
        one_percent,
        two_percent,
        three_percent,
        four_percent,
        five_percent,
        ten_percent,
    ) = (0, 0, 0, 0, 0, 0, 0)

    gap_tensor = (optimal_value - results) * 100 / torch.abs(results)
    optimal = count_with_condition(gap_tensor, 0.1)
    one_percent = count_with_condition(gap_tensor, 1)
    two_percent = count_with_condition(gap_tensor, 2)
    three_percent = count_with_condition(gap_tensor, 3)
    four_percent = count_with_condition(gap_tensor, 4)
    five_percent = count_with_condition(gap_tensor, 5)
    ten_percent = count_with_condition(gap_tensor, 10)

    return [
        optimal,
        one_percent,
        two_percent,
        three_percent,
        four_percent,
        five_percent,
        ten_percent,
    ]


if __name__ == "__main__":
    post_method = "None"

    t0 = time.time()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(device)

    sizes = np.arange(20, 71, 10)
    densities = [50, 60, 70, 80, 90, 100]
    num_instances = 10
    # Types can be "basic" for the instances on the website, "tuning" for the tuning
    # instances we generated, or "test" for the test instances we generated.
    instance_type = "test"

    parameters_search = "key"  # options: "tune" and "key"

    p_values = [1.5, 1.8, 2.0, 4.0]

    batch_size = 1000
    a_hist = None

    instances = get_instance_info(instance_type, sizes, densities, num_instances)
    N, my_q, my_c, w_dist1, w_dist2, w1, w2, scaling_param = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    iter_options = [15000]
    scaling_options = [None]
    lr_options = [0.0025, 0.005, 0.01]
    noises = [[0.02, 0.5], [0.002, 0.05], [0.0002, 0.005]]
    log_file_path_basic = (
        f"./results/results_Langevin-CIM_{post_method}_{device}_{instance_type}.txt"
    )

    open(
        log_file_path_basic, "w"
    ).close()  # opeining file matters not the name of the opend file
    log_file_basic = open(
        log_file_path_basic, "a"
    )  # opeining file matters not the name of the opend file

    if parameters_search == "key":
        p_values = [None]
        scaling_options = [None]
        iter_options = [None]
        lr_options = [None]
        noises = [None]
        parameters_key = {
            "20": {
                "p": 1.5,
                "scale": None,
                "noise": [0.02, 0.5],
                "lr": 0.01,
                "iter": 15000,
            },
            "30": {
                "p": 1.5,
                "scale": None,
                "noise": [0.02, 0.5],
                "lr": 0.01,
                "iter": 15000,
            },
            "40": {
                "p": 2.0,
                "scale": None,
                "noise": [0.02, 0.5],
                "lr": 0.005,
                "iter": 15000,
            },
            "50": {
                "p": 2.0,
                "scale": None,
                "noise": [0.02, 0.5],
                "lr": 0.01,
                "iter": 15000,
            },
            "60": {
                "p": 1.8,
                "scale": None,
                "noise": [0.02, 0.5],
                "lr": 0.005,
                "iter": 15000,
            },
            "70": {
                "p": 1.8,
                "scale": None,
                "noise": [0.02, 0.5],
                "lr": 0.01,
                "iter": 15000,
            },
        }

    counter = 0
    for ins in instances:
        instance = ins
        if instance_type == "basic":
            instance = ins[0]
        sub_folder = instance_type + "/"
        file_path = "./BoxQP_instances/" + sub_folder + instance + ".in"
        N, optimal_sol, optimality, sol_time_gb, my_q, my_c = load_problem(
            file_path, instance_type
        )
        if instance_type == "basic":
            optimal_sol = ins[1]

        print(instance)
        for noise_option in noises:
            if parameters_search == "key":
                noise_option = parameters_key[str(N)]["noise"]
            w_dist1 = tdist.Normal(
                torch.Tensor([0.0] * batch_size).to(device),
                torch.Tensor([noise_option[0]] * batch_size).to(device),
            )
            w_dist2 = tdist.Normal(
                torch.Tensor([0.0] * batch_size).to(device),
                torch.Tensor([noise_option[1]] * batch_size).to(device),
            )
            for scaling_value in scaling_options:
                if parameters_search == "key":
                    scaling_value = parameters_key[str(N)]["scale"]
                scaling_param = scaling_value
                my_q_scaled, my_c_scaled, q_scale, c_scale = scale_coefs(
                    my_q, my_c, scaling_param
                )
                for my_iter in iter_options:
                    if parameters_search == "key":
                        my_iter = parameters_key[str(N)]["iter"]
                    for my_lr in lr_options:
                        if parameters_search == "key":
                            my_lr = parameters_key[str(N)]["lr"]
                        for p in p_values:
                            if parameters_search == "key":
                                p = parameters_key[str(N)]["p"]
                            for pump_rate_flag in [True]:
                                t0 = time.time()
                                a_hist = np.zeros((batch_size, N, my_iter))
                                c_variables, objective_value = solve(
                                    N,
                                    my_q_scaled,
                                    my_c_scaled,
                                    q_scale,
                                    c_scale,
                                    p,
                                    pump_rate_flag,
                                    lr=my_lr,
                                    n_iter=my_iter,
                                )
                                t1 = time.time()
                                sol_time = t1 - t0
                                results_performance = get_result_stats(
                                    -objective_value, optimal_sol
                                )
                                print(
                                    p,
                                    scaling_value,
                                    my_lr,
                                    noise_option[0],
                                    c_variables.size()[1],
                                    optimal_sol,
                                    round(float(torch.max(-objective_value)), 4),
                                    results_performance[0],
                                    results_performance[1],
                                    results_performance[2],
                                    results_performance[3],
                                    round(sol_time / batch_size, 4),
                                )
                                log_file_basic.write(instance)
                                log_file_basic.write("\t")
                                log_file_basic.write(str(batch_size))
                                log_file_basic.write("\t")
                                log_file_basic.write(str(p))
                                log_file_basic.write("\t")
                                log_file_basic.write(str(scaling_param))
                                log_file_basic.write("\t")
                                log_file_basic.write(str(noise_option[0]))
                                log_file_basic.write("\t")
                                log_file_basic.write(str(my_lr))
                                log_file_basic.write("\t")
                                log_file_basic.write(str(my_iter))
                                log_file_basic.write("\t")
                                log_file_basic.write(str(pump_rate_flag))
                                log_file_basic.write("\t")
                                log_file_basic.write(
                                    "{:.3f}".format(results_performance[0])
                                )
                                log_file_basic.write("\t")
                                log_file_basic.write(
                                    "{:.3f}".format(results_performance[1])
                                )
                                log_file_basic.write("\t")
                                log_file_basic.write(
                                    "{:.3f}".format(results_performance[2])
                                )
                                log_file_basic.write("\t")
                                log_file_basic.write(
                                    "{:.3f}".format(results_performance[3])
                                )
                                log_file_basic.write("\t")
                                log_file_basic.write(
                                    "{:.3f}".format(results_performance[4])
                                )
                                log_file_basic.write("\t")
                                log_file_basic.write(
                                    "{:.3f}".format(results_performance[5])
                                )
                                log_file_basic.write("\t")
                                log_file_basic.write(
                                    "{:.3f}".format(results_performance[6])
                                )
                                log_file_basic.write("\t")
                                log_file_basic.write(
                                    "{:.3f}".format(sol_time / batch_size)
                                )
                                log_file_basic.write("\n")

    log_file_basic.close()
