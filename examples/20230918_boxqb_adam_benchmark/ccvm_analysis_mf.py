import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def plot_success_probability(alphas, beta2, beta1, baseline_sp, adam_sp, niter, ps_max):
    plt.suptitle(
        r"$\beta_1=$"
        + f"{beta1}"
        + r", $\beta_2=$"
        + f"{beta2}"
        + r", $n_\mathrm{iter}$"
        + f"={niter}"
    )
    ylim = [-0.1, ps_max + 0.1]

    titles = ["opt", "one", "two", "three", "four", "five", "ten"]

    def sub_plot_rest(n):
        plt.subplot(1, 7, n + 1)
        plt.title(titles[n], fontsize=8)
        plt.plot(alphas, adam_sp[:, n], marker="o", color=f"C3", label="adam")
        plt.plot(
            [alphas[0], alphas[-1]],
            [baseline_sp[n], baseline_sp[n]],
            color="C0",
            linewidth=2.75,
            label="orig",
        )
        plt.xscale("log")
        plt.ylim(ylim)
        if n == 0:
            plt.ylabel(r"$P_\mathrm{s}$")
        plt.grid(True)
        plt.xlabel(r"$\alpha$")

    for n in range(adam_sp.shape[1]):
        sub_plot_rest(n)

    plt.legend(loc="lower right")


def process_baseline(RESULTS_DIR, N, iterations, nrep):
    # Original method: base_N20_iter015000
    filename = f"{RESULTS_DIR}base_N{N}_iter{iterations:06d}.pkl"

    with open(filename, "rb") as file:
        data = pickle.load(file)

    best_objective_value = 0
    optimal = 0
    one_percent = 0
    two_percent = 0
    three_percent = 0
    four_percent = 0
    five_percent = 0
    ten_percent = 0
    solve_time = 0
    for r in range(1, nrep + 1):
        rkey = f"r{r:02d}"
        best_objective_value += data[rkey]["best_objective_value"]
        solve_time += data[rkey]["solve_time"]
        optimal += data[rkey]["solution_performance"]["optimal"]
        one_percent += data[rkey]["solution_performance"]["one_percent"]
        two_percent += data[rkey]["solution_performance"]["two_percent"]
        three_percent += data[rkey]["solution_performance"]["three_percent"]
        four_percent += data[rkey]["solution_performance"]["four_percent"]
        five_percent += data[rkey]["solution_performance"]["five_percent"]
        ten_percent += data[rkey]["solution_performance"]["ten_percent"]

    baseline = dict(
        best_objective_value=best_objective_value / nrep,
        solve_time=solve_time / nrep,
        solution_performance=(
            optimal / nrep,
            one_percent / nrep,
            two_percent / nrep,
            three_percent / nrep,
            four_percent / nrep,
            five_percent / nrep,
            ten_percent / nrep,
        ),
    )
    return baseline


def process_adam(alpha, beta1, beta2, RESULTS_DIR, N, iterations, nrep):
    filename = f"{RESULTS_DIR}N_{N}_A_{alpha:.05f}_B1_{beta1:.03f}_B2_{beta2:.04f}_iter{iterations:06d}.pkl"
    with open(filename, "rb") as file:
        data = pickle.load(file)

    best_objective_value = 0
    optimal = 0
    one_percent = 0
    two_percent = 0
    three_percent = 0
    four_percent = 0
    five_percent = 0
    ten_percent = 0
    solve_time = 0
    for r in range(1, nrep + 1):
        rkey = f"r{r:02d}"
        best_objective_value += data[rkey]["best_objective_value"]
        solve_time += data[rkey]["solve_time"]
        optimal += data[rkey]["solution_performance"]["optimal"]
        one_percent += data[rkey]["solution_performance"]["one_percent"]
        two_percent += data[rkey]["solution_performance"]["two_percent"]
        three_percent += data[rkey]["solution_performance"]["three_percent"]
        four_percent += data[rkey]["solution_performance"]["four_percent"]
        five_percent += data[rkey]["solution_performance"]["five_percent"]
        ten_percent += data[rkey]["solution_performance"]["ten_percent"]

    adam = dict(
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        best_objective_value=best_objective_value / nrep,
        solve_time=solve_time / nrep,
        solution_performance=(
            optimal / nrep,
            one_percent / nrep,
            two_percent / nrep,
            three_percent / nrep,
            four_percent / nrep,
            five_percent / nrep,
            ten_percent / nrep,
        ),
    )
    return adam


resultdir = "./results_mf/"
Beta2 = np.array([0.1, 0.3, 0.5, 0.7, 0.8, 0.999, 1.0])
Beta1 = np.array([0.1, 0.3, 0.5, 0.7, 0.8, 0.9])
Alpha = np.array(
    [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0]
)

RESULTS_DIR = resultdir + "data/"
PLOT_DIR = resultdir + "plot/"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

## Data analysis
iterations = 15000  # 10000 #5000 #
N = 20  # 70 # 50 #
nrep = 5
save_plot = True

Baseline = process_baseline(RESULTS_DIR, N, iterations, nrep)
SuccessProb = np.zeros(
    (
        Beta2.shape[0],
        Beta1.shape[0],
        Alpha.shape[0],
        len(Baseline["solution_performance"]),
    )
)
BestObjectVal = np.zeros((Beta2.shape[0], Beta1.shape[0], Alpha.shape[0]))
SolveTime = np.zeros((Beta2.shape[0], Beta1.shape[0], Alpha.shape[0]))

ps_max = np.full(
    (Beta2.shape[0], Beta1.shape[0]), np.max(Baseline["solution_performance"])
)

# Process data
for i in range(Beta2.shape[0]):
    for j in range(Beta1.shape[0]):
        for k in range(Alpha.shape[0]):
            adam = process_adam(
                Alpha[k], Beta1[j], Beta2[i], RESULTS_DIR, N, iterations, nrep
            )
            SuccessProb[i, j, k, :] = adam["solution_performance"]

            tmp_ps_max = np.max(SuccessProb[i, j, k, :])
            if ps_max[i, j] < tmp_ps_max:
                ps_max[i, j] = tmp_ps_max

            BestObjectVal[i, j, k] = adam["best_objective_value"]
            SolveTime[i, j, k] = adam["solve_time"]

# Plot outputs
for i in range(Beta2.shape[0]):
    for j in range(Beta1.shape[0]):
        plt.rcParams.update({"font.size": 10})
        plt.figure(figsize=(12, 4))
        plot_success_probability(
            Alpha,
            Beta2[i],
            Beta1[j],
            Baseline["solution_performance"],
            SuccessProb[i, j, :, :],
            iterations,
            ps_max[i, j],
        )
        plt.tight_layout()
        if save_plot == True:
            plt.savefig(
                f"{PLOT_DIR}Ps_N{N}_beta1_{Beta1[j]:.03f}_beta2_{Beta2[i]:.04f}_iter{iterations:06d}.png",
                dpi=500,
            )
            plt.close()
        else:
            plt.show()
