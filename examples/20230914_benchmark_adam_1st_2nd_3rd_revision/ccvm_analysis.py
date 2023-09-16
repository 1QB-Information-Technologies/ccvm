import os
import pickle
import matplotlib.pyplot as plt


def sub_scatter(baseline, data, subkey, nrep):
    base = baseline["solution_performance"][subkey]
    base_ret = base 
    if base==0.0: base=1
    plt.scatter(
        data["alpha"],
        data["r01"]["solution_performance"][subkey] / base,
        color=f"C{c}",
        marker="D",
    )
    plt.scatter(
        data["alpha"],
        data["r02"]["solution_performance"][subkey] / base,
        color=f"C{c}",
        marker="o",
    )
    plt.scatter(
        data["alpha"],
        data["r03"]["solution_performance"][subkey] / base,
        color=f"C{c}",
        marker="v",
    )
    plt.scatter(
        data["alpha"],
        data["r04"]["solution_performance"][subkey] / base,
        color=f"C{c}",
        marker="^",
    )
    plt.scatter(
        data["alpha"],
        data["r05"]["solution_performance"][subkey] / base,
        color=f"C{c}",
        marker="<",
    )
    avg_subkey_val = 0.0
    for r in range(1, nrep+1):
        avg_subkey_val += data[f"r{r:02d}"]["solution_performance"][subkey]
        # print(f"r{r:02d}", data[f"r{r:02d}"]["solution_performance"][subkey])
    avg_subkey_val = avg_subkey_val / nrep / base
    
    plt.scatter(
        data["alpha"],
        avg_subkey_val,
        color=f"k",
        marker="+", s=200
    )
    plt.scatter(
        data["alpha"],
        avg_subkey_val,
        color=f"k", 
        marker="o", s=20
    )
    return base_ret, avg_subkey_val

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
    for r in range(1, nrep+1):
        best_objective_value += data[f"r{1:02d}"]["best_objective_value"]
        solve_time += data[f"r{1:02d}"]["solve_time"]
        optimal += data[f"r{1:02d}"]["solution_performance"]["optimal"]
        one_percent += data[f"r{1:02d}"]["solution_performance"]["one_percent"]
        two_percent += data[f"r{1:02d}"]["solution_performance"]["two_percent"]
        three_percent += data[f"r{1:02d}"]["solution_performance"]["three_percent"]
        four_percent += data[f"r{1:02d}"]["solution_performance"]["four_percent"]
        five_percent += data[f"r{1:02d}"]["solution_performance"]["five_percent"]
        ten_percent += data[f"r{1:02d}"]["solution_performance"]["ten_percent"]
    
    baseline = dict(
        best_objective_value=best_objective_value / nrep,
        solve_time=solve_time / nrep,
        solution_performance=dict(
            optimal=optimal / nrep,
            one_percent=one_percent / nrep,
            two_percent=two_percent / nrep,
            three_percent=three_percent / nrep,
            four_percent=four_percent / nrep,
            five_percent=five_percent / nrep,
            ten_percent=ten_percent / nrep,
        ),
    )
    return baseline 

#===============================================================================
# RESULTS_DIR = "./results/mf/"
# PLOT_DIR = "./results/mf-plots/"
RESULTS_DIR = "./results/dl/"
PLOT_DIR = "./results/dl-plots/"
# RESULTS_DIR = "./results/lan/"
# PLOT_DIR = "./results/lan-plots/"
#===============================================================================
# maindir = "/Users/mehmetcanturk/CCVM/benchmarks/20230908_results/"
# RESULTS_DIR = maindir+"/dl/"
# PLOT_DIR = maindir+"/new-dl-plots/"
# RESULTS_DIR = maindir+"/lan/"
# PLOT_DIR = maindir+"/new-lan-plots/"
# RESULTS_DIR = maindir+"/mf/"
# PLOT_DIR = maindir+"/new-mf-plots/"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


## Data analysis
iterations = 15000  # 10000 #5000 #
N = 20 #70 # 50 # 
nrep = 5
ylim = [-0.1, 5.0] # [-0.1, 2.0] # scatter limit 

baseline = process_baseline(RESULTS_DIR, N, iterations, nrep)

for beta2 in [0.1, 0.3, 0.5, 0.7, 0.8, 0.999, 1.0]: #[0.5, 0.8, 0.999, 1.0]:  #
    for beta1 in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]: #[0.1, 0.5, 0.7, 0.9]:
        c = 0
        plt.rcParams.update({"font.size": 10})
        plt.figure(figsize=(12, 4))
        #=======================================================================
        # avg_performance_list = dict(opt=[],one=[],two=[],three=[], four=[], five=[], ten=[])
        #=======================================================================
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]:#[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            filename = f"{RESULTS_DIR}N_{N}_A_{alpha:.05f}_B1_{beta1:.03f}_B2_{beta2:.04f}_iter{iterations:06d}.pkl"
            with open(filename, "rb") as file:
                data = pickle.load(file)
                # Optimal
                plt.subplot(1, 7, 1)
                base_avg, opt_avg = sub_scatter(baseline, data, "optimal", nrep)
                plt.xscale("log")
                plt.ylim(ylim)
                plt.ylabel(r"$\mathrm{optimal-rate}$")
                plt.title(r"$\overline{\mathrm{optimal}}$" + f"={base_avg:0.3f}")
                plt.grid(True)
                plt.xlabel(r"$\alpha$")

                plt.subplot(1, 7, 2)
                base_avg, one_avg = sub_scatter(baseline, data, "one_percent", nrep)
                plt.xscale("log")
                plt.ylim(ylim)
                # plt.ylabel(r'$\%01$')
                plt.title(r"$\overline{\%01}$" + f"={base_avg:0.3f}")
                plt.grid(True)
                plt.xlabel(r"$\alpha$")

                plt.subplot(1, 7, 3)
                base_avg, two_avg = sub_scatter(baseline, data, "two_percent", nrep)
                plt.xscale("log")
                plt.ylim(ylim)
                # plt.ylabel(r'$\%02$')
                plt.title(r"$\overline{\%02}$" + f"={base_avg:0.3f}")
                plt.grid(True)
                plt.xlabel(r"$\alpha$")

                plt.subplot(1, 7, 4)
                base_avg, three_avg = sub_scatter(baseline, data, "three_percent", nrep)
                plt.xscale("log")
                plt.ylim(ylim)
                # plt.ylabel(r'$\%03$')
                plt.title(r"$\overline{\%03}$" + f"={base_avg:0.3f}")
                plt.grid(True)
                plt.xlabel(r"$\alpha$")

                plt.subplot(1, 7, 5)
                base_avg, four_avg = sub_scatter(baseline, data, "four_percent", nrep)
                plt.xscale("log")
                plt.ylim(ylim)
                # plt.ylabel(r'$\%04$')
                plt.title(r"$\overline{\%04}$" + f"={base_avg:0.3f}")
                plt.grid(True)
                plt.xlabel(r"$\alpha$")

                plt.subplot(1, 7, 6)
                base_avg, five_avg = sub_scatter(baseline, data, "five_percent", nrep)
                plt.xscale("log")
                # plt.ylabel(r'$\%05$')
                plt.ylim(ylim)
                plt.title(r"$\overline{\%05}$" + f"={base_avg:0.3f}")
                plt.grid(True)
                plt.xlabel(r"$\alpha$")

                plt.subplot(1, 7, 7)
                base_avg, ten_avg = sub_scatter(baseline, data, "ten_percent", nrep)
                plt.xscale("log")
                plt.ylim(ylim)
                # plt.ylabel(r'$\%10$')
                plt.title(r"$\overline{\%10}$" + f"={base_avg:0.3f}")
                plt.grid(True)
                plt.xlabel(r"$\alpha$")
                # avg_performance_list['opt'].append(opt_avg)
                # avg_performance_list['one'].append(one_avg)
                # avg_performance_list['two'].append(two_avg)
                # avg_performance_list['three'].append(three_avg)
                # avg_performance_list['four'].append(four_avg)
                # avg_performance_list['five'].append(five_avg)
                # avg_performance_list['ten'].append(ten_avg)
            c += 1
            # print(data['r01']['solution_performance'].keys())
        plt.suptitle(
            r"$\beta_1=$"
            + f"{beta1}"
            + r", $\beta_2=$"
            + f"{beta2}"
            + r", $n_\mathrm{iter}$"
            + f"={data['params']['iterations']}"
        )
        plt.text(0.001, 0.2, '+: avg')
        # plt.yscale('log')
        plt.tight_layout()
        # plt.show()
        plt.savefig(
            f"{PLOT_DIR}Ps_N{N}_beta1_{beta1:.03f}_beta2_{beta2:.04f}_iter{iterations:06d}.png",
            dpi=500,
        )
        plt.close()
        