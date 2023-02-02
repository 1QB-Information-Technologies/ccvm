import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from statistics import median
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from IPython.display import Image
import sys
import os

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(os.path.join(root_folder, "utility"))
from sampleTTSmetric import SampleTTSMetric

plot_TTS = True
plot_success_p = False
save_results = False

n_iter = 15000
device = "cuda"  # options: "cuda" or "cpu"
post_method = "None"  # options: "None" or "BFGS"
instance_type = "test"  # options: "basic", "tuning", or "test"
jacobian = "jac"
TTS_type = "wallclock"  # options: "physical" or "wallclock"
TTS_method = "key"  # options: "best" or "key"
file_path = f"./results/results_Langevin_{post_method}_{device}_{instance_type}.txt"
result_path = f"./results/tts_Langevin_{post_method}_{device}_{instance_type}.txt"
print(file_path)

all_data = []
with open(file_path, "r") as stream:
    lines = stream.readlines()
    for line in lines:
        line_data = line.split("\n")[0].split("\t")
        all_data.append(line_data)

my_df = pd.DataFrame(
    np.array(all_data),
    columns=[
        "instance",
        "bsize",
        "scaling",
        "noise",
        "lr",
        "n_iter",
        "pump_rate_flag",
        "opt",
        "1per",
        "2per",
        "3per",
        "4per",
        "5per",
        "10per",
        "time",
    ],
)
b_size = int(my_df.at[0, "bsize"])

df = my_df.drop(["bsize"], axis=1)
df["category"] = df["scaling"] + df["noise"] + df["lr"] + df["n_iter"]
if (instance_type == "basic") or (instance_type == "test"):
    df["N"] = df["instance"].astype(str).str[4:7].astype(int)

elif instance_type == "tuning":
    df["N"] = df["instance"].astype(str).str[6:9].astype(int)

df = df.rename({"1per": "1% gap", "5per": "5% gap", "10per": "10% gap"}, axis=1)
df.head()

category_list = df["category"].drop_duplicates().values
# instance_sizes = ["20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120"]
instance_sizes = ["20", "30", "40", "50", "60", "70"]
# instance_sizes = ["20", "30"]
# print(category_list)
basic_hypers = {
    "20": "",
    "30": "",
    "40": "",
    "50": "",
    "60": "",
    "70": "",
    "80": "",
    "90": "",
    "100": "",
    "110": "",
    "120": "",
}

if TTS_method == "best":
    for n in instance_sizes:
        perc_nonzero = 0.0
        mean_values = 0.0
        type_ind = 0
        types = ["opt", "1% gap", "5% gap", "10% gap"]
        while (perc_nonzero == 0.0) & (type_ind < 4):
            for category in category_list:
                passed_df = df.loc[(df["category"] == category) & (df["N"] == int(n))][
                    types[type_ind]
                ]
                sp_all = np.array(passed_df.values).astype(float)
                perc_new = np.count_nonzero(sp_all) / len(sp_all)
                if perc_new > perc_nonzero:
                    basic_hypers[str(n)] = category
                    perc_nonzero = perc_new
                    mean_values = sp_all.mean()
                elif perc_new == perc_nonzero:
                    if sp_all.mean() >= mean_values:
                        basic_hypers[str(n)] = category
                        mean_values = sp_all.mean()
            type_ind += 1
elif TTS_method == "key":
    basic_hypers = {
        "20": "100.020.0115000",
        "30": "None0.020.0115000",
        "40": "None0.020.0115000",
        "50": "None0.020.00515000",
        "60": "None0.020.0115000",
        "70": "None0.020.0115000",
        "80": "",
        "90": "",
        "100": "",
        "110": "",
        "120": "",
    }


def generate_arrays_of_TTS(_df, type_tts, n, TTS_type):
    # arrays do not ahve the real values,
    # instead they have the vlaues that would give the same TTS
    # Since we know the succes probabilty
    # we will make arrays that give the same succes probability
    row_count = _df.shape[0]
    best_values = np.ones(row_count) * 10
    # print(row_count)
    all_results = []
    sp_all = _df[type_tts].values
    time_all = _df["time"].values
    # post_time_all = _df["post_time"].values
    post_time_all = 0.0
    for row in range(row_count):
        percentage_counter = 0
        sp = float(sp_all[row])
        array_to_add = []
        sp_checker = int(sp * b_size)
        # post_time = float(post_time_all[row])
        post_time = 0.0
        if TTS_type == "wallclock":
            cpu_time = float(time_all[row])
        elif TTS_type == "physical":
            cpu_time = float(n) * 100e-12 * n_iter + post_time
        for i in range(b_size):
            if percentage_counter < sp_checker:
                array_to_add.append({"best_energy": 10, "time": cpu_time})
            else:
                array_to_add.append({"best_energy": 15, "time": cpu_time})
            percentage_counter += 1
        all_results.append(array_to_add)

    return best_values, all_results, sp_all


basic_perc_results = {
    "opt": {
        "20": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "30": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "40": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "50": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "60": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "70": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "80": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "90": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "100": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "125": {"25": 0, "50": 0, "75": 0, "sp": 0},
    },
    "1% gap": {
        "20": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "30": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "40": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "50": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "60": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "70": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "80": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "90": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "100": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "125": {"25": 0, "50": 0, "75": 0, "sp": 0},
    },
    "5% gap": {
        "20": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "30": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "40": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "50": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "60": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "70": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "80": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "90": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "100": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "125": {"25": 0, "50": 0, "75": 0, "sp": 0},
    },
    "10% gap": {
        "20": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "30": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "40": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "50": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "60": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "70": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "80": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "90": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "100": {"25": 0, "50": 0, "75": 0, "sp": 0},
        "125": {"25": 0, "50": 0, "75": 0, "sp": 0},
    },
}
opt_types = ["opt", "1% gap", "5% gap", "10% gap"]
# opt_types = ["opt"]

if save_results:
    open(result_path, "w").close()
    log_file_tts = open(result_path, "a")

for opt_type in opt_types:
    if save_results:
        log_file_tts.write(opt_type)
        log_file_tts.write("\t")
    for n in instance_sizes:
        passed_df = df.loc[(df["category"] == basic_hypers[n]) & (df["N"] == int(n))]
        for perc in [25, 50, 75]:
            sampler = SampleTTSMetric(
                tau_attribute="time", percentile=perc, seed=1, num_bootstraps=1000
            )
            # print(passed_df[['opt', 'time', 'category', 'N']])
            best_known_energies, results, sp_all = generate_arrays_of_TTS(
                passed_df, opt_type, n, TTS_type
            )
            sp_mean = np.mean(np.array([float(sp) for sp in sp_all]))
            # print(sp_all)
            if len(best_known_energies) == 0:
                mean_TTS = np.inf
                std_TTS = np.inf
            else:
                mean_TTS, std_TTS = sampler.calc(results, best_known_energies)
            print(
                "{}\t{}\t{}\t{}\t{}\t{}".format(
                    opt_type, n, perc, basic_hypers[n], mean_TTS, std_TTS
                )
            )
            if save_results:
                log_file_tts.write(str(round(mean_TTS, 4)))
                log_file_tts.write(",")
                log_file_tts.write(str(round(std_TTS, 4)))
                log_file_tts.write(" ")
            basic_perc_results[opt_type][n][str(perc)] = mean_TTS
        if save_results:
            log_file_tts.write("\t")

    basic_perc_results[opt_type][n]["sp"] = sp_mean
    if save_results:
        log_file_tts.write("\n")

x_vals = instance_sizes
tts_50_opt = [basic_perc_results["opt"][n]["50"] for n in x_vals]
tts_25_opt = [basic_perc_results["opt"][n]["25"] for n in x_vals]
tts_75_opt = [basic_perc_results["opt"][n]["75"] for n in x_vals]

tts_50_1per = [basic_perc_results["1% gap"][n]["50"] for n in x_vals]
tts_25_1per = [basic_perc_results["1% gap"][n]["25"] for n in x_vals]
tts_75_1per = [basic_perc_results["1% gap"][n]["75"] for n in x_vals]

tts_50_5per = [basic_perc_results["5% gap"][n]["50"] for n in x_vals]
tts_25_5per = [basic_perc_results["5% gap"][n]["25"] for n in x_vals]
tts_75_5per = [basic_perc_results["5% gap"][n]["75"] for n in x_vals]

tts_50_10per = [basic_perc_results["10% gap"][n]["50"] for n in x_vals]
tts_25_10per = [basic_perc_results["10% gap"][n]["25"] for n in x_vals]
tts_75_10per = [basic_perc_results["10% gap"][n]["75"] for n in x_vals]

sp_opt = [basic_perc_results["opt"][n]["sp"] for n in x_vals]
sp_1per = [basic_perc_results["1% gap"][n]["sp"] for n in x_vals]
sp_5per = [basic_perc_results["5% gap"][n]["sp"] for n in x_vals]
sp_10per = [basic_perc_results["10% gap"][n]["sp"] for n in x_vals]


if plot_TTS:
    plt.figure(figsize=(7.7, 7.0))
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    fonts = {"xlabel": 36, "ylabel": 36, "legend": 26, "xticks": 32, "yticks": 32}
    plt.fill_between(x_vals, tts_25_opt, tts_75_opt, color="blue", alpha=0.2)
    plt.fill_between(x_vals, tts_25_1per, tts_75_1per, color="orange", alpha=0.2)
    plt.fill_between(x_vals, tts_25_5per, tts_75_5per, color="green", alpha=0.2)
    plt.fill_between(x_vals, tts_25_10per, tts_75_10per, color="red", alpha=0.2)

    plt.plot(
        x_vals,
        tts_50_opt,
        linestyle="-",
        marker="s",
        label="0.1\% gap",
        color="blue",
        linewidth=4.0,
    )
    plt.plot(
        x_vals,
        tts_50_1per,
        linestyle="-",
        marker="s",
        label="1\% gap",
        color="orange",
        linewidth=4.0,
    )
    plt.plot(
        x_vals,
        tts_50_5per,
        linestyle="-",
        marker="s",
        label="5\% gap",
        color="green",
        linewidth=4.0,
    )
    plt.plot(
        x_vals,
        tts_50_10per,
        linestyle="-",
        marker="s",
        label="10\% gap",
        color="red",
        linewidth=4.0,
    )

    plt.xlabel("Problem Size $N$", fontsize=fonts["xlabel"])
    plt.ylabel("TTS (seconds)", fontsize=fonts["ylabel"])
    plt.plot(
        [],
        [],
        linestyle="-",
        marker="s",
        label="(median)",
        color="black",
        linewidth=4.0,
    )
    plt.fill_between([], [], alpha=0.2, label="(IQR)")
    plt.yscale("log")  # log scale
    tts_50_opt = np.array(tts_50_opt)
    tts_50_10per = np.array(tts_50_10per)
    tts_50_opt[np.isnan(tts_50_opt)] = 1.0
    tts_50_opt[np.isinf(tts_50_opt)] = 1.0
    tts_50_10per[np.isnan(tts_50_10per)] = 1.0
    tts_50_10per[np.isinf(tts_50_10per)] = 1.0
    # upper_lim = 10 ** (int(np.log10(min(1000, np.max(tts_50_opt)))) + 1)
    if TTS_type == "wallclock":
        upper_lim = 10 ** 2
        # lower_lim = 10 ** (int(np.log10(np.min(tts_50_10per))) - 1)
        lower_lim = 10 ** (-3)
    elif TTS_type == "physical":
        upper_lim = 10 ** (-1)
        lower_lim = 10 ** (-6)
    plt.ylim(lower_lim, upper_lim)  # limit on y values
    # plt.xticks([50,100,150]) # Values on the graph for the x axis
    plt.grid(
        b=True, which="major", color="#666666", linestyle="--"
    )  # grid lines on the graph
    plt.legend(
        loc="upper left",
        fontsize=fonts["legend"],
        labelspacing=0.5,
        handlelength=0.75,
        ncol=2,
        columnspacing=1.8,
    )  # chosing the location of the legend on the graph
    plt.xticks(fontsize=fonts["xticks"])
    plt.yticks(fontsize=fonts["yticks"])
    plt.tight_layout()

    plt.savefig(
        f"./results/tts_boxed_qp_Langevin_{post_method}_{TTS_type}_{device}_{instance_type}.pdf",
        bbox_inches="tight",
    )

if plot_success_p:
    plt.rcParams.update({"font.size": 11})
    plt.figure()
    plt.plot(x_vals, sp_opt, linestyle="-", marker="s", label="0.1% gap", color="blue")
    plt.plot(x_vals, sp_1per, linestyle="-", marker="s", label="1% gap", color="orange")
    plt.plot(x_vals, sp_5per, linestyle="-", marker="s", label="5% gap", color="green")
    plt.plot(x_vals, sp_10per, linestyle="-", marker="s", label="10% gap", color="red")
    plt.grid(b=True, which="major", color="#666666", linestyle="--")
    plt.legend(loc="best")
    plt.yscale("log")
    plt.xlabel("Problem Size $N$")
    plt.ylabel("Success Probability")
    plt.tight_layout()
    plt.savefig(
        f"./results/sp_boxed_qp_Langevin_{post_method}_{n_iter}_{device}.png",
        bbox_inches="tight",
    )
    plt.show()
