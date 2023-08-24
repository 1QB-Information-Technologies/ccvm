import os
import pickle
import matplotlib.pyplot as plt

RESULTS_DIR = "./results/dl/"
PLOT_DIR = "./results/dl-plots/"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


def sub_scatter(baseline, data, subkey):
    base = baseline['solution_performance'][subkey]
    plt.scatter(data['alpha'], data['r01']['solution_performance'][subkey]/base, color=f'C{c}', marker='D')                
    plt.scatter(data['alpha'], data['r02']['solution_performance'][subkey]/base, color=f'C{c}', marker='o')
    plt.scatter(data['alpha'], data['r03']['solution_performance'][subkey]/base, color=f'C{c}', marker='v')
    plt.scatter(data['alpha'], data['r04']['solution_performance'][subkey]/base, color=f'C{c}', marker='^')
    plt.scatter(data['alpha'], data['r05']['solution_performance'][subkey]/base, color=f'C{c}', marker='<')
    plt.scatter(data['alpha'], data['r06']['solution_performance'][subkey]/base, color=f'C{c}', marker='>')
    plt.scatter(data['alpha'], data['r07']['solution_performance'][subkey]/base, color=f'C{c}', marker='1')
    plt.scatter(data['alpha'], data['r08']['solution_performance'][subkey]/base, color=f'C{c}', marker='s')
    plt.scatter(data['alpha'], data['r09']['solution_performance'][subkey]/base, color=f'C{c}', marker='p')
    plt.scatter(data['alpha'], data['r10']['solution_performance'][subkey]/base, color=f'C{c}', marker='.')
    return base

## Data analysis
iterations = 15000

# Original method
filename = f"{RESULTS_DIR}original_iter{iterations:06d}.pkl"
with open(filename, 'rb') as file:
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
for r in range(1,11):
    best_objective_value += data[f'r{1:02d}']['best_objective_value']
    solve_time += data[f'r{1:02d}']['solve_time']
    optimal += data[f'r{1:02d}']['solution_performance']['optimal']
    one_percent += data[f'r{1:02d}']['solution_performance']['one_percent']
    two_percent += data[f'r{1:02d}']['solution_performance']['two_percent']
    three_percent += data[f'r{1:02d}']['solution_performance']['three_percent']
    four_percent += data[f'r{1:02d}']['solution_performance']['four_percent']
    five_percent += data[f'r{1:02d}']['solution_performance']['five_percent']
    ten_percent += data[f'r{1:02d}']['solution_performance']['ten_percent']
    
baseline=dict(
    best_objective_value = best_objective_value/10,
    solve_time = solve_time / 10,
    solution_performance = dict(
        optimal = optimal / 10,
        one_percent = one_percent / 10,
        two_percent = two_percent / 10,
        three_percent = three_percent / 10,
        four_percent = four_percent / 10,
        five_percent = five_percent / 10,
        ten_percent = ten_percent / 10,
    ),
)
print(baseline)

for beta2 in [0.999]:
    for beta1 in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        c = 0
        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(12,4))
        for alpha in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            filename = f"{RESULTS_DIR}adam_alpha{alpha:.05f}_1beta{beta1:.03f}_2beta{beta2:.04f}_iter{iterations:06d}.pkl"
            with open(filename, 'rb') as file:
                data = pickle.load(file)
                # Optimal
                plt.subplot(1,7,1)
                avg = sub_scatter(baseline, data, 'optimal')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])                
                plt.ylabel(r'$\mathrm{optimal}$')
                plt.title(r'$\mathrm{optimal}$'+f'={avg:0.3f}')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,2)
                avg = sub_scatter(baseline, data, 'one_percent')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])
                # plt.ylabel(r'$\%01$')
                plt.title(r'$\%01$'+f'={avg:0.3f}')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,3)
                avg=sub_scatter(baseline, data, 'two_percent')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])                
                # plt.ylabel(r'$\%02$')
                plt.title(r'$\%02$'+f'={avg:0.3f}')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,4)
                avg=sub_scatter(baseline, data, 'three_percent')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])
                # plt.ylabel(r'$\%03$')
                plt.title(r'$\%03$'+f'={avg:0.3f}')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,5)
                avg=sub_scatter(baseline, data, 'four_percent')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])
                # plt.ylabel(r'$\%04$')
                plt.title(r'$\%04$'+f'={avg:0.3f}')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,6)
                avg=sub_scatter(baseline, data, 'five_percent')
                plt.xscale('log')
                # plt.ylabel(r'$\%05$')
                plt.ylim([-0.1, 1.1])
                plt.title(r'$\%05$'+f'={avg:0.3f}')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,7)
                avg=sub_scatter(baseline, data, 'ten_percent')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])
                # plt.ylabel(r'$\%10$')
                plt.title(r'$\%10$'+f'={avg:0.3f}')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
            c+=1
            # print(data['r01']['solution_performance'].keys())
        plt.suptitle(r'$\beta_1=$'+f'{beta1}'+r', $\beta_2=$'+f'{beta2}'+r', $n_\mathrm{iter}$'+f"={data['params']['iterations']}")
        # plt.yscale('log')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{PLOT_DIR}performance_1beta{beta1:.03f}_2beta{beta2:.04f}_iter{iterations:06d}.png", dpi=500)
        plt.close()
        
