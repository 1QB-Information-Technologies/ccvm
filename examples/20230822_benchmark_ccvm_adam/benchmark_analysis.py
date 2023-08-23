import os
import pickle
import matplotlib.pyplot as plt

RESULTS_DIR = "./results/dl/"
PLOT_DIR = "./results/dl-plot/"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


def sub_scatter(data, subkey):
    plt.scatter(data['alpha'], data['r01']['solution_performance'][subkey], color=f'C{c}', marker='D')                
    plt.scatter(data['alpha'], data['r02']['solution_performance'][subkey], color=f'C{c}', marker='o')
    plt.scatter(data['alpha'], data['r03']['solution_performance'][subkey], color=f'C{c}', marker='v')
    plt.scatter(data['alpha'], data['r04']['solution_performance'][subkey], color=f'C{c}', marker='^')
    plt.scatter(data['alpha'], data['r05']['solution_performance'][subkey], color=f'C{c}', marker='<')
    plt.scatter(data['alpha'], data['r06']['solution_performance'][subkey], color=f'C{c}', marker='>')
    plt.scatter(data['alpha'], data['r07']['solution_performance'][subkey], color=f'C{c}', marker='1')
    plt.scatter(data['alpha'], data['r08']['solution_performance'][subkey], color=f'C{c}', marker='s')
    plt.scatter(data['alpha'], data['r09']['solution_performance'][subkey], color=f'C{c}', marker='p')
    plt.scatter(data['alpha'], data['r10']['solution_performance'][subkey], color=f'C{c}', marker='.')


iterations = 15000
for beta2 in [0.999]:
    for beta1 in [0.8, 0.9]:
        c = 0
        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(12,4))
        for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
            filename = f"{RESULTS_DIR}adam_alpha{alpha:.05f}_1beta{beta1:.03f}_2beta{beta2:.04f}_iter{iterations:06d}.pkl"
            with open(filename, 'rb') as file:
                data = pickle.load(file)
                # Optimal
                plt.subplot(1,7,1)
                sub_scatter(data, 'optimal')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])                
                plt.ylabel(r'$\mathrm{optimal}$')
                # plt.title(r'$\mathrm{optimal}$')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,2)
                sub_scatter(data, 'one_percent')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])
                # plt.ylabel(r'$\%01$')
                plt.title(r'$\%01$')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,3)
                sub_scatter(data, 'two_percent')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])                
                # plt.ylabel(r'$\%02$')
                plt.title(r'$\%02$')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,4)
                sub_scatter(data, 'three_percent')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])
                # plt.ylabel(r'$\%03$')
                plt.title(r'$\%03$')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,5)
                sub_scatter(data, 'four_percent')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])
                # plt.ylabel(r'$\%04$')
                plt.title(r'$\%04$')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,6)
                sub_scatter(data, 'five_percent')
                plt.xscale('log')
                # plt.ylabel(r'$\%05$')
                plt.ylim([-0.1, 1.1])
                plt.title(r'$\%05$')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
                
                plt.subplot(1,7,7)
                sub_scatter(data, 'ten_percent')
                plt.xscale('log')
                plt.ylim([-0.1, 1.1])
                # plt.ylabel(r'$\%10$')
                plt.title(r'$\%10$')
                plt.grid(True)
                plt.xlabel(r'$\alpha$')
            c+=1
            print(data['r01']['solution_performance'].keys())
        plt.suptitle(r'$\beta_1=$'+f'{beta1}'+r', $\beta_2=$'+f'{beta2}'+r', $\mathrm{iter}$'+f"={data['params']['iterations']}")
        # plt.yscale('log')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{PLOT_DIR}performance_1beta{beta1:.03f}_2beta{beta2:.04f}_iter{iterations:06d}.png", dpi=500)
        plt.close()
        
