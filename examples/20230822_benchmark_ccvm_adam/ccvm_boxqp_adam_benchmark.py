import glob
import pickle
import os, sys 
# sys.path.insert(0, os.path.abspath("../"))
from ccvm_simulators.problem_classes.boxqp import ProblemInstance
from ccvm_simulators.solvers import DLSolver

TEST_INSTANCES_DIR = "../test_instances/"
RESULTS_DIR = "./results/dl/"

if __name__ == "__main__":
    # Initialize solver
    batch_size = 1000
    adam_solver = DLSolver(device="cpu", batch_size=batch_size)  # or "cuda"

    # Supply solver parameters for different problem sizes
    adam_solver.parameter_key = {
        20: {"pump": 2.0, "lr": 0.005, "iterations": 15000, "noise_ratio": 10},
    }
    
    # Create directory for hyperparameters required for ADAM algorithm in CCVM-solver
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    # adam_hyperparam = dict(beta1=0.9, beta2=0.999, alpha=0.001)

    # Load test instances to solve
    test_instances_files = [f for f in glob.glob(TEST_INSTANCES_DIR + "*.in")]
    for instance_file in test_instances_files:
        # Load the problem from the problem file into the instance
        boxqp_instance = ProblemInstance(
            instance_type="test",
            file_path=instance_file,
            device=adam_solver.device,
        )

        # Scale the problem's coefficients for more stable optimization
        boxqp_instance.scale_coefs(adam_solver.get_scaling_factor(boxqp_instance.q_matrix))
        
        # Solve the problem
        for alpha in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            for beta1 in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                for beta2 in [0.999]:
                    
                    adam_hyperparameters = dict(alpha=alpha, beta1=beta1, beta2=beta2)
                    
                    dataset = dict(
                        beta1 = adam_hyperparameters['beta1'],
                        beta2 = adam_hyperparameters['beta2'],
                        alpha = adam_hyperparameters['alpha']
                    )
                    
                    # Repeat the experiment 10 times
                    for repeat in range(1, 11):
                        solution = adam_solver(
                            instance=boxqp_instance,
                            post_processor=None,
                            adam_hyperparam=adam_hyperparameters
                        )
                        # print(solution)
                        
                        dataset[f"r{repeat:02d}"] = dict(
                            best_objective_value = solution.best_objective_value,
                            solution_performance = solution.solution_performance,
                            solve_time = solution.solve_time,
                        )
                    
                    dataset['params'] = dict(
                        batch_size = solution.batch_size,
                        device = solution.device,
                        instance_name = solution.instance_name,
                        iterations = solution.iterations,
                        optimal_value = solution.optimal_value,
                        problem_size = solution.problem_size,
                    )
            
                    filename = f"{RESULTS_DIR}adam_alpha{alpha:.05f}_1beta{beta1:.03f}_2beta{beta2:.04f}_iter{solution.iterations:06d}.pkl"
                    with open(filename, 'wb') as file:
                        pickle.dump(dataset, file, pickle.HIGHEST_PROTOCOL)
                    
                    
                    print(dataset)
