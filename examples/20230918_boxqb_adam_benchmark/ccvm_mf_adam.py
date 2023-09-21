import sys 
sys.path.append("../../")
import os, pickle
import glob
from ccvm_simulators.problem_classes.boxqp import ProblemInstance
from ccvm_simulators.solvers import MFSolver

TEST_INSTANCES_DIR = "../tuning_instances/"
RESULTS_DIR = "./results_mf/data/"

if __name__ == "__main__":
    # Initialize solver
    batch_size = 1000
    nrepeats = 5
    solver = MFSolver(device="cpu", batch_size=batch_size)  # or "cuda"
    
    # Supply solver parameters for different problem sizes
    solver.parameter_key = {
        20: {
            "pump": 0.5,
            "feedback_scale": 20,
            "j": 20,
            "S": 0.2,
            "dt": 0.0025,
            "iterations": 15000,
        }
    }
    # Create directory for hyperparameters required for ADAM algorithm in CCVM-solver
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Load test instances to solve
    test_instances_files = [f for f in glob.glob(TEST_INSTANCES_DIR + "*.in")]
    for instance_file in test_instances_files:
        # Load the problem from the problem file into the instance
        boxqp_instance = ProblemInstance(
            instance_type="test",
            file_path=instance_file,
            device=solver.device,
        )

        # Scale the problem's coefficients for more stable optimization
        boxqp_instance.scale_coefs(solver.get_scaling_factor(boxqp_instance.q_matrix))

        # Solve the problem
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0]:
            for beta1 in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:
                for beta2 in [0.1, 0.3, 0.5, 0.7, 0.8, 0.999, 1.0]:
                    dataset = {"ADD_ASSIGN":{}, "ASSIGN":{}, "hyperparameters": dict(alpha=alpha, beta1=beta1, beta2=beta2)}
                    for which_adam in ["ADD_ASSIGN", "ASSIGN"]:
                        for repeat in range(1, nrepeats + 1):
                            solution = solver(
                                instance=boxqp_instance,
                                solve_type="Adam",  # solve_type=None refers to default (original) solver
                                post_processor=None,
                                hyperparameters=dict(alpha=alpha, beta1=beta1, beta2=beta2, which_adam=which_adam),
                            )
                            disp = f"{which_adam}: {solution.instance_name}: {repeat=}, \tsolve-time={solution.solve_time}\n"
                            disp += f"performance={solution.solution_performance}\n"
                            # print(disp)
                            
                            '''
                            objective_value_relative += data[rkey]['f_relative_objective_value']
                            objective_value_absolute += data[rkey]['f_absolute_objective_value']
                            objective_value_best += data[rkey]["f_best_objective_value"]
                            objective_value_optimal_value += data[rkey]['f_optimal_value']
                            '''
                            dataset[which_adam][f"r{repeat:02d}"] = dict(
                                f_relative_objective_value = solution.optimal_value - solution.best_objective_value,
                                f_absolute_objective_value = abs((solution.optimal_value - solution.best_objective_value) / solution.optimal_value),
                                f_best_objective_value=solution.best_objective_value,
                                f_optimal_value=solution.optimal_value,
                                solution_performance=solution.solution_performance,
                                solve_time=solution.solve_time,
                            )
                    dataset["instance"] = dict(
                        batch_size=solution.batch_size,
                        device=solution.device,
                        instance_name=solution.instance_name,
                        iterations=solution.iterations,
                        problem_size=solution.problem_size,
                    )
                    filename = f"{RESULTS_DIR}adam_N{solution.problem_size}_iter{solution.iterations:05d}_B2_{beta2:.03f}_B1_{beta1:.03f}_A{alpha:.03f}.pkl"
                    with open(filename, "wb") as file:
                        pickle.dump(dataset, file, pickle.HIGHEST_PROTOCOL)
                    print(dataset) 
