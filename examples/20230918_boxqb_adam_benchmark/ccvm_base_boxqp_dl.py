import os, pickle
import glob
from ccvm_simulators.problem_classes.boxqp import ProblemInstance
from ccvm_simulators.solvers import DLSolver

TEST_INSTANCES_DIR = "../tuning_instances/"
RESULTS_DIR = "./results/dl/"

if __name__ == "__main__":
    # Initialize solver
    batch_size = 1000
    nrepeats = 5
    solver = DLSolver(device="cpu", batch_size=batch_size)  # or "cuda"

    # Supply solver parameters for different problem sizes
    solver.parameter_key = {
        20: {"pump": 2.0, "dt": 0.005, "iterations": 15000, "noise_ratio": 10},
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
        dataset = dict()
        # Repeat the experiment nrepeats times
        for repeat in range(1, nrepeats + 1):
            solution = solver(
                instance=boxqp_instance,
                post_processor=None,
            )
            disp = f"Base: {solution.instance_name}: {repeat=}, \tsolve-time={solution.solve_time}\n"
            disp += f"performance={solution.solution_performance}\n"
            print(disp)

            dataset[f"r{repeat:02d}"] = dict(
                best_objective_value=solution.best_objective_value,
                solution_performance=solution.solution_performance,
                solve_time=solution.solve_time,
            )

        dataset["params"] = dict(
            batch_size=solution.batch_size,
            device=solution.device,
            instance_name=solution.instance_name,
            iterations=solution.iterations,
            optimal_value=solution.optimal_value,
            problem_size=solution.problem_size,
        )
        filename = f"{RESULTS_DIR}base_N{solution.problem_size}_iter{solution.iterations:06d}.pkl"
        with open(filename, "wb") as file:
            pickle.dump(dataset, file, pickle.HIGHEST_PROTOCOL)
