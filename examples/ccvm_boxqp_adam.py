import glob
import os, sys 
# sys.path.insert(0, os.path.abspath("../"))
from ccvm_simulators.problem_classes.boxqp import ProblemInstance
from ccvm_simulators.solvers import DLSolver

TEST_INSTANCES_DIR = "./test_instances/"

if __name__ == "__main__":
    # Initialize solver
    batch_size = 1000
    adam_solver = DLSolver(device="cpu", batch_size=batch_size)  # or "cuda"

    # Supply solver parameters for different problem sizes
    adam_solver.parameter_key = {
        20: {"pump": 2.0, "lr": 0.005, "iterations": 15000, "noise_ratio": 10},
    }

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
        solution = adam_solver(
            instance=boxqp_instance,
            post_processor=None,
        )

        print(solution)
