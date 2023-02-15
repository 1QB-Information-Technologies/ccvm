import glob
from ccvm.problem_classes.boxqp import ProblemInstance
from ccvm.solvers import DLSolver
import time

TEST_INSTANCES_DIR = "./test_instances/"

if __name__ == "__main__":

    # Initialize solver
    batch_size = 1000
    solver = DLSolver(device="cpu", batch_size=batch_size)  # or "cuda"

    # Supply solver parameters for different problem sizes
    solver.parameter_key = {
        20: {"pump": 2.0, "lr": 0.005, "iterations": 15000, "noise_ratio": 10},
    }

    # Load test instances to solve
    test_instances_files = [f for f in glob.glob(TEST_INSTANCES_DIR + "*.in")]
    for instance_file in test_instances_files:
        # Load the problem from the problem file into the instance
        boxqp_instance = ProblemInstance(
            instance_type="test",
            file_path=instance_file,
            device=solver.device,
        )

        boxqp_instance.scale_coefs(solver.get_scaling_factor(boxqp_instance.q_matrix))

        # Start a timer
        start_time = time.time()
        # Solve the problem
        solution = solver.solve(
            instance=boxqp_instance,
            post_processor=None,
            evolution_step_size=1,
        )
        # Stop the timer
        end_time = time.time()
        # Print the time
        print(f"Time elapsed (variables on CPU): {end_time - start_time}")

        # Start a timer
        start_time = time.time()
        # Solve the problem
        solution = solver.solve_gpu_vars(
            instance=boxqp_instance,
            post_processor=None,
            evolution_step_size=1,
        )
        # Stop the timer
        end_time = time.time()
        # Print the time
        print(f"Time elapsed (variables on GPU): {end_time - start_time}")

        print(solution)
