import glob
from ccvm_simulators.problem_classes.boxqp import ProblemInstance
from ccvm_simulators.solvers import DLSolver
from ccvm_simulators.solvers.algorithms import AdamParameters

# Inputs
TEST_INSTANCES_DIR_NAME = "single_test_instance"
TEST_INSTANCES_PATH = f"./benchmarking_instances/{TEST_INSTANCES_DIR_NAME}/"

if __name__ == "__main__":
    # Initialize solver
    batch_size = 1000
    solver = DLSolver(device="cpu", batch_size=batch_size)  # or "cuda"

    # Supply solver parameters for different problem sizes
    solver.parameter_key = {
        20: {"pump": 2.0, "dt": 0.005, "iterations": 15000, "noise_ratio": 10},
    }

    # Load test instances to solve
    test_instances_files = [f for f in glob.glob(TEST_INSTANCES_PATH + "*.in")]
    for instance_file in test_instances_files:
        # Load the problem from the problem file into the instance
        boxqp_instance = ProblemInstance(
            instance_type="test",
            file_path=instance_file,
            device=solver.device,
        )

        # Scale the problem's coefficients for more stable optimization
        boxqp_instance.scale_coefs(solver.get_scaling_factor(boxqp_instance.q_matrix))

        # Solve the problem with one of the methods by setting
        # (1) algorithm_parameters=None (Default) for original algorithm
        # (2) algorithm_parameters=AdamParameters(..) for the Adam algorithm
        solution = solver(
            instance=boxqp_instance,
            post_processor=None,
            # algorithm_parameters=AdamParameters(
            #     alpha=0.001, beta1=0.9, beta2=0.999, add_assign=False
            # ),
        )

        print(solution)
