import glob
from ccvm_simulators.problem_classes.boxqp import ProblemInstance
from ccvm_simulators.solvers import DLSolver

TEST_INSTANCES_DIR = "./test_instances/"

if __name__ == "__main__":
    # Initialize solver
    batch_size = 1000
    solver = DLSolver(device="cpu", batch_size=batch_size)  # or "cuda"

    # Supply solver parameters for different problem sizes
    solver.parameter_key = {
        20: {"pump": 2.0, "dt": 0.005, "iterations": 15000, "noise_ratio": 10},
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

        # Scale the problem's coefficients for more stable optimization
        boxqp_instance.scale_coefs(solver.get_scaling_factor(boxqp_instance.q_matrix))

        # Solve the problem
        solution = solver(
            instance=boxqp_instance,
            solve_type = "Adam", # solve_type=None refers to default (original) solver
            post_processor=None,
            hyperparameters=dict(beta1=0.9, beta2=1.0, alpha=0.001),
        )

        print(solution)
