import glob
from ccvm_simulators.problem_classes.boxqp import ProblemInstance
from ccvm_simulators.solvers import MFSolver
from ccvm_simulators.solvers.algorithms import AdamParameters

TEST_INSTANCES_DIR = "./test_instances/"

if __name__ == "__main__":
    # Initialize solver
    batch_size = 1000
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

        # Solve the problem using the Adam algorithm
        adam_parameters = AdamParameters(alpha=0.001, beta1=0.9, beta2=0.999)

        solution = solver(
            instance=boxqp_instance,
            post_processor=None,
            algorithm_parameters=adam_parameters, # Set to None to use the original MF algorithm
        )

        print(solution)
