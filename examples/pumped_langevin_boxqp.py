import glob
from ccvm_simulators.problem_classes.boxqp import ProblemInstance
from ccvm_simulators.solvers import PumpedLangevinSolver
from ccvm_simulators.solvers.algorithms import AdamParameters

TEST_INSTANCES_DIR = "./test_instances/"  # "./tuning_instances/"

if __name__ == "__main__":
    # Initialize solver
    batch_size = 1000
    solver = PumpedLangevinSolver(
        device="cpu", batch_size=batch_size, S=0.5
    )  # or "cuda"

    # Supply solver parameters for different problem sizes
    solver.parameter_key = {
        20: {
            "pump": 2.0,  # p0
            "dt": 0.002,
            "iterations": 15000,
            "sigma": 0.5,
            "feedback_scale": 1.0,
        },
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

        # Solve the problem with one of the methods by setting
        # (1) algorithm_parameters=None for original algorithm
        # (2) algorithm_parameters=AdamParameters(..) for the Adam algorithm
        solution = solver(
            instance=boxqp_instance,
            post_processor=None,
            algorithm_parameters=AdamParameters(
                alpha=0.001, beta1=0.9, beta2=0.999, add_assign=True
            ),
        )

        print(solution)
