import glob
from problem_classes.boxqp.problem_instance import ProblemInstance
from ccvm.solvers.dl_solver import DLSolver

TEST_INSTANCES_DIR = "./test_instances/"

if __name__ == "__main__":

    # Initialize solver
    batch_size = 1000
    solver = DLSolver(device="cpu", batch_size=batch_size)  # or "cuda"

    # Supply solver parameters for different problem sizes
    solver.parameter_key = {
        20: {"p": 2.0, "lr": 0.005, "iter": 15000, "nr": 10},
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

        boxqp_instance.scale_coefs(solver.get_scaling_factor(boxqp_instance.q))

        # Solve the problem
        solution = solver.solve(
            instance=boxqp_instance,
            post_processor=None,
        )

        print(solution)