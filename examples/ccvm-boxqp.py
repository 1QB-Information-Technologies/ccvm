import glob
import os
from ccvmplotlib import ccvmplotlib

from problem_classes.boxqp.problem_instance import ProblemInstance
from problem_classes.boxqp.results import Results
from ccvm.solvers.dl_solver import DLSolver

RESULTS_DIR = "./results"
TEST_OUTPUT_DEST = f"{RESULTS_DIR}/DL-CCVM_LGFGS_cpu_test.txt"
TEST_INSTANCES_DIR = "./test_instances/"
PLOT_OUTPUT_DIR = "./plots"
PLOT_OUTPUT_DEST = f"{PLOT_OUTPUT_DIR}/DL-CCVM_LBFGS_cpu_plot.png"


if __name__ == "__main__":

    # Initialize solver
    batch_size = 1000
    solver = DLSolver(device="cpu", batch_size=batch_size)  # or "cpu"

    # Supply solver parameters for different problem sizes
    solver.parameter_key = {
        10: {"p": 1.0, "scale": None, "lr": 0.001, "iter": 10000, "nr": 15},
        20: {"p": 2.0, "scale": None, "lr": 0.005, "iter": 15000, "nr": 10},
    }

    results = Results()
    # Load test instances to solve
    test_instances_files = [f for f in glob.glob(TEST_INSTANCES_DIR + "*.in")]
    for instance_file in test_instances_files:
        # Load the problem from the problem file into the instance
        boxqp_instance = ProblemInstance(
            instance_type="test",
            file_path=instance_file,
            device=solver.device,
        )

        boxqp_instance.scale_coefs(
            solver.get_scaling_factor(boxqp_instance.N, boxqp_instance.q)
        )

        # Solve the problem
        # TODO: Explain significance of variables here
        solver_result = solver.solve(
            instance=boxqp_instance,
            post_processor=None,
        )

        # store in results
        results.add_result(
            problem_size=boxqp_instance.N,
            batch_size=solver.batch_size,
            instance_name=boxqp_instance.name,
            c_variables=solver_result["c_variables"],
            objective_value=solver_result["objective_value"],
            solve_time=solver_result["solve_time"],
            pp_time=solver_result["post_processing_time"],
            optimal_value=boxqp_instance.optimal_sol,
            device=solver.device,
        )
        print(vars(results))

    plot_fig = ccvmplotlib.plot_TTS(
        solution_filepath="./results/valid_result.json",
        problem="BoxQP",
        solver=solver,
        TTS_type="wallclock",
        show_plot=True,
    )

    # If PLOT_OUTPUT_DIR not exists, create the path
    if not os.path.isdir(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)
        print("Plot folder doesn't exist yet. Creating: ", PLOT_OUTPUT_DIR)

    # Save to local
    plot_fig.savefig(PLOT_OUTPUT_DEST, format="png")
    print(f"Sucessfully saved the plot to {PLOT_OUTPUT_DEST}")
