import glob
import os
from ccvm.ccvmplotlib import ccvmplotlib
from ccvm.problem_classes.boxqp import ProblemInstance
from ccvm.metadata_list import MetadataList
from ccvm.solvers import DLSolver


METADATA_DIR = "./metadata"
TEST_OUTPUT_DEST = f"{METADATA_DIR}/DL-CCVM_LGFGS_cpu_test.txt"
TEST_INSTANCES_DIR = "./test_instances/"
PLOT_OUTPUT_DIR = "./plots"
PLOT_OUTPUT_DEST = f"{PLOT_OUTPUT_DIR}/DL-CCVM_LBFGS_cpu_plot.png"


if __name__ == "__main__":

    # Initialize solver
    batch_size = 1000
    solver = DLSolver(device="cpu", batch_size=batch_size)  # or "cuda"

    # Supply solver parameters for different problem sizes
    solver.parameter_key = {
        10: {"pump": 1.0, "lr": 0.001, "iterations": 10000, "nr": 15},
        20: {"pump": 2.0, "lr": 0.005, "iterations": 15000, "nr": 10},
    }

    metadata_list = MetadataList()
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
        # Add metadata to list
        metadata_list.add_metadata(solution.get_metadata_dict())

    # Save metadata to file
    metadata_filepath = metadata_list.save_metadata_to_file(METADATA_DIR)

    plot_fig = ccvmplotlib.plot_TTS(
        solution_filepath=metadata_filepath,
        problem="BoxQP",
        TTS_type="wallclock",
        show_plot=True,
    )

    # If PLOT_OUTPUT_DIR not exists, create the path
    if not os.path.isdir(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)
        print("Plot folder doesn't exist yet. Creating: ", PLOT_OUTPUT_DIR)

    # Save to local
    plot_fig.savefig(PLOT_OUTPUT_DEST)
    print(f"Sucessfully saved the plot to {PLOT_OUTPUT_DEST}")
