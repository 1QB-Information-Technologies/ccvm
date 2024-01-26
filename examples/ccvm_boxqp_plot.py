import glob
import os
from ccvm_simulators.ccvmplotlib import ccvmplotlib
from ccvm_simulators.problem_classes.boxqp import ProblemInstance
from ccvm_simulators.metadata_list import MetadataList
from ccvm_simulators.solvers import DLSolver

import matplotlib.pyplot as plt

METADATA_DIR = "./metadata"
TEST_OUTPUT_DEST = f"{METADATA_DIR}/DL-CCVM_LGFGS_cpu_test.txt"
TEST_INSTANCES_DIR = "./test_instances/"
PLOT_OUTPUT_DIR = "./plots"
PLOT_OUTPUT_DEST = f"{PLOT_OUTPUT_DIR}/DL-CCVM_TTS_cpu_plot.png"


if __name__ == "__main__":
    # Initialize solver
    batch_size = 1000
    solver = DLSolver(device="cpu", batch_size=batch_size)  # or "cuda"

    # Supply solver parameters for different problem sizes
    solver.parameter_key = {
        10: {"pump": 1.0, "dt": 0.001, "iterations": 10000, "noise_ratio": 15},
        20: {"pump": 2.0, "dt": 0.005, "iterations": 15000, "noise_ratio": 10},
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

        # Scale the problem's coefficients for more stable optimization
        boxqp_instance.scale_coefs(solver.get_scaling_factor(boxqp_instance.q_matrix))

        # Solve the problem
        solution = solver(
            instance=boxqp_instance,
            post_processor=None,
        )
        # Add metadata to list
        metadata_list.add_metadata(solution.get_metadata_dict())

    # Save metadata to file
    metadata_filepath = metadata_list.save_metadata_to_file(METADATA_DIR)

    # Customize machine_parameters
    machine_parameters = {
        "cpu_power": {20: 5.0, 30: 5.0, 40: 5.0, 50: 5.0, 60: 5.0, 70: 5.0}
    }

    plot_fig, plot_ax = ccvmplotlib.plot_TTS(
        metadata_filepath=metadata_filepath,
        problem="BoxQP",
        machine_time_func=solver.cpu_machine_time(
            machine_parameters=machine_parameters
        ),
    )

    ccvmplotlib.apply_default_tts_styling(plot_fig, plot_ax)  # apply default styling
    plt.show()  # show plot in a new window

    # If PLOT_OUTPUT_DIR not exists, create the path
    if not os.path.isdir(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)
        print("Plot folder doesn't exist yet. Creating: ", PLOT_OUTPUT_DIR)

    # Save to local
    plot_fig.savefig(PLOT_OUTPUT_DEST)
    print(f"Sucessfully saved the plot to {PLOT_OUTPUT_DEST}")
