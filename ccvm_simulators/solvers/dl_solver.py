import torch
import numpy as np
import torch.distributions as tdist
import time
from pandas import DataFrame

from ccvm_simulators.solvers import CCVMSolver
from ccvm_simulators.solvers.algorithms import AdamParameters
from ccvm_simulators.solution import Solution
from ccvm_simulators.post_processor.factory import PostProcessorFactory

DL_SCALING_MULTIPLIER = 0.2
"""The value used by the DLSolver when calculating a scaling value in
super.get_scaling_factor()"""


class DLSolver(CCVMSolver):
    """The DLSolver class models the delay line coherent continuous-variable machine
    (DL-CCVM)."""

    def __init__(
        self,
        device,
        problem_category="boxqp",
        batch_size=1000,
        S=1,
    ):
        """
        Args:
            device (str): The device to use for the solver. Can be "cpu" or "cuda".
            problem_category (str): The category of problem to solve. Can be one of
                "boxqp". Defaults to "boxqp".
            batch_size (int): The number of times to solve a problem instance
                simultaneously. Defaults to 1000.
            S (float or torch.tensor): Enforced saturation value. Defaults to 1.

        Raises:
            ValueError: If the problem category is not supported by the solver.

        Returns:
            DLSolver: The DLSolver object.
        """
        super().__init__(device)
        self.batch_size = batch_size
        self.S = S
        self._default_optics_machine_parameters = {
            "laser_power": 1200e-6,
            "modulators_power": 10e-3,
            "squeezing_power": 180e-3,
            "electronics_power": 0.0,
            "amplifiers_power": 222.2e-3,
            "electronics_latency": 1e-9,
            "laser_clock": 10e-12,
            "postprocessing_power": {
                20: 4.96,
                30: 5.1,
                40: 4.95,
                50: 5.26,
                60: 5.11,
                70: 5.09,
            },
        }
        self._scaling_multiplier = DL_SCALING_MULTIPLIER
        # Use the method selector to choose the problem-specific methods to use
        self._method_selector(problem_category)

    @property
    def parameter_key(self):
        """The parameters that will be used by the solver when solving the problem.

        Note:
            Setting this parameter after calling tune() will overwrite tuned parameters.

        The parameter_key must match the following format:

            * key: problem size (the number of variables in the problem).
            * value: dict with these keys:
                * pump (float),
                * dt (float),
                * iterations (int),
                * noise_ratio (float)

            With values, the parameter key might look like this::

                {
                    20: {"pump": 2.0, "dt": 0.005, "iterations": 15000, "noise_ratio": 10},
                    30: {"pump": 2.0, "dt": 0.005, "iterations": 15000, "noise_ratio": 5},
                }

        Raises:
            ValueError: If the parameter key does not contain the solver-specific
                combination of keys described above.
        """
        return self._parameter_key

    @parameter_key.setter
    def parameter_key(self, parameters):
        expected_dlparameter_key_set = set(
            ["pump", "dt", "iterations", "noise_ratio", "feedback_scale"]
        )
        parameter_key_list = parameters.values()
        # Iterate over the parameters for each given problem size
        for parameter_key in parameter_key_list:
            if parameter_key.keys() != expected_dlparameter_key_set:
                # The parameter key is not valid for this solver
                raise ValueError(
                    "The parameter key is not valid for this solver. Expected keys: "
                    + str(expected_dlparameter_key_set)
                    + " Given keys: "
                    + str(parameter_key.keys())
                )

        # If we get here, the parameter key is valid
        self._parameter_key = parameters
        self._is_tuned = False

    def _calculate_drift_boxqp(
        self, c, s, pump, rate, feedback_scale=100, lower_limit=0, upper_limit=1, S=1
    ):
        """We treat the SDE that simulates the CIM of NTT as drift
        calculation.

        Args:
            c (torch.Tensor): In-phase amplitudes of the solver
            s (torch.Tensor): Quadrature amplitudes of the solver
            pump (float): The maximum pump field strength
            rate (float): The multiplier for the pump field strength at a given instance
            of time.
            lower_limit (float): The lower bound of the box constraints. Defaults to 0.
            upper_limit (float): The upper bound of the box constraints. Defaults to 1.
            S (float): The saturation value of the amplitudes. Defaults to 1.

        Returns:
            tuple: The calculated change in the variable amplitudes.
        """

        c_pow = torch.pow(c, 2)
        s_pow = torch.pow(s, 2)

        if pump > 1:
            S = np.sqrt(pump - 1)

        c_grad_1 = (
            0.25
            * torch.einsum(
                "bi,ij -> bj",
                c * (upper_limit - lower_limit) / S + (upper_limit + lower_limit),
                self.q_matrix,
            )
            * (upper_limit - lower_limit)
            / S
        )
        c_grad_2 = torch.einsum("cj,cj -> cj", -1 + (pump * rate) - c_pow - s_pow, c)
        c_grad_3 = self.v_vector * (upper_limit - lower_limit) / (2 * S)

        s_grad_1 = (
            0.25
            * torch.einsum(
                "bi,ij -> bj",
                s * (upper_limit - lower_limit) / S + (upper_limit + lower_limit),
                self.q_matrix,
            )
            * (upper_limit - lower_limit)
            / S
        )
        s_grad_2 = torch.einsum("cj,cj -> cj", -1 - (pump * rate) - c_pow - s_pow, s)
        s_grad_3 = self.v_vector * (upper_limit - lower_limit) / (2 * S)

        feedback_scale_dynamic = feedback_scale * (0.5 + rate)
        c_drift = -feedback_scale_dynamic * (c_grad_1 + c_grad_3) + c_grad_2
        s_drift = -feedback_scale_dynamic * (s_grad_1 + s_grad_3) + s_grad_2
        return c_drift, s_drift

    def _calculate_grads_boxqp(self, c, s, lower_limit=0, upper_limit=1, S=1):
        """We treat the SDE that simulates the CIM of NTT as gradient
        calculation. Original SDE considers only quadratic part of the objective
        function. Therefore, we need to modify and add linear part of the QP to
        the SDE.

        Args:
            c (torch.Tensor): In-phase amplitudes of the solver
            s (torch.Tensor): Quadrature amplitudes of the solver
            lower_limit (float): The lower bound of the box constraints. Defaults to 0.
            upper_limit (float): The upper bound of the box constraints. Defaults to 1.
            S (float): The saturation value of the amplitudes. Defaults to 1.

        Returns:
            tuple: The calculated change in the variable amplitudes.
        """

        c_grad_1 = (
            0.25
            * torch.einsum(
                "bi,ij -> bj",
                c * (upper_limit - lower_limit) / S + (upper_limit + lower_limit),
                self.q_matrix,
            )
            * (upper_limit - lower_limit)
            / S
        )
        c_grad_3 = self.v_vector * (upper_limit - lower_limit) / (2 * S)

        s_grad_1 = (
            0.25
            * torch.einsum(
                "bi,ij -> bj",
                s * (upper_limit - lower_limit) / S + (upper_limit + lower_limit),
                self.q_matrix,
            )
            * (upper_limit - lower_limit)
            / S
        )
        s_grad_3 = self.v_vector * (upper_limit - lower_limit) / (2 * S)

        c_grads = -c_grad_1 - c_grad_3
        s_grads = -s_grad_1 - s_grad_3
        return c_grads, s_grads

    def _change_variables_boxqp(
        self, problem_variables, lower_limit=0, upper_limit=1, S=1
    ):
        """Perform a change of variables to enforce the box constraints.

        Args:
            problem_variables (torch.Tensor): The variables to change.
            lower_limit (float): The lower bound of the box constraints. Defaults to 0.
            upper_limit (float): The upper bound of the box constraints. Defaults to 1.
            S (float or torch.tensor): The enforced saturation value. Defaults to 1

        Returns:
            torch.Tensor: The changed variables.
        """
        return 0.5 * problem_variables / S * (upper_limit - lower_limit) + 0.5 * (
            upper_limit + lower_limit
        )

    def _fit_to_constraints_boxqp(self, c, lower_clamp, upper_clamp):
        """Clamps the values of c to be within the box constraints

        Args:
            c (torch.Tensor): The variables to clamp.
            lower_clamp (float or torch.tensor): The lower bound of the box constraints.
            upper_clamp (float or torch.tensor): The upper bound of the box constraints.

        Returns:
            torch.Tensor: The clamped variables.
        """

        c_clamped = torch.clamp(c, lower_clamp, upper_clamp)
        return c_clamped

    def _append_samples_to_file(self, c_sample, s_sample, evolution_file_object):
        """Saves samples of the amplitude values to a file.
        The end file will contain the values of the c_sample followed by the s_sample.
        Each line corresponds to a row in the tensor, with tab-delineated values.

        Args:
            c_sample (torch.Tensor): The sample of in-phase amplitudes to add to the
                file. Expected Dimensions: problem_size x num_samples.
            s_sample (torch.Tensor): The sample of quadrature amplitudes to add to the
                file. Expected Dimensions: problem_size x num_samples.
            evolution_file_object (io.TextIOWrapper): The file object of the file to
                save the samples to.
        """
        # Save the c samples to the file
        c_rows = c_sample.shape[0]  # problem_size
        c_columns = c_sample.shape[1]  # num_samples
        for nn in range(c_rows):
            for ii in range(c_columns):
                evolution_file_object.write(str(round(c_sample[nn, ii].item(), 4)))
                evolution_file_object.write("\t")
            evolution_file_object.write("\n")

        # Save the s samples to the file
        s_rows = s_sample.shape[0]  # problem_size
        s_columns = s_sample.shape[1]  # num_samples
        for nn in range(s_rows):
            for ii in range(s_columns):
                evolution_file_object.write(str(round(s_sample[nn, ii].item(), 4)))
                evolution_file_object.write("\t")
            evolution_file_object.write("\n")

    def _is_valid_optics_machine_parameters(self, machine_parameters):
        """Validates that the given optics machine parameters are valid for this solver.

        Args:
            machine_parameters (dict): The machine parameters to validate.

        Raises:
            ValueError: If the given machine parameters are invalid.
        """

        required_keys = [
            "laser_power",
            "modulators_power",
            "squeezing_power",
            "electronics_power",
            "amplifiers_power",
            "electronics_latency",
            "laser_clock",
            "postprocessing_power",
        ]

        # Check that all required keys are present
        missing_keys = [key for key in required_keys if key not in machine_parameters]

        if missing_keys:
            raise ValueError(
                f"Invalid optics_machine_parameters: Missing required keys - {missing_keys}"
            )

    def tune(self, instances, post_processor=None, pump_rate_flag=True, g=0.05):
        """Determines the best parameters for the solver to use by adjusting each
        parameter over a number of iterations on the problems in the given set of
        problems instances. The `parameter_key` attribute of the solver will be
        updated with the best parameters found.

        Args:
            instances (list): A list of problem instances to tune the solver on.
            post_processor (str): The name of the post processor to use to process the
                results of the solver. None if no post processing is desired.
                Defaults to None.
            pump_rate_flag (bool): Whether or not to scale the pump rate based on the
            iteration number. If False, the pump rate will be 1.0. Defaults to True.
            g (float): The nonlinearity coefficient. Defaults to 0.05.
        """
        # TODO: This implementation is a placeholder; full implementation is a
        #       future consideration
        self.is_tuned = True

    def _optics_machine_energy(self, machine_parameters=None):
        """The wrapper function of calculating the average energy consumption of the
            solver, as if the solving process was to be performed on an optical DL-CCVM
            machine.

        Args:
            machine_parameters (dict, optional): Parameters of the optical DL-CCVM
                machine. Defaults to None.

        Raises:
            ValueError: when the given machine parameters are not valid.
            ValueError: when the given dataframe does not contain the required columns.

        Returns:
            Callable: A callable function that takes in a dataframe and problem size and
                returns the average energy consumption of the solver.
        """

        if machine_parameters is None:
            machine_parameters = self._default_optics_machine_parameters
        else:
            self._is_valid_optics_machine_parameters(machine_parameters)

        def _optics_machine_energy_callable(dataframe: DataFrame, problem_size: int):
            """Calculate the average energy consumption of the solver simulating on a
                DL-CCVM machine.

            Args:
                dataframe (DataFrame): The necessary data to calculate the average
                    energy.
                problem_size (int): The size of the problem.

            Raises:
                ValueError: when the given dataframe does not contain the required
                    columns.

            Returns:
                float: The average energy consumption of the solver.
            """
            self._validate_machine_energy_dataframe_columns(dataframe)

            try:
                pump = self.parameter_key[problem_size]["pump"]
            except KeyError as e:
                raise KeyError(
                    f"Pump for the given instance size: {problem_size} is not defined."
                )

            T_clock = machine_parameters["laser_clock"]
            P_opt = machine_parameters["laser_power"]
            T_elec = machine_parameters["electronics_latency"]
            P_mod = machine_parameters["modulators_power"]
            P_sq = machine_parameters["squeezing_power"]
            P_elec = machine_parameters["electronics_power"]
            P_opa = machine_parameters["amplifiers_power"]
            postprocessing_time = np.mean(dataframe["pp_time"].values)
            iterations = np.mean(dataframe["iterations"].values)
            optics_energy = (
                pump * P_opt * T_elec
                + pump * P_opt * T_clock * float(problem_size)
                + 2 * P_mod * T_clock * float(problem_size) * (float(problem_size) - 1)
                + P_sq * T_elec
                + P_sq * T_clock * float(problem_size)
                + P_elec * T_elec
                + P_elec * T_clock * float(problem_size)
                + P_opa * T_elec * (float(problem_size) - 1)
                + P_opa * T_clock * float(problem_size) * (float(problem_size) - 1)
            ) * iterations
            postprocessing_energy = (
                machine_parameters["postprocessing_power"][problem_size]
                * postprocessing_time
            )
            machine_energy = optics_energy + postprocessing_energy
            return machine_energy

        return _optics_machine_energy_callable

    def _optics_machine_time(self, machine_parameters: dict = None):
        """The wrapper function of calculating the average time spent by the
            solver on a single instance, as if the solving process was to be performed on
            an optical DL-CCVM machine.

        Args:
            machine_parameters (dict, optional): Parameters of the optical DL-CCVM
                machine. Defaults to None.

        Raises:
            ValueError: when the given machine parameters are not valid.
            ValueError: when the given dataframe does not contain the required columns.

        Returns:
            Callable: A callable function that takes in a dataframe and problem size and
                returns the average average time spent by the solver on a single instance.
        """

        if machine_parameters is None:
            machine_parameters = self._default_optics_machine_parameters
        else:
            self._is_valid_optics_machine_parameters(machine_parameters)

        def _optics_machine_time_callable(dataframe: DataFrame, problem_size: int):
            """Calculate the average average time spent by the solver on a single instance,
                simulating on a DL-CCVM machine.

            Args:
                dataframe (DataFrame): The necessary data to calculate the average
                    time.
                problem_size (int): The size of the problem.

            Raises:
                ValueError: when the given dataframe does not contain the required
                    columns.

            Returns:
                float: The average average time spent by the solver on a single instance.
            """
            try:
                iterations = np.mean(dataframe["iterations"].values)
                postprocessing_time = np.mean(dataframe["pp_time"].values)
            except KeyError as e:
                missing_column = e.args[0]
                raise KeyError(
                    f"The given dataframe is missing the {missing_column} "
                    f"column. Required columns are: ['iterations', 'pp_time']."
                )

            # Machine parameters are pre-validated in the wrapper, so this is safe
            laser_clock = machine_parameters["laser_clock"]

            machine_time = (
                float(problem_size) * laser_clock * iterations + postprocessing_time
            )

            return machine_time

        return _optics_machine_time_callable

    def _solve(
        self,
        problem_size,
        batch_size,
        device,
        S,
        pump,
        dt,
        iterations,
        noise_ratio,
        feedback_scale,
        pump_rate_flag,
        g,
        evolution_step_size,
        samples_taken,
    ):
        """Solves the given problem instance using the original DL-CCVM solver.

        Args:
            problem_size (int): instance size.
            batch_size (int): The number of times to solve a problem instance
            device (str): The device to use for the solver. Can be "cpu" or "cuda".
            S (float): Saturation bound.
            dt (float): Simulation time step.
            pump (float): Pump field strength.
            iterations (int): number of steps.
            noise_ratio (float): noise ratio.
            pump_rate_flag (bool): Whether or not to scale the pump rate based on the
                iteration number.
            g (float): The nonlinearity coefficient.
            evolution_step_size (int): If set, the c/s values will be sampled once
                per number of iterations equivalent to the value of this variable.
                At the end of the solve process, the best batch of sampled values
                will be written to a file that can be specified by setting the
                evolution_file parameter.
            samples_taken (int): sample slice.

        Returns:
            c, s (tensor): random variables
        """
        # Initialize tensor variables on the device that will be used to perform the
        # calculations
        c = torch.zeros((batch_size, problem_size), dtype=torch.float, device=device)
        s = torch.zeros((batch_size, problem_size), dtype=torch.float, device=device)
        wiener_dist_c = tdist.Normal(
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([1.0] * batch_size, device=device),
        )
        wiener_dist_s = tdist.Normal(
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([1.0] * batch_size, device=device),
        )

        # Perform the solve over the specified number of iterations
        pump_rate = 1
        for i in range(iterations):
            if pump_rate_flag:
                pump_rate = (i + 1) / iterations

            noise_ratio_i = (noise_ratio - 1) * np.exp(-(i + 1) / iterations * 3) + 1

            c_drift, s_drift = self.calculate_drift(
                c,
                s,
                pump,
                pump_rate,
                feedback_scale,
                self.solution_bounds[0],
                self.solution_bounds[1],
            )
            wiener_increment_c = (
                wiener_dist_c.sample((problem_size,)).transpose(0, 1)
                * np.sqrt(dt)
                * noise_ratio_i
            )
            wiener_increment_s = (
                wiener_dist_s.sample((problem_size,)).transpose(0, 1)
                * np.sqrt(dt)
                / noise_ratio_i
            )

            diff = 2 * g * torch.sqrt(c**2 + s**2 + 0.5)

            c += dt * c_drift + diff * wiener_increment_c

            s += dt * s_drift + diff * wiener_increment_s

            # If evolution_step_size is specified, save the values if this iteration
            # aligns with the step size or if this is the last iteration
            if evolution_step_size and (
                i % evolution_step_size == 0 or i + 1 >= iterations
            ):
                # Update the record of the sample values with the values found at
                # this iteration
                self.c_sample[:, :, samples_taken] = c
                self.s_sample[:, :, samples_taken] = s
                samples_taken += 1

        # Ensure variables are within any problem constraints
        c = self.fit_to_constraints(c, -S, S)

        return c, s

    def _solve_adam(
        self,
        problem_size,
        batch_size,
        device,
        S,
        pump,
        dt,
        iterations,
        noise_ratio,
        pump_rate_flag,
        g,
        evolution_step_size,
        samples_taken,
        hyperparameters,
    ):
        """Solves the given problem instance using the DL-CCVM solver with Adam
            algorithm.

        Args:
            problem_size (int): instance size.
            batch_size (int): The number of times to solve a problem instance
            device (str): The device to use for the solver. Can be "cpu" or "cuda".
            S (float): Saturation bound.
            pump (float): Pump field strength.
            dt (float): Simulation time step.
            iterations (int): number of steps.
            noise_ratio (float): noise ratio.
            pump_rate_flag (bool): Whether or not to scale the pump rate based on the
                iteration number.
            g (float): The nonlinearity coefficient.
            evolution_step_size (int): If set, the c/s values will be sampled once
                per number of iterations equivalent to the value of this variable.
                At the end of the solve process, the best batch of sampled values
                will be written to a file that can be specified by setting the
                evolution_file parameter.
            samples_taken (int): sample slice.
            hyperparameters (dict): Hyperparameters for Adam algorithm.

        Returns:
            c, s (tensor): random variables
        """
        # Initialize tensor variables on the device that will be used to perform the
        # calculations
        c = torch.zeros((batch_size, problem_size), dtype=torch.float, device=device)
        s = torch.zeros((batch_size, problem_size), dtype=torch.float, device=device)
        wiener_dist_c = tdist.Normal(
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([1.0] * batch_size, device=device),
        )
        wiener_dist_s = tdist.Normal(
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([1.0] * batch_size, device=device),
        )

        # Pump rate update selection: time-dependent or constant
        pump_rate_i = lambda i: pump * (i + 1) / iterations
        pump_rate_c = lambda i: pump  # Constant
        if pump_rate_flag:
            calc_pump_rate = pump_rate_i
        else:
            calc_pump_rate = pump_rate_c

        if pump > 1:
            S = np.sqrt(pump - 1)

        alpha = hyperparameters["alpha"]
        beta1 = hyperparameters["beta1"]
        beta2 = hyperparameters["beta2"]
        epsilon = 1e-8

        # Compute bias corrected grads using 1st and 2nd moments
        # with element-wise division
        def update_grads_with_moment2_assign(gradc, grads, mhatc, vhatc, mhats, vhats):
            return (
                alpha * torch.div(mhatc, torch.sqrt(vhatc) + epsilon),
                alpha * torch.div(mhats, torch.sqrt(vhats) + epsilon),
            )

        def update_grads_with_moment2_addassign(
            gradc, grads, mhatc, vhatc, mhats, vhats
        ):
            return (
                gradc + alpha * torch.div(mhatc, torch.sqrt(vhatc) + epsilon),
                grads + alpha * torch.div(mhats, torch.sqrt(vhats) + epsilon),
            )

        # Compute bias corrected grads using only 1st moment
        def update_grads_without_moment2_assign(gradc, grads, mhatc, mhats):
            return (alpha * mhatc, alpha * mhats)

        def update_grads_without_moment2_addassign(gradc, grads, mhatc, mhats):
            return (gradc + alpha * mhatc, grads + alpha * mhats)

        # Choose desired update method.
        if hyperparameters["add_assign"]:
            update_grads_with_moment2 = update_grads_with_moment2_addassign
            update_grads_without_moment2 = update_grads_without_moment2_addassign
        else:
            update_grads_with_moment2 = update_grads_with_moment2_assign
            update_grads_without_moment2 = update_grads_without_moment2_assign

        # Initialize first moment vectors for c and s
        m_c = torch.zeros((batch_size, problem_size), dtype=torch.float, device=device)
        m_s = torch.zeros((batch_size, problem_size), dtype=torch.float, device=device)
        # Initialize second moment vectors conditionally
        if not beta2 == 1.0:
            v_c = torch.zeros(
                (batch_size, problem_size), dtype=torch.float, device=device
            )
            v_s = torch.zeros(
                (batch_size, problem_size), dtype=torch.float, device=device
            )
        else:
            v_c = None
            v_s = None

        # Perform the solve with Adam over the specified number of iterations
        for i in range(iterations):
            pump_rate = calc_pump_rate(i)

            noise_ratio_i = (noise_ratio - 1) * np.exp(-(i + 1) / iterations * 3) + 1

            # Calculate gradient
            c_grads, s_grads = self.calculate_grads(
                c, s, self.solution_bounds[0], self.solution_bounds[1], S
            )

            # Update biased first moment estimate
            m_c = beta1 * m_c + (1.0 - beta1) * c_grads
            m_s = beta1 * m_s + (1.0 - beta1) * s_grads

            # Compute bias correction in 1st moment
            beta1i = 1.0 - beta1 ** (i + 1)
            mhat_c = m_c / beta1i
            mhat_s = m_s / beta1i

            # Conditional second moment estimation
            if not beta2 == 1.0:
                # Update biased 2nd moment estimate
                v_c = beta2 * v_c + (1.0 - beta2) * torch.pow(c_grads, 2)
                v_s = beta2 * v_s + (1.0 - beta2) * torch.pow(s_grads, 2)

                # Compute bias correction in 2nd moment
                beta2i = 1.0 - beta2 ** (i + 1)
                vhat_c = v_c / beta2i
                vhat_s = v_s / beta2i

                # Compute bias corrected grads
                c_grads, s_grads = update_grads_with_moment2(
                    c_grads, s_grads, mhat_c, vhat_c, mhat_s, vhat_s
                )
            else:
                # Compute bias corrected grads only with 1st moment
                c_grads, s_grads = update_grads_without_moment2(
                    c_grads, s_grads, mhat_c, mhat_s
                )

            # Calculate drift and diffusion terms of dl-ccvm
            c_pow = torch.pow(c, 2)
            s_pow = torch.pow(s, 2)
            c_drift = torch.einsum("cj,cj -> cj", -1 + pump_rate - c_pow - s_pow, c)
            s_drift = torch.einsum("cj,cj -> cj", -1 - pump_rate - c_pow - s_pow, s)

            wiener_increment_c = (
                wiener_dist_c.sample((problem_size,)).transpose(0, 1)
                * np.sqrt(dt)
                * noise_ratio_i
            )
            wiener_increment_s = (
                wiener_dist_s.sample((problem_size,)).transpose(0, 1)
                * np.sqrt(dt)
                / noise_ratio_i
            )

            c += (
                dt * (c_drift + c_grads)
                + 2 * g * torch.sqrt(c_pow + s_pow + 0.5) * wiener_increment_c
            )
            s += (
                dt * (s_drift + s_grads)
                + 2 * g * torch.sqrt(c_pow + s_pow + 0.5) * wiener_increment_s
            )

            # If evolution_step_size is specified, save the values if this iteration
            # aligns with the step size or if this is the last iteration
            if evolution_step_size and (
                i % evolution_step_size == 0 or i + 1 >= iterations
            ):
                # Update the record of the sample values with the values found at
                # this iteration
                self.c_sample[:, :, samples_taken] = c
                self.s_sample[:, :, samples_taken] = s
                samples_taken += 1

        # Ensure variables are within any problem constraints
        c = self.fit_to_constraints(c, -S, S)

        return c, s

    def __call__(
        self,
        instance,
        post_processor=None,
        pump_rate_flag=True,
        g=0.05,
        evolution_step_size=None,
        evolution_file=None,
        algorithm_parameters=None,
    ):
        """Solves the given problem instance choosing one of the available DL-CCVM
            solvers.

        Args:
            instance (ProblemInstance): The problem instance to solve.
            post_processor (str): The name of the post processor to use to process the
                results of the solver. None if no post processing is desired. Defaults
                to None.
            pump_rate_flag (bool): Whether or not to scale the pump rate based on the
            iteration number. If False, the pump rate will be 1.0. Defaults to True.
            g (float): The nonlinearity coefficient. Defaults to 0.05.
            evolution_step_size (int): If set, the c/s values will be sampled once
                per number of iterations equivalent to the value of this variable.
                At the end of the solve process, the best batch of sampled values
                will be written to a file that can be specified by setting the
                evolution_file parameter. Defaults to None, meaning no problem variables
                will be written to the file.
            evolution_file (str): The file to save the best set of c/s samples to.
                Only revelant when evolution_step_size is set.
                If a file already exists with the same name, it will be overwritten.
                Defaults to None, which generates a filename based on the problem
                instance name.
            algorithm_parameters (None, AdamParameters): Specify for the solver to use a
                specialized algorithm by passing in an instance of the algorithm's
                parameters class. Options include: AdamParameters.
                Defaults to None, which uses the original DL solver.

        Returns:
            solution (Solution): The solution to the problem instance.
        """
        # If the instance and the solver don't specify the same device type, raise
        # an error
        if instance.device != self.device:
            raise ValueError(
                f"The device type of the instance ({instance.device}) and the solver"
                f" ({self.device}) must match."
            )

        # Get problem from problem instance
        problem_size = instance.problem_size
        self.q_matrix = instance.q_matrix
        self.v_vector = instance.v_vector
        self.solution_bounds = instance.solution_bounds

        # Get solver setup variables
        S = self.S
        batch_size = self.batch_size
        device = self.device

        # Get parameters from parameter_key
        try:
            pump = self.parameter_key[problem_size]["pump"]
            dt = self.parameter_key[problem_size]["dt"]
            iterations = self.parameter_key[problem_size]["iterations"]
            noise_ratio = self.parameter_key[problem_size]["noise_ratio"]
            feedback_scale = self.parameter_key[problem_size]["feedback_scale"]
        except KeyError as e:
            raise KeyError(
                f"The parameter '{e.args[0]}' for the given instance size is not defined."
            ) from e

        # If S is a 1-D tensor, convert it to to a 2-D tensor
        if torch.is_tensor(S) and S.ndim == 1:
            # Dimension indexing in pytorch starts at 0
            if S.size(dim=0) == problem_size:
                S = torch.outer(torch.ones(batch_size), S)
            else:
                raise ValueError("Tensor S size should be equal to problem size.")

        # Start the timer for the solve
        solve_time_start = time.time()

        samples_taken = None
        self.c_sample = None
        self.s_sample = None
        if evolution_step_size:
            # Check that the value is valid
            if evolution_step_size < 1:
                raise ValueError(
                    f"The evolution step size must be greater than or equal to 1."
                )
            # Generate evolution file name
            if evolution_file is None:
                evolution_file = f"./{instance.name}_evolution.txt"

            # Get the number of samples to save
            # Find the number of full steps that will be taken
            num_steps = int(iterations / evolution_step_size)
            # We will also capture the first iteration through
            num_samples = num_steps + 1
            # And capture the last iteration if the step size doesn't evenly divide
            if iterations % evolution_step_size != 0:
                num_samples += 1

            # Initialize tensors
            # Store on CPU to keep the memory usage lower on the GPU
            self.c_sample = torch.zeros(
                (batch_size, problem_size, num_samples),
                dtype=torch.float,
                device="cpu",
            )
            self.s_sample = torch.zeros(
                (batch_size, problem_size, num_samples),
                dtype=torch.float,
                device="cpu",
            )
            samples_taken = 0

        if algorithm_parameters is None:
            # Use the original DL solver
            c, s = self._solve(
                problem_size,
                batch_size,
                device,
                S,
                pump,
                dt,
                iterations,
                noise_ratio,
                feedback_scale,
                pump_rate_flag,
                g,
                evolution_step_size,
                samples_taken,
            )
        elif isinstance(algorithm_parameters, AdamParameters):
            # Use the DL solver with the Adam algorithm
            c, s = self._solve_adam(
                problem_size,
                batch_size,
                device,
                S,
                pump,
                dt,
                iterations,
                noise_ratio,
                feedback_scale,
                pump_rate_flag,
                g,
                evolution_step_size,
                samples_taken,
                algorithm_parameters.to_dict(),
            )
        else:
            raise ValueError(
                f"Solver option type {type(algorithm_parameters)} is not supported."
            )

        # Stop the timer for the solve to compute the solution time for solving an instance once
        # Due to the division by batch_size, the solve_time improves for larger batches
        # when the solver is run on GPU. This is expected since GPU is hardware specifically
        # deployed to improve the solution time of solving one single instance by using parallelization
        solve_time = (time.time() - solve_time_start) / batch_size

        # Run the post processor on the results, if specified
        if post_processor:
            post_processor_object = PostProcessorFactory.create_postprocessor(
                post_processor
            )

            problem_variables = post_processor_object.postprocess(
                self.change_variables(
                    c, self.solution_bounds[0], self.solution_bounds[1], S
                ),
                self.q_matrix,
                self.v_vector,
            )
            # Post-processing time for solving an instance once
            pp_time = post_processor_object.pp_time / batch_size
        else:
            problem_variables = c
            pp_time = 0.0

        # Calculate the objective value
        # Perform a change of variables to enforce the box constraints
        confs = self.change_variables(
            problem_variables, self.solution_bounds[0], self.solution_bounds[1], S
        )
        objval = instance.compute_energy(confs)

        if evolution_step_size:
            # Write samples to file
            # Overwrite file if it exists
            open(evolution_file, "w")

            # Get the indices of the best objective values over the sampled iterations
            # to use to get and save the best sampled values of c and s
            batch_index = torch.argmax(-objval)
            with open(evolution_file, "a") as evolution_file_obj:
                self._append_samples_to_file(
                    c_sample=self.c_sample[batch_index],
                    s_sample=self.s_sample[batch_index],
                    evolution_file_object=evolution_file_obj,
                )

        solution = Solution(
            problem_size=instance.problem_size,
            batch_size=batch_size,
            instance_name=instance.name,
            iterations=iterations,
            objective_values=objval,
            solve_time=solve_time,
            pp_time=pp_time,
            optimal_value=instance.optimal_sol,
            best_value=instance.best_sol,
            num_frac_values=instance.num_frac_values,
            solution_vector=instance.solution_vector,
            variables={
                "problem_variables": problem_variables,
                "s": s,
            },
            device=device,
        )

        # Add evolution filename to solution if it was generated
        if evolution_step_size:
            solution.evolution_file = evolution_file

        return solution
