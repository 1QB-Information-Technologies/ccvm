from ccvm.solvers.mf_solver import MFSolver

class BMFSolver(MFSolver):
    """Constructor method
    """
    def __init__(self):
        super().__init__()
        self._scaling_multiplier = 0.1

    def _validate_parameters(self, parameters):
        """Validate the parameter key against the expected parameters for this solver.

        :param parameters: The set of parameters that will be used by the solver when solving problems.
        :type parameters: dict
        :return: True if the given parameter key is valid for this solver
        :rtype: bool
        """
        # TODO: Add description of the parameters to the docstring
        # TODO: Implementation
        return True

    def calculate_grads(self, mu, mu_tilde, sigma, q_matrix, c_vector, pump, Wt, j):
        """_summary_

        :param mu: _description_
        :type mu: torch.Tensor
        :param mu_tilde: _description_
        :type mu_tilde: torch.Tensor
        :param sigma: _description_
        :type sigma: torch.Tensor
        :param q_matrix: _description_
        :type q_matrix: torch.Tensor
        :param c_vector: _description_
        :type c_vector: torch.Tensor
        :param pump: _description_
        :type pump: float
        :param Wt: _description_
        :type Wt: torch.Tensor
        :param j: _description_
        :type j: float
        :return: _description_
        :rtype: torch.Tensor
        """
        # TODO: summary/descriptions
        # TODO: Implementation
        pass

    def tune(self, instances, scaling_val, post_processor, feedback_scale):
        """_summary_

        :param instances: _description_
        :type instances: List[ccvm.problem.ProblemInstance]
        :param scaling_val: _description_
        :type scaling_val: float
        :param post_processor: _description_
        :type post_processor: PostProcessorType
        :param feedback_scale: _description_
        :type feedback_scale: float
        """
        # TODO: summary/descriptions
        # TODO: This implementation is a placeholder; full implementation is a
        #       future consideration
        self.is_tuned = True

    def solve(self, instance, scaling_val, post_processor, feedback_scale):
        """_summary_

        :param instance: _description_
        :type instance: ccvm.problem.ProblemInstance
        :param scaling_val: _description_
        :type scaling_val: float
        :param post_processor: _description_
        :type post_processor: PostProcessorType
        :param feedback_scale: _description_
        :type feedback_scale: float
        """
        # TODO: summary/descriptions
        # TODO: Implementation
        pass
