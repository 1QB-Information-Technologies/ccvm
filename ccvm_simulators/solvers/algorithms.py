class AdamParameters:
  """Validates and stores the parameters required for the Adam algorithm."""
  def __init__(self, alpha=0.1, beta1=0.9, beta2=0.999):
    """
    Initializes the AdamParameters object with the given hyperparameters.
    Please refer to [the paper](https://doi.org/10.48550/arXiv.1412.6980)
    for more information about the Adam algorithm and its hyperparameters.

    Args:
        alpha (float, optional): The step size, must be positive. Defaults to 0.1.
        beta1 (float, optional): Exponential decay rate for the first moment estimates,
          must be between 0 and 1. Defaults to 0.9.
        beta2 (float, optional): Exponential decay rate for the second moment estimates,
          must be between 0 and 1. Defaults to 0.999.
    """
    if alpha < 0.0:
        raise ValueError(f"AdamAlgorithm: Invalid `alpha` value: {alpha}")
    else:
        self.alpha = alpha

    if beta1 <= 0 or 1 <= beta1:
        raise ValueError(f"AdamAlgorithm: Invalid `beta1` value: {beta1}")
    else:
        self.beta1 = beta1

    if beta2 <= 0 or 1 < beta2:
        raise ValueError(f"AdamAlgorithm: Invalid `beta2` value: {beta2}")
    else:
        self.beta2 = beta2

  def to_dict(self):
    """Returns the parameters as a dictionary."""
    return {
        "alpha": self.alpha,
        "beta1": self.beta1,
        "beta2": self.beta2,
    }
