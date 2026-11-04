import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (pi_theta / pi_theta_old).
        A: List of advantage estimates.
        pi_theta_old: List of old policy probabilities (per-sample, not normalized).
        pi_theta_ref: List of reference policy probabilities (per-sample, not normalized).
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value.
    """
    # Step 0: Convert lists to NumPy arrays
    rhos = np.asarray(rhos, dtype=float)
    A = np.asarray(A, dtype=float)
    pi_theta_old = np.asarray(pi_theta_old, dtype=float)
    pi_theta_ref = np.asarray(pi_theta_ref, dtype=float)

    # Step 1: Clip the probability ratios
    clipped_rhos = np.clip(rhos, 1 - epsilon, 1 + epsilon)

    # Step 2: Compute two surrogate terms
    unclipped = rhos * A
    clipped = clipped_rhos * A

    # Step 3: Mean clipped surrogate objective
    surrogate_objective = np.minimum(unclipped, clipped).mean()

    # Step 4.1: Recover current policy values
    pi_theta = rhos * pi_theta_old

    # Step 4.2: Normalize current policy
    pi_theta = pi_theta / np.sum(pi_theta)

    # Step 4.3: Normalize reference policy
    pi_theta_ref = pi_theta_ref / np.sum(pi_theta_ref)

    # Step 4.4: KL divergence
    kl_divergence = np.sum(
        pi_theta * np.log(pi_theta / pi_theta_ref)
    )

    # Step 5: Final GRPO objective
    objective = surrogate_objective - beta * kl_divergence

    return float(objective)
    # Your code here
    #pass