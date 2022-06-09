"""
Contains useful functions for the Simulations, such as reward functions
or strategies.
"""
from typing import Callable
import numpy as np
from simulations import Reward, Profile, FloatVec, AgentType, State


def ucb_strategy(z: State, t: int) -> int:
    """
    This function implements an UCB strategy, where an agent
    chooses an arm that maximizes the expected.

    Args:
        z: State of the agent at the current step.
        t: time steps since regeneration of the agent.

    Returns:
        index of the chosen arm
    """
    unpulled_arms = (z.wins + z.losses) == 0
    if np.any(unpulled_arms):
        # pull any arm that has not been pulled yet
        return np.random.choice(np.where(unpulled_arms)[0])
    else:
        # else choose arm according to UCB strategy
        norm = (z.wins + z.losses)
        scores = z.wins / norm + np.sqrt(np.log(t) / norm)
        maximizers = np.where(scores == [scores.max()])[0]
        return np.random.choice(maximizers)


def two_arm_beta_type_sampler() -> AgentType:
    """
    Returns a simple way of sampling types for an agent that may choose between
    two arms.

    The type is sampled according to: (theta_1 ⟂ theta_2)
        theta_1 ~ Beta(3,1)
        theta_2 ~ Beta(1,3)
    So arm 1 is vastly preferred
    """
    return np.array([np.random.beta(3, 1), np.random.beta(1, 3)])


def adjusted_two_arm_beta_type_sampler(a1: float) -> Callable[[], AgentType]:
    """
    Returns a simple way of sampling types for agents where the
    total rewards for the arm are adjusted by a constant factor.
    a1 is the factor for arm 1, while a2 = 1 - a1.

    The type is sampled according to: (theta_1 ⟂ theta_2)
        theta_1 ~ a1 * Beta(3,1)
        theta_2 ~ a2 * Beta(1,3)
    For appropriate choice of a1, we should be able to get any population
    profile we want.
    """
    assert 0 <= a1 <= 1
    a2 = 1 - a1
    return lambda: np.array([a1 * np.random.beta(3, 1), a2 * np.random.beta(1, 3)])

def adjusted_multi_arm_arm_beta_type_sampler(thetas: list[float], betas: list[tuple[int, int]]) -> Callable[[], AgentType]:
    """
    Analog to the adjusted two arm beta sampler, generalized to multiple arms and arbitrary beta distributions.
    """
    assert all(0 <= t <= 1 for t in thetas)
    return lambda: np.array([theta * np.random.beta(beta[0], beta[1]) for theta, beta in zip(thetas, betas)])


def two_arm_uniform_type_sampler() -> AgentType:
    """    
    Returns a type sampled uniform on [0,1] for each arm independently.
    """
    return np.random.random(2)


def negative_externality(L: float) -> Reward:
    """
    Returns a simple negative externality reward function parametrized by L.

    Args:
        L: The punishment behaviour. If L is large, the agent is punished more
        for picking a popular option.

    Returns:
        Reward function of the form theta/ (1+ L * f) (^= type / (1 + L * profile)).
    """
    def reward(theta: AgentType, profile: Profile) -> FloatVec:
        return theta / (1 + L * profile)
    return reward


def positive_externality() -> Reward:
    """
    Returns a simple positive externality reward function.

    Returns:
        Reward function of the form theta * profile
    """
    def reward(theta: AgentType, profile: Profile) -> FloatVec:
        return theta * profile
    return reward


def separable_rewards() -> Reward:
    """
    Returns a separable reward function.

    Returns:
        Reward function of the type (theta + profile)/2
    """
    def reward(theta: AgentType, profile: Profile) -> FloatVec:
        return (theta + profile)/2
    return reward
