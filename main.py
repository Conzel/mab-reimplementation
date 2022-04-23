import numpy as np
import numpy.typing as npt
from typing import Callable, Any, Optional


class State():
    """
    Represents that state of an agent.
    """

    def __init__(self, n: int):
        self.state = np.zeros(2*n)

    @property
    def wins(self):
        return self.state[::2]

    @property
    def losses(self):
        return self.state[1::2]

    def add_result(self, arm: int, won: bool):
        """
        Adds result to the current state. Modifies state in-place.

        Args:
            arm: Index of the arm that has been played
            won: True if won, False otherwise
        """
        if won:
            self.state[arm * 2] += 1
        else:
            self.state[arm * 2 + 1] += 1


FloatVec = npt.NDArray[np.floating]
IntVec = npt.NDArray[np.integer]

"""Reward chance per arm for agent."""
AgentType = FloatVec
""" % of agents playing an arm. """
Profile = FloatVec
""" Determines probabilities of agent picking an arm. """
Strategy = Callable[[State, int], int]
""" Determines probability of agent getting a reward from arm, given a profile."""
Reward = Callable[[AgentType, Profile], FloatVec]


class Agent():
    """
    This class represents an agent operating in the Multi-Armed Bandit setting.
    Notational explanations:
        n represents the number of arms.
        S represents the set of possible arms.

    The state is represented by two vectors, wins and losses, for easier handling.

    Attributes:
        type: Random variable in [0,1]^n, also referred to as theta. This
            models the reward of the agent on each of the arms. This variable
            is unknown to the agent when acting.
        state: A vector in Z^2n+, representing the number of wins (even) and losses
            (odd) the agent has obtained.
        policy: Function form Z^2n+ -> S, indicating the strategy the agent uses.
        lifetime: The number of steps the agent has been alive for.
    """

    def __init__(self, n: int, strategy: Strategy, type: AgentType) -> None:
        """
        Initializes an agent.
        """
        self.lifetime = 0
        self.has_chosen = False
        self.chosen: Optional[int] = None
        self.state = State(n)
        self.strategy = strategy
        self.type = type

    def choose(self) -> int:
        """
        Returns the chosen arm of the agent at the current time step.
        """
        if self.has_chosen:
            raise RuntimeError(
                "Agent has already chosen an arm. Update agent's state before choosing again.")
        choice = self.strategy(self.state, self.lifetime)
        self.has_chosen = True
        self.chosen = choice
        return choice

    def update(self, won: bool) -> None:
        """
        Updates the agent's state with the reward from the arm,
        and increments the lifetime of the agent.
        """
        if not self.has_chosen:
            raise RuntimeError(
                "Agent has not chosen an arm yet.")
        assert self.chosen is not None
        self.state.add_result(self.chosen, won)
        self.lifetime += 1
        self.has_chosen = False
        self.chosen = None


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


class Simulation():
    """
    Models the simulation of the agents.
    """

    def __init__(self, n: int, m: int, type_sampler: Callable[[], AgentType], reward: Reward, beta: float, strategy: Strategy = ucb_strategy):
        """
        Initializes the simulation.

        Args:
            n: The number of arms.
            m: The number of agents.
            beta: Probability that agent regenerates in each timestep
        """
        self.n = n
        self.m = m
        self.reward = reward
        self.beta = beta
        self.sample_type = type_sampler
        self.strategy = strategy
        # call after all instance variables have been initialized
        self.agents = [self.gen_agent() for _ in range(m)]
        self.t = 0

    def gen_agent(self) -> Agent:
        """
        Samples a fresh agent from the distribution.
        """
        return Agent(self.n, self.strategy, self.sample_type())

    def step(self) -> Profile:
        """
        Runs one step of the simulation.
        This proceeds by the following steps:
        1.  Each agent chooses an arm according to their strategy.
        2.  A reward is determined for each agent according to the
            reward function and the population profile determined in 1
        3.  The agents update their state.
        4.  The agent is checked for regeneration.
        5.  Timestep t is updated

        Returns:
            The population profile of the agents.
        """
        # Step 1
        choices = [a.choose() for a in self.agents]
        print(choices)
        agents_per_arm = np.bincount(choices, minlength=self.n)
        assert isinstance(agents_per_arm, np.ndarray)
        profile = agents_per_arm / self.m
        # Steps 2-4
        for i, a in enumerate(self.agents):
            # Step 2
            reward = self.reward(a.type, profile)
            print(reward)
            # Step 3
            a.update(np.random.random() < reward[choices[i]])
            # Step 4
            if np.random.random() > self.beta:
                self.agents[i] = self.gen_agent()
        self.t += 1
        return profile


def s():
    return Simulation(2, 10, two_arm_beta_type_sampler,
                      negative_externality(1), 0.95)
