"""
Contains classes describing the agents and the environment
in the MAB paper.
"""
import numpy as np
import numpy.typing as npt
from typing import Callable, Any, Optional
import pandas as pd

FloatVec = npt.NDArray[np.floating]
IntVec = npt.NDArray[np.integer]
"""Reward chance per arm for agent."""
AgentType = FloatVec
""" % of agents playing an arm. """
Profile = FloatVec
""" Determines probability of agent getting a reward from arm, given a profile."""
Reward = Callable[[AgentType, Profile], FloatVec]

class State():
    """
    Represents that state of an agent.

    Attributes:
        wins: The number of wins the agent has made on each arm.
        losses: The number of wins the agent has made on each arm.
    """

    def __init__(self, n: int):
        """
        Initializes the state of an agent.

        Args:
            n: Number of arms
        """
        self._state = np.zeros(2*n, dtype=np.int64)

    @property
    def wins(self) -> IntVec:
        return self._state[::2]

    @property
    def losses(self):
        return self._state[1::2]

    def add_result(self, arm: int, won: bool):
        """
        Adds result to the current state. Modifies state in-place.

        Args:
            arm: Index of the arm that has been played
            won: True if won, False otherwise
        """
        if won:
            self._state[arm * 2] += 1
        else:
            self._state[arm * 2 + 1] += 1

""" Determines probabilities of agent picking an arm. """
Strategy = Callable[[State, int], int]

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
        strategy: Function of form Z^2n+ -> S, indicating the strategy the agent uses.
            lifetime: The number of steps the agent has been alive for.
        lifetime: The number of steps the agent has been alive for.
    """

    def __init__(self, n: int, strategy: Strategy, type: AgentType) -> None:
        """
        Initializes an agent.
        """
        self.lifetime = 0
        self.state = State(n)
        self.strategy = strategy
        self.type = type
        self._has_chosen = False
        self._chosen: Optional[int] = None

    def choose(self) -> int:
        """
        Returns the chosen arm of the agent at the current time step.
        """
        if self._has_chosen:
            raise RuntimeError(
                "Agent has already chosen an arm. Update agent's state before choosing again.")
        choice = self.strategy(self.state, self.lifetime)
        self._has_chosen = True
        self._chosen = choice
        return choice

    def update(self, won: bool) -> None:
        """
        Updates the agent's state with the reward from the arm,
        and increments the lifetime of the agent.

        won: True if the agent won on the arm it has chosen last, False otherwise.
        """
        if not self._has_chosen:
            raise RuntimeError(
                "Agent has not chosen an arm yet.")
        assert self._chosen is not None
        self.state.add_result(self._chosen, won)
        self.lifetime += 1
        self._has_chosen = False
        self._chosen = None




class Simulation():
    """
    Models the simulation of the agents. The resulting profile over time
    can be retrieved by calling the "get_history_df" method.
    """

    def __init__(self, n: int, m: int, type_sampler: Callable[[], AgentType], reward: Reward, beta: float, strategy: Strategy):
        """
        Initializes the simulation.

        Args:
            n: The number of arms.
            m: The number of agents.
            type_sampler: Random function that returns the type of an agent.
            reward: Reward function for each agent (see Agent for more information).
            strategy: Strategy function for each agent (see Agent for more information).
            beta: Probability that agent regenerates in each timestep
        """
        self._n = n
        self._m = m
        self._reward = reward
        self._beta = beta
        self._sample_type = type_sampler
        self._strategy = strategy
        self._t = 0
        self._history = []
        # call after all instance variables for agents have been initialized
        self._agents = [self.gen_agent() for _ in range(m)]
    
    def get_history_df(self) -> pd.DataFrame:
        """
        Returns a dataframe describing the history of the simulation up to the
        current time step. The dataframe is in "wide-format". 

        The dataframe is organized as such:
            The value variable "profile" contains the fraction of agents which 
            which picked arm "arm" at time step "t".
        """
        history_array = np.array(self._history)
        assert np.allclose(history_array.sum(axis=1), 1), "Profile didn't sum to 1"
        labels = list(range(1, self._n + 1))
        df = pd.DataFrame(history_array, columns=labels)
        df = pd.melt(df.reset_index(), id_vars=['index'], value_vars=labels)
        df.columns = ["t", "arm", "profile"]
        return df
         

    def gen_agent(self) -> Agent:
        """
        Samples a fresh agent from the distribution.
        """
        return Agent(self._n, self._strategy, self._sample_type())

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
        choices = [a.choose() for a in self._agents]
        agents_per_arm = np.bincount(choices, minlength=self._n)
        assert isinstance(agents_per_arm, np.ndarray)
        profile = agents_per_arm / self._m
        # Steps 2-4
        for i, a in enumerate(self._agents):
            # Step 2
            reward = self._reward(a.type, profile)
            # Step 3
            a.update(np.random.random() < reward[choices[i]])
            # Step 4
            if np.random.random() > self._beta:
                self._agents[i] = self.gen_agent()
        self._t += 1
        self._history.append(profile)
        return profile
