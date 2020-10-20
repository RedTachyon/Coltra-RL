import numpy as np
import gym
from typing import Dict, Any, Tuple, Callable, List

StateDict = Dict[str, np.ndarray]
ActionDict = Dict[str, Any]
RewardDict = Dict[str, float]
DoneDict = Dict[str, bool]
InfoDict = Dict[str, Any]


class MultiAgentEnv(gym.Env):
    """
    Base class for a gym-like environment for multiple agents. An agent is identified with its id (string),
    and most interactions are communicated through that API (actions, states, etc)
    """
    def __init__(self):
        self.config = {}

    def reset(self, *args, **kwargs) -> StateDict:
        """
        Resets the environment and returns the state.
        Returns:
            A dictionary holding the state visible to each agent.
        """
        raise NotImplementedError

    def step(self, action_dict: ActionDict) -> Tuple[StateDict, RewardDict, DoneDict, InfoDict]:
        """
        Executes the chosen actions for each agent and returns information about the new state.

        Args:
            action_dict: dictionary holding each agent's action

        Returns:
            states: new state for each agent
            rewards: reward obtained by each agent
            dones: whether the environment is done for each agent
            infos: any additional information
        """
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    @property
    def current_obs(self):
        raise NotImplementedError


# TODO: create a UnityEnvironment MAEnv wrapper/adapter