import numpy as np
import gym
from typing import Dict, Any, Tuple, Callable, List

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


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


class UnityCrowdEnv:
    def __init__(self, *args, **kwargs):

        self.engine_channel = EngineConfigurationChannel()

        self.active_agents: List[str] = []

        kwargs.setdefault("side_channels", []).append(self.engine_channel)

        self.unity = UnityEnvironment(*args, **kwargs)
        self.behaviors = {}

    def _get_step_info(self, use_terminals: bool = False) -> Tuple[StateDict, RewardDict, DoneDict]:
        names = self.behaviors.keys()
        obs_dict: StateDict = {}
        reward_dict: RewardDict = {}
        done_dict: DoneDict = {}

        for name in names:
            decisions, terminals = self.unity.get_steps(name)
            dec_obs, dec_ids = decisions.obs, list(decisions.agent_id)
            for idx in dec_ids:
                agent_name = f"{name}&id={idx}"
                obs = np.concatenate([o[dec_ids.index(idx)] for o in dec_obs])
                obs_dict[agent_name] = obs
                reward_dict[agent_name] = decisions.reward[dec_ids.index(idx)]
                done_dict[agent_name] = obs[-1] == 0.

            if use_terminals:
                ter_obs, ter_ids = terminals.obs, list(terminals.agent_id)
                for idx in terminals.agent_id:
                    agent_name = f"{name}&id={idx}"
                    obs_dict[agent_name] = np.concatenate([o[ter_ids.index(idx)] for o in ter_obs])
                    reward_dict[agent_name] = terminals.reward[ter_ids.index(idx)]
                    done_dict[agent_name] = True

        for done_agent in done_dict:
            if done_dict[done_agent]:
                self.active_agents.remove(done_agent)

        done_dict["__all__"] = len(self.active_agents) == 0

        return obs_dict, reward_dict, done_dict

    def step(self, action: ActionDict) -> Tuple[StateDict, RewardDict, DoneDict, InfoDict]:
        if len(self.active_agents) == 0:
            raise EnvironmentError("The episode ended - reset the game to continue!")

        for name in self.behaviors.keys():
            decisions, terminals = self.unity.get_steps(name)
            action_shape = self.behaviors[name].action_shape
            dec_obs, dec_ids = decisions.obs, list(decisions.agent_id)
            all_actions = np.array([action.get(f"{name}&id={id_}", np.zeros(action_shape)).ravel()
                                    for id_ in dec_ids])
            if len(all_actions) == 0:
                all_actions = np.zeros((0, action_shape))
            self.unity.set_actions(name, all_actions)

        self.unity.step()

        obs_dict, reward_dict, done_dict = self._get_step_info(True)
        info_dict = {}  # TODO

        return obs_dict, reward_dict, done_dict, info_dict

    def reset(self) -> StateDict:
        # Double reset, after way too much pain to try to circumvent it... might come back to it later
        self.unity.reset()
        self.unity.reset()

        # All behavior names, except for Manager agents which do not take actions but manage the environment
        self.behaviors = dict(self.unity.behavior_specs)
        self.behaviors = {key: value for key, value in self.behaviors.items() if not key.startswith("Manager")}

        obs_dict, _, _ = self._get_step_info()

        self.active_agents = list(obs_dict.keys())

        return obs_dict

    @property
    def current_obs(self) -> StateDict:
        obs_dict, _, _ = self._get_step_info()
        return obs_dict

    def close(self):
        self.unity.close()
