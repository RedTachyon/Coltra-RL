from collections import OrderedDict
from typing import Dict, Callable, List, Tuple, Optional, TypeVar, Any

import gym
import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from agents import BaseAgent, Agent, StillAgent, RandomAgent
from preprocessors import simple_padder
from utils import DataBatch, with_default_config, np_float, transpose_batch, concat_batches
from base_env import MultiAgentEnv

T = TypeVar('T')


def append_dict(var: Dict[str, T], data_dict: Dict[str, List[T]]):
    """
    Works like append, but operates on dictionaries of lists and dictionaries of values (as opposed to lists and values)

    Args:
        var: values to be appended
        data_dict: lists to be appended to
    """
    for key, value in var.items():
        data_dict[key].append(value)


class Memory:
    """
    Holds the rollout data in a nested dictionary structure as follows:
    {
        "observations":
            {
                "Agent0": [obs1, obs2, ...],
                "Agent1": [obs1, obs2, ...]
            },
        "actions":
            {
                "Agent0": [act1, act2, ...],
                "Agent1": [act1, act2, ...]
            },
        ...,
        "states":
            {
                "Agent0": [(h1, c1), (h2, c2), ...]
                "Agent1": [(h1, c1), (h2, c2), ...]
            }
    }
    """

    def __init__(self, agents: List[str]):
        """
        Creates the memory container. The only argument is a list of agent names to set up the dictionaries.

        Args:
            agents: names of agents
        """

        self.agents = agents

        # Dictionaries to hold all the relevant data, with appropriate type annotations
        _observations: Dict[str, List[np.ndarray]] = {agent: [] for agent in self.agents}
        _actions: Dict[str, List[int]] = {agent: [] for agent in self.agents}
        _rewards: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        _logprobs: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        _dones: Dict[str, List[bool]] = {agent: [] for agent in self.agents}
        _toms: Dict[str, List[np.ndarray]] = {agent: [] for agent in self.agents}
        _sms: Dict[str, List[np.ndarray]] = {agent: [] for agent in self.agents}
        _states: Dict[str, List[Tuple[Tensor, ...]]] = {agent: [] for agent in self.agents}

        self.data = {
            "observations": _observations,
            "actions": _actions,
            "rewards": _rewards,
            "logprobs": _logprobs,
            "dones": _dones,
            "toms": _toms,
            "sms": _sms,
            "states": _states,
        }

    def store(self,
              obs: Dict[str, np.ndarray],
              action: Dict[str, int],
              reward: Dict[str, float],
              logprob: Dict[str, float],
              done: Dict[str, bool],
              tom: Dict[str, np.ndarray],
              sm: Dict[str, np.ndarray],
              state: Dict[str, Tuple[Tensor, ...]]):

        update = (obs, action, reward, logprob, done, tom, sm, state)
        for key, var in zip(self.data, update):
            append_dict(var, self.data[key])

    def reset(self):
        for key in self.data:
            self.data[key] = {agent: [] for agent in self.agents}

    def apply_to_agent(self, func: Callable) -> Dict[str, Any]:
        return {
            agent: func(agent) for agent in self.agents
        }

    def get_torch_data(self) -> DataBatch:
        """
        Gather all the recorded data into torch tensors (still keeping the dictionary structure)
        """
        observations = self.apply_to_agent(lambda agent: torch.tensor(np.stack(self.data["observations"][agent])))
        actions = self.apply_to_agent(lambda agent: torch.tensor(self.data["actions"][agent]))
        rewards = self.apply_to_agent(lambda agent: torch.tensor(self.data["rewards"][agent]))
        logprobs = self.apply_to_agent(lambda agent: torch.tensor(self.data["logprobs"][agent]))
        dones = self.apply_to_agent(lambda agent: torch.tensor(self.data["dones"][agent]))
        # breakpoint()
        try:
            toms = self.apply_to_agent(lambda agent: torch.tensor(np.stack(self.data["toms"][agent]).astype(np.float32)))
        except TypeError:  # ToM variables are Nones - as in none have been passed into the collector
            toms = self.apply_to_agent(lambda agent: torch.empty(len(self.data['toms'][agent]), 0))

        sms = self.apply_to_agent(lambda agent: torch.tensor(np.stack(self.data["sms"][agent]).astype(np.float32)))

        def stack_states(states_: List[Tuple[Tensor, ...]]):
            # transposed_states: Tuple[List[Tensor], ...] = tuple(list(i) for i in zip(*states_))
            # ([h1, h2, ...], [c1, c2, ...]) /\

            transposed_states: Tuple[List[Tensor], ...] = tuple(list(i) for i in zip(*states_))
            # ([h1, h2, ...], [c1, c2, ...]) /\

            tensor_states: Tuple[Tensor, ...] = tuple(torch.cat(state_type) for state_type in transposed_states)
            # (tensor(h1, h2, ...), tensor(c1, c2, ...)) /\

            return tensor_states

        states: Dict[str, List[Tuple[Tensor, ...]]] = self.data["states"]
        states = self.apply_to_agent(lambda agent: stack_states(states[agent]))

        torch_data = {
            "observations": observations,  # (batch_size, obs_size) float
            "actions": actions,  # (batch_size, ) int
            "rewards": rewards,  # (batch_size, ) float
            "logprobs": logprobs,  # (batch_size, ) float
            "dones": dones,  # (batch_size, ) bool
            "toms": toms,  # (batch_size, tom_params)
            "sms": sms,
            "states": states,  # (batch_size, 2, lstm_nodes),
        }

        return torch_data

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return self.data.__str__()


class Collector:
    """
    Class to perform data collection from two agents.
    """

    def __init__(self, agents: Dict[str, Agent], env: MultiAgentEnv, config: Dict):
        self.agents = agents
        self.agent_ids: List[str] = list(self.agents.keys())
        self.env = env
        self.memory = Memory(self.agent_ids)

        default_config = {
            # Whether or not to finish the last episode, even if that would exceed the step count
            "finish_episode": True,
        }

        self.config = with_default_config(config, default_config)

    def collect_data(self,
                     num_steps: Optional[int] = None,
                     num_episodes: Optional[int] = None,
                     deterministic: Optional[Dict[str, bool]] = None,
                     tom: Optional[Dict[str, np.ndarray]] = None,
                     disable_tqdm: bool = True,
                     max_steps: int = 102,
                     reset_memory: bool = True,
                     include_last: bool = False,
                     reset_start: bool = True,
                     env_goal_rewards: Tuple[float, float] = (.6, .2)) -> DataBatch:
        """
        Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.

        Args:
            num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
            num_episodes: number of episodes to generate
            deterministic: whether each agent should use the greedy policy; False by default
            tom: dictionary with GT parameters of the other agent
            disable_tqdm: whether a live progress bar should be (not) displayed
            max_steps: maximum number of steps that can be taken in episodic mode, recommended just above env maximum
            reset_memory: whether to reset the memory before generating data
            include_last: whether to include the last observation in episodic mode - useful for visualizations
            reset_start: whether the environment should be reset at the beginning of collection

        Returns: dictionary with the gathered data in the following format:

        {
            "observations":
                {
                    "Agent0": tensor([obs1, obs2, ...]),

                    "Agent1": tensor([obs1, obs2, ...])
                },
            "actions":
                {
                    "Agent0": tensor([act1, act2, ...]),

                    "Agent1": tensor([act1, act2, ...])
                },
            ...,

            "states":
                {
                    "Agent0": (tensor([h1, h2, ...]), tensor([c1, c2, ...])),

                    "Agent1": (tensor([h1, h2, ...]), tensor([c1, c2, ...]))
                }
        }
        """
        assert not ((num_steps is None) == (num_episodes is None)), ValueError("Exactly one of num_steps, num_episodes "
                                                                               "should receive a value")

        if deterministic is None:
            deterministic = {agent_id: False for agent_id in self.agent_ids}

        if tom is None:
            tom = {agent_id: None for agent_id in self.agent_ids}

        if reset_memory:
            self.reset()

        if reset_start:
            obs = self.env.reset(env_goal_rewards)
        else:
            obs = self.env.current_obs

        state = {
            agent_id: self.agents[agent_id].get_initial_state(requires_grad=False) for agent_id in self.agent_ids
        }

        episode = 0

        end_flag = False
        full_steps = (num_steps + 100 * int(self.config["finish_episode"])) if num_steps else max_steps * num_episodes
        for step in trange(full_steps, disable=disable_tqdm):
            # Compute the action for each agent
            action_info = {  # action, logprob, entropy, state, sm
                agent_id: self.agents[agent_id].compute_single_action(obs[agent_id],
                                                                      state[agent_id],
                                                                      deterministic[agent_id],
                                                                      tom[agent_id])
                for agent_id in self.agent_ids
            }

            # Unpack the actions
            action = {agent_id: action_info[agent_id][0] for agent_id in self.agent_ids}
            logprob = {agent_id: action_info[agent_id][1] for agent_id in self.agent_ids}
            next_state = {agent_id: action_info[agent_id][2] for agent_id in self.agent_ids}
            sm_pred = {agent_id: action_info[agent_id][3] for agent_id in self.agent_ids}

            # Actual step in the environment
            next_obs, reward, done, info = self.env.step(action)

            # Saving to memory
            self.memory.store(obs, action, reward, logprob, done, tom, sm_pred, state)

            # Handle episode/loop ending
            if self.config["finish_episode"] and step + 1 == num_steps:
                end_flag = True

            # Update the current obs and state - either reset, or keep going
            if all(done.values()):  # episode is over
                if include_last:  # record the last observation along with placeholder action/reward/logprob
                    self.memory.store(next_obs, action, reward, logprob, done, tom, next_state)

                # Episode mode handling
                episode += 1
                if episode == num_episodes:
                    break

                # Step mode with episode finish handling
                if end_flag:
                    break

                # If we didn't end, create a new environment
                obs = self.env.reset(env_goal_rewards)

                state = {
                    agent_id: self.agents[agent_id].get_initial_state(requires_grad=False)
                    for agent_id in self.agent_ids
                }

            else:  # keep going
                obs = next_obs
                state = next_state

        return self.memory.get_torch_data()

    def reset(self):
        self.memory.reset()

    def change_agent(self, agents_to_replace: Dict[str, BaseAgent]):
        """Replace the agents in the collector"""
        for agent_id in agents_to_replace:
            if agent_id in self.agents:
                self.agents[agent_id] = agents_to_replace[agent_id]

    def update_agent_state_dict(self, agents_to_update: Dict[str, Dict]):
        """Update the state dict of the agents in the collector"""
        for agent_id in agents_to_update:
            if agent_id in self.agents:
                self.agents[agent_id].model.load_state_dict(agents_to_update[agent_id])


def collect_training_batch(collector: Collector,
                           main_weights: Dict,
                           old_weights: List[Dict],
                           num_episodes: int = 100) -> Tuple[Dict, Tensor]:
    """Collects a batch of data with an expert and a novice, concats those batches and pads them for use in a separate
    SM predictor. To be developed further."""

    collector.update_agent_state_dict({
        "Agent0": main_weights,
        "Agent1": old_weights[-1]
    })

    expert_data = collector.collect_data(num_episodes=num_episodes, tom={
        "Agent0": np_float(1.),
        "Agent1": np_float(1.)
    })

    collector.update_agent_state_dict({
        "Agent1": old_weights[0],
    })

    novice_data = collector.collect_data(num_episodes=num_episodes, tom={
        "Agent0": np_float(0.),
        "Agent1": np_float(1.),
    })

    agent_novice_data = transpose_batch(novice_data)['Agent0']
    agent_expert_data = transpose_batch(expert_data)['Agent0']

    agent_data = concat_batches([agent_novice_data, agent_expert_data])

    padded_batch, mask = simple_padder(agent_data)

    return padded_batch, mask


def collect_simple_batch(collector: Collector,
                         num_episodes: int = 100,
                         padding: bool = True) -> Tuple[Dict, Tensor]:
    """Collects a batch of data with an expert and a novice, concats those batches and pads them for use in a separate
    SM predictor. To be developed further."""

    collector.change_agent({
        "Agent0": RandomAgent(),
        "Agent1": StillAgent()
    })

    expert_data = collector.collect_data(num_episodes=num_episodes, tom={
        "Agent0": np_float(1.),
        "Agent1": np_float(1.)
    })

    collector.change_agent({
        "Agent0": StillAgent(),
    })

    novice_data = collector.collect_data(num_episodes=num_episodes, tom={
        "Agent0": np_float(0.),
        "Agent1": np_float(0.),
    })

    agent_novice_data = transpose_batch(novice_data)['Agent0']
    agent_expert_data = transpose_batch(expert_data)['Agent0']

    agent_data = concat_batches([agent_novice_data, agent_expert_data])

    if padding:
        padded_batch, mask = simple_padder(agent_data)
    else:
        padded_batch = agent_data
        mask = None

    return padded_batch, mask

if __name__ == '__main__':
    pass

    # env = foraging_env_creator({})
    #
    # agent_ids = ["Agent0", "Agent1"]
    #
    # agents: Dict[str, Agent] = {
    #     agent_id: Agent(LSTMModel({}), name=agent_id)
    #     for agent_id in agent_ids
    # }
    #
    # runner = Collector(agents, env, {})
    #
    # data_steps = runner.collect_data(num_steps=1000, disable_tqdm=False)
    # data_episodes = runner.collect_data(num_episodes=2, disable_tqdm=False)
    # print(data_episodes['observations']['Agent0'].shape)
    # generate_video(data_episodes['observations']['Agent0'], 'vids/video.mp4')
