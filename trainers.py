import os
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, OrderedDict
import copy

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from agents import Agent
from environments import MultiAgentEnv
from utils import Timer, with_default_config, write_dict, transpose_batch, concat_batches, AgentDataBatch, DataBatch, \
    np_float, get_episode_rewards
from collectors import Collector, collect_training_batch, collect_simple_batch
from policy_optimization import PPOptimizer


class Trainer:
    def __init__(self,
                 agents: Dict[str, Agent],
                 env: MultiAgentEnv,
                 config: Dict[str, Any]):
        self.agents = agents
        self.agent_ids: List[str] = list(self.agents.keys())
        self.env = env
        self.config = config

    def train(self, num_iterations: int,
              disable_tqdm: bool = False,
              save_path: Optional[str] = None,
              **collect_kwargs):
        raise NotImplementedError


class PPOSimpleTrainer(Trainer):
    """This performs training in a sampling paradigm, where each agent is stored, and during data collection,
    some part of the dataset is collected with randomly sampled old agents"""

    def __init__(self, agents: Dict[str, Agent], env: MultiAgentEnv, config: Dict[str, Any]):
        super().__init__(agents, env, config)

        default_config = {
            "agents_to_optimize": ["Agent0"],  # ids of agents that should be optimized
            "steps": 10000,  # number of steps we want in one PPO step

            # Tensorboard settings
            "tensorboard_name": None,  # str, set explicitly

            # Collector
            "collector_config": {
                "finish_episode": True,
            },

            # PPO
            "ppo_config": {
                # GD settings
                "optimizer": "adam",
                "optimizer_kwargs": {
                    "lr": 1e-4,
                    "betas": (0.9, 0.999),
                    "eps": 1e-7,
                    "weight_decay": 0,
                    "amsgrad": False
                },
                "gamma": 1.,  # Discount factor

                # PPO settings
                "ppo_steps": 25,  # How many max. gradient updates in one iterations
                "eps": 0.1,  # PPO clip parameter
                "target_kl": 0.01,  # KL divergence limit
                "value_loss_coeff": 0.1,
                "entropy_coeff": 0.1,
                "max_grad_norm": 0.5,

                # Backpropagation settings
                "pad_sequences": False,  # BPTT toggle
                "use_gpu": False,
            }
        }

        self.config = with_default_config(config, default_config)

        self.collector = Collector(agents=self.agents, env=self.env, config=self.config["collector_config"])
        self.ppo = PPOptimizer(agents={"Agent0": agents["Agent0"]}, config=self.config["ppo_config"])

        # Setup tensorboard
        self.writer: SummaryWriter
        if self.config["tensorboard_name"]:
            dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.path = Path.home() / "tb_logs" / f"{self.config['tensorboard_name']}_{dt_string}"

            self.writer = SummaryWriter(str(self.path))
            os.mkdir(str(self.path / "saved_weights"))

            # Log the configs
            with open(str(self.path / "trainer_config.json"), "w") as f:
                json.dump(self.config, f)

            for agent_id in self.agent_ids:
                with open(str(self.path / f"{agent_id}_config.json"), "w") as f:
                    json.dump(self.agents[agent_id].model.config, f)

            with open(str(self.path / "env_config.json"), "w") as f:
                try:
                    env_config = self.env.config
                    json.dump(env_config, f)
                except AttributeError:  # if env doesn't have a config for some reason
                    pass

            self.path = str(self.path)
        else:
            self.path = None
            self.writer = None

    def train(self, num_iterations: int,
              save_path: Optional[str] = None,
              disable_tqdm: bool = False,
              **collect_kwargs):

        if save_path is None:
            save_path = self.path  # Can still be None

        print(f"Begin training, logged in {self.path}")
        timer = Timer()
        step_timer = Timer()

        # Store the first agent
        saved_agents = [copy.deepcopy(self.agents["Agent0"].model.state_dict())]

        # List to keep each agent's mean return as a crude skill approximation
        old_returns = []

        self_skill = np_float(1.)

        if save_path:
            torch.save(self.agents["Agent0"].model, os.path.join(save_path, "base_agent.pt"))

        for step in trange(num_iterations, disable=disable_tqdm):
            ########################################### Collect the data ###############################################
            timer.checkpoint()

            full_batch = self.collector.collect_data(num_steps=self.config["steps"])

            data_time = timer.checkpoint()

            ############################################## Update policy ##############################################

            # Perform the PPO update
            metrics = self.ppo.train_on_data(full_batch, step, writer=self.writer)

            end_time = step_timer.checkpoint()

            ########################################## Save the updated agent ##########################################

            # Save the agent
            saved_agents.append(copy.deepcopy(self.agents["Agent0"].model.state_dict()))

            # Save the agent to disk
            if save_path:
                # torch.save(old_returns, os.path.join(save_path, "returns.pt"))
                torch.save(self.agents["Agent0"].model.state_dict(),
                           os.path.join(save_path, "saved_weights", f"weights_{step + 1}"))

            # Write training time metrics to tensorboard
            time_metric = {f"{agent_id}/time_data_collection": data_time for agent_id in self.ppo.agent_ids}
            final_metric = {f"{agent_id}/total_time": end_time for agent_id in self.ppo.agent_ids}

            write_dict(time_metric, step, self.writer)
            write_dict(final_metric, step, self.writer)




if __name__ == '__main__':
    pass
    # from rollout import Collector

    # env_ = foraging_env_creator({})

    # agent_ids = ["Agent0", "Agent1"]
    # agents_: Dict[str, Agent] = {
    #     agent_id: Agent(LSTMModel({}), name=agent_id)
    #     for agent_id in agent_ids
    # }
    #
    # runner = Collector(agents_, env_)
    # data_batch = runner.rollout_steps(num_episodes=10, disable_tqdm=True)
    # obs_batch = data_batch['observations']['Agent0']
    # action_batch = data_batch['actions']['Agent0']
    # reward_batch = data_batch['rewards']['Agent0']
    # done_batch = data_batch['dones']['Agent0']
    #
    # logprob_batch, value_batch, entropy_batch = agents_['Agent0'].evaluate_actions(obs_batch,
    #                                                                                action_batch,
    #                                                                                done_batch)
