import numpy as np

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Normal

from models import BaseModel, MLPModel, LSTMModel, SkillPredictor

from typing import Tuple

from utils import AgentDataBatch


class BaseAgent:
    """A base class for an agent, exposing the basic API methods"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.stateful = False

    def compute_actions(self, obs_batch: Tensor,
                        state_batch: Tuple = (),
                        deterministic: bool = False) -> Tuple:
        raise NotImplementedError

    def compute_single_action(self, obs: np.ndarray,
                              state: Tuple[Tensor, ...] = (),
                              deterministic: bool = False):
        raise NotImplementedError

    def evaluate_actions(self, data_batch: AgentDataBatch,
                         padded: bool = False):
        raise NotImplementedError

    def get_initial_state(self, requires_grad=True):
        return getattr(self.model, "get_initial_state", lambda *x, **xx: ())(requires_grad=requires_grad)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def cuda(self):
        if self.model is not None:
            self.model.cuda()

    def cpu(self):
        if self.model is not None:
            self.model.cpu()


class Agent(BaseAgent):
    """Agent class for discrete-action models, with a categorical action distribution"""
    model: BaseModel

    def __init__(self, model: BaseModel):
        super().__init__(model)
        self.stateful = model.stateful

        self.tom_params = getattr(self.model, "tom_params", 0)

    def compute_actions(self, obs_batch: Tensor,
                        state_batch: Tuple = (),
                        deterministic: bool = False,
                        tom: Tensor = None) -> Tuple[Tensor, Tensor, Tuple, Tensor]:
        """
        Computes the action for a batch of observations with given hidden states. Breaks gradients.

        Args:
            obs_batch: observation array in shape either (batch_size, obs_size)
            state_batch: tuple of state tensors of shape (batch_size, lstm_nodes)
            deterministic: whether to always take the best action
            tom: [batch_size, tom_params] tensor holding the ToM GT parameters

        Returns:
            action, logprob of the action, new state vectors
        """
        action_distribution: Categorical
        states: Tuple
        with torch.no_grad():
            action_distribution, states, extra_outputs = self.model(obs_batch, state_batch, tom=tom)

        if deterministic:
            actions = torch.argmax(action_distribution.probs, dim=-1)
        else:
            actions = action_distribution.sample()

        logprobs = action_distribution.log_prob(actions)

        return actions, logprobs, states, extra_outputs["sm"]

    def compute_single_action(self, obs: np.ndarray,
                              state: Tuple[Tensor, ...] = (),
                              deterministic: bool = False,
                              tom: np.ndarray = None) -> Tuple[int, float, Tuple, float]:
        """
        Computes the action for a single observation with the given hidden state. Breaks gradients.

        Args:
            obs: flat observation array in shape either
            state: tuple of state tensors of shape (1, lstm_nodes)
            deterministic: boolean, whether to always take the best action
            tom: flat array of ToM GT parameters

        Returns:
            action, logprob of the action, new state vectors
        """
        obs = torch.tensor([obs])
        if tom is not None:
            tom = torch.tensor([tom])

        with torch.no_grad():
            action, logprob, new_state, sm = self.compute_actions(obs, state, deterministic, tom=tom)

        return action.item(), logprob.item(), new_state, sm.item()

    def compute_action_probs(self, obs: np.ndarray,
                             state: Tuple[Tensor, ...] = (),
                             deterministic: bool = False,
                             tom: np.ndarray = None) -> Tuple[int, np.ndarray, Tuple, np.ndarray]:
        """
        Same as above, but also returns the action probs - for the manual mode
        """
        obs = torch.tensor([obs])
        if tom is not None:
            tom = torch.tensor([tom])

        action_distribution: Categorical
        states: Tuple
        with torch.no_grad():
            action_distribution, states, extra_outputs = self.model(obs, state, tom=tom)
            sm_preds = extra_outputs["sm"]

        if deterministic:
            actions = torch.argmax(action_distribution.probs, dim=-1)
        else:
            actions = action_distribution.sample()

        action = actions.item()
        probs = action_distribution.probs.cpu().numpy().ravel()

        return action, probs, states, sm_preds.cpu().numpy().ravel()

    def evaluate_actions(self, data_batch: AgentDataBatch,
                         padded: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Computes action logprobs, observation values and policy entropy for each of the (obs, action, hidden_state)
        transitions. Preserves all the necessary gradients.

        Args:
            data_batch: data collected from a Collector for this agent
            padded: whether the data is passed as 1D (not padded; [T*B, *]) or 2D (padded; [T, B, *]) tensor

        Returns:
            action_logprobs: tensor of action logprobs (batch_size, )
            values: tensor of observation values (batch_size, )
            entropies: tensor of entropy values (batch_size, )
        """
        obs_batch = data_batch['observations']
        action_batch = data_batch['actions']
        tom_batch = data_batch['toms']
        state_batch = data_batch['states']
        if not padded:  # BP or non-recurrent
            action_distribution, new_states, extra_outputs = self.model(obs_batch, state_batch, tom=tom_batch)
            values = extra_outputs["value"]
            sm_preds = extra_outputs["sm"]
            action_logprobs = action_distribution.log_prob(action_batch)
            values = values.view(-1)
            entropies = action_distribution.entropy()
            sm_preds = sm_preds.view(-1)

        else:  # padded == True, BPTT
            batch_size = obs_batch.size()[1]  # assume it's padded, so in [L, B, *] format
            state: Tuple[Tensor, ...] = self.get_initial_state()
            state = tuple(_state.repeat(batch_size, 1) for _state in state)
            entropies = []
            action_logprobs = []
            values = []
            sm_preds = []
            # states_cache = [state]
            # breakpoint()

            for (obs, action, tom) in zip(obs_batch, action_batch, tom_batch):
                action_distribution, new_state, extra_outputs = self.model(obs, state, tom=tom)
                value = extra_outputs["value"]
                sm_pred = extra_outputs["sm"]
                action_logprob = action_distribution.log_prob(action)
                entropy = action_distribution.entropy()
                action_logprobs.append(action_logprob)
                values.append(value.T)
                entropies.append(entropy)
                sm_preds.append(sm_pred)

                state = new_state
                # states_cache.append(state)

            action_logprobs = torch.stack(action_logprobs)
            values = torch.cat(values, dim=0)
            entropies = torch.stack(entropies)
            sm_preds = torch.stack(sm_preds)

        return action_logprobs, values, entropies, sm_preds


class ContinuousAgent(BaseAgent):
    """Agent variant for Continuous (Normal) action distributions"""
    model: BaseModel

    def __init__(self, model: BaseModel):
        super().__init__(model)
        self.stateful = model.stateful

        self.tom_params = getattr(self.model, "tom_params", 0)

    def compute_actions(self, obs_batch: Tensor,
                        state_batch: Tuple = (),
                        deterministic: bool = False,
                        tom: Tensor = None) -> Tuple[Tensor, Tensor, Tuple, Tensor]:
        """
        Computes the action for a batch of observations with given hidden states. Breaks gradients.

        Args:
            obs_batch: observation array in shape either (batch_size, obs_size)
            state_batch: tuple of state tensors of shape (batch_size, lstm_nodes)
            deterministic: whether to always take the best action
            tom: [batch_size, tom_params] tensor holding the ToM GT parameters

        Returns:
            action, logprob of the action, new state vectors
        """
        action_distribution: Normal
        states: Tuple
        with torch.no_grad():
            action_distribution, states, extra_outputs = self.model(obs_batch, state_batch, tom=tom)

        if deterministic:
            actions = action_distribution.loc
        else:
            actions = action_distribution.sample()

        logprobs = action_distribution.log_prob(actions).sum(1)

        return actions, logprobs, states, extra_outputs["sm"]

    def compute_single_action(self, obs: np.ndarray,
                              state: Tuple[Tensor, ...] = (),
                              deterministic: bool = False,
                              tom: np.ndarray = None) -> Tuple[np.ndarray, float, Tuple, float]:
        """
        Computes the action for a single observation with the given hidden state. Breaks gradients.

        Args:
            obs: flat observation array in shape either
            state: tuple of state tensors of shape (1, lstm_nodes)
            deterministic: boolean, whether to always take the best action
            tom: flat array of ToM GT parameters

        Returns:
            action, logprob of the action, new state vectors
        """
        obs = torch.tensor([obs])
        if tom is not None:
            tom = torch.tensor([tom])

        with torch.no_grad():
            action, logprob, new_state, sm = self.compute_actions(obs, state, deterministic, tom=tom)

        return action.numpy().ravel(), logprob.item(), new_state, sm.item()

    def compute_action_probs(self, obs: np.ndarray,
                             state: Tuple[Tensor, ...] = (),
                             deterministic: bool = False,
                             tom: np.ndarray = None) -> Tuple[int, np.ndarray, Tuple, np.ndarray]:
        """
        Same as above, but also returns the action probs - for the manual mode
        """
        obs = torch.tensor([obs])
        if tom is not None:
            tom = torch.tensor([tom])

        action_distribution: Categorical
        states: Tuple
        with torch.no_grad():
            action_distribution, states, extra_outputs = self.model(obs, state, tom=tom)
            sm_preds = extra_outputs["sm"]

        if deterministic:
            actions = torch.argmax(action_distribution.probs, dim=-1)
        else:
            actions = action_distribution.sample()

        action = actions.item()
        probs = action_distribution.probs.cpu().numpy().ravel()

        return action, probs, states, sm_preds.cpu().numpy().ravel()

    def evaluate_actions(self, data_batch: AgentDataBatch,
                         padded: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Computes action logprobs, observation values and policy entropy for each of the (obs, action, hidden_state)
        transitions. Preserves all the necessary gradients.

        Args:
            data_batch: data collected from a Collector for this agent
            padded: whether the data is passed as 1D (not padded; [T*B, *]) or 2D (padded; [T, B, *]) tensor

        Returns:
            action_logprobs: tensor of action logprobs (batch_size, )
            values: tensor of observation values (batch_size, )
            entropies: tensor of entropy values (batch_size, )
        """
        obs_batch = data_batch['observations']
        action_batch = data_batch['actions']
        tom_batch = data_batch['toms']
        state_batch = data_batch['states']
        if not padded:  # BP or non-recurrent
            action_distribution, new_states, extra_outputs = self.model(obs_batch, state_batch, tom=tom_batch)
            values = extra_outputs["value"]
            sm_preds = extra_outputs["sm"]
            action_logprobs = action_distribution.log_prob(action_batch).sum(1)
            values = values.view(-1)
            entropies = action_distribution.entropy().sum(1)
            sm_preds = sm_preds.view(-1)

        else:  # padded == True, BPTT
            # TODO: Might not work, copied from Agent
            batch_size = obs_batch.size()[1]  # assume it's padded, so in [L, B, *] format
            state: Tuple[Tensor, ...] = self.get_initial_state()
            state = tuple(_state.repeat(batch_size, 1) for _state in state)
            entropies = []
            action_logprobs = []
            values = []
            sm_preds = []
            # states_cache = [state]
            # breakpoint()

            for (obs, action, tom) in zip(obs_batch, action_batch, tom_batch):
                action_distribution, new_state, extra_outputs = self.model(obs, state, tom=tom)
                value = extra_outputs["value"]
                sm_pred = extra_outputs["sm"]
                action_logprob = action_distribution.log_prob(action)
                entropy = action_distribution.entropy()
                action_logprobs.append(action_logprob)
                values.append(value.T)
                entropies.append(entropy)
                sm_preds.append(sm_pred)

                state = new_state
                # states_cache.append(state)

            action_logprobs = torch.stack(action_logprobs)
            values = torch.cat(values, dim=0)
            entropies = torch.stack(entropies)
            sm_preds = torch.stack(sm_preds)

        return action_logprobs, values, entropies, sm_preds


class StillAgent(BaseAgent):
    def __init__(self, model: nn.Module = None, action_value: int = 4):
        super().__init__(model)
        self.action_value = action_value

    def compute_actions(self, obs_batch: Tensor,
                        *args, **kwargs) -> Tuple[Tensor, Tensor, Tuple, Tensor]:
        batch_size = obs_batch.shape[0]
        actions = torch.ones(batch_size) * self.action_value
        actions = actions.to(torch.int64)

        logprobs = torch.zeros(batch_size)

        states = ()

        extra_outputs = {
            "sm": torch.zeros(batch_size)
        }

        return actions, logprobs, states, extra_outputs["sm"]

    def compute_single_action(self, obs: np.ndarray,
                              *args, **kwargs) -> Tuple[int, float, Tuple, float]:

        return self.action_value, 0., (), 0.

    def evaluate_actions(self, data_batch: AgentDataBatch, padded: bool = False):
        batch_size = data_batch["observations"].shape[0]

        action_logprobs = torch.zeros(batch_size)
        values = torch.zeros(batch_size)
        entropies = torch.zeros(batch_size)
        sm_preds = torch.zeros(batch_size)

        return action_logprobs, values, entropies, sm_preds


class RandomAgent(BaseAgent):
    def __init__(self, model: nn.Module = None, action_value: int = 4):
        super().__init__(model)
        self.action_value = action_value

    def compute_actions(self, obs_batch: Tensor,
                        *args, **kwargs) -> Tuple[Tensor, Tensor, Tuple, Tensor]:
        batch_size = obs_batch.shape[0]
        actions = torch.randint(0, self.action_value + 1, (batch_size,))
        actions = actions.to(torch.int64)

        logprobs = torch.ones(batch_size) * np.log(1 / (self.action_value + 1))

        states = ()

        extra_outputs = {
            "sm": torch.zeros(batch_size)
        }

        return actions, logprobs, states, extra_outputs["sm"]

    def compute_single_action(self, obs: np.ndarray,
                              *args, **kwargs) -> Tuple[int, float, Tuple, float]:

        return torch.randint(0, self.action_value + 1, (1,)).item(), 0., (), 0.

    def evaluate_actions(self, data_batch: AgentDataBatch, padded: bool = False):
        batch_size = data_batch["observations"].shape[0]

        action_logprobs = torch.zeros(batch_size)
        values = torch.zeros(batch_size)
        entropies = torch.zeros(batch_size)
        sm_preds = torch.zeros(batch_size)

        return action_logprobs, values, entropies, sm_preds
