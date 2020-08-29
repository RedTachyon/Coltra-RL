from typing import Dict, List, Union, Tuple, Any, Callable, Type, Optional, Iterator

import numpy as np

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import time

from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamw import AdamW
from torch.optim.adamax import Adamax
from torch.optim.sgd import SGD

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter


DataBatch = DataBatchT = Dict[str, Dict[str, Any]]
AgentDataBatch = Dict[str, Union[Tensor, Tuple]]


def np_float(x: float) -> np.ndarray:
    """Convenience function to create a one-element float32 numpy array"""
    return np.array([x], dtype=np.float32)


def with_default_config(config: Dict, default: Dict) -> Dict:
    """
    Adds keys from default to the config, if they don't exist there yet.
    Serves to ensure that all necessary keys are always present.
    Now also recursive.

    Args:
        config: config dictionary
        default: config dictionary with default values

    Returns:
        config with the defaults added
    """
    if config is None:
        config = {}
    else:
        config = config.copy()
    for key in default.keys():
        if isinstance(default[key], dict):
            config[key] = with_default_config(config.get(key), default.get(key))
        else:
            config.setdefault(key, default[key])
    return config


def discount_rewards_to_go(rewards: Tensor, dones: Tensor, gamma: float = 1., batch_mode: bool = False) -> Tensor:
    """
    Computes the discounted rewards to go, handling episode endings. Nothing unusual.
    """
    if batch_mode:
        current_reward = 0
        discounted_rewards = []
        for reward in rewards.flip(0):
            current_reward = reward + gamma * current_reward
            discounted_rewards.insert(0, current_reward)
        return torch.stack(discounted_rewards)

    else:
        dones = dones.to(torch.int32)  # Torch can't handle reversing boolean tensors
        current_reward = 0
        discounted_rewards = []
        for reward, done in zip(rewards.flip(0), dones.flip(0)):
            if done:
                current_reward = 0
            current_reward = reward + gamma * current_reward
            discounted_rewards.insert(0, current_reward)
        return torch.tensor(discounted_rewards)


def get_optimizer(opt_name: str) -> Callable[..., Optimizer]:
    """Gets an optimizer by name"""
    optimizers = {
        "adam": Adam,
        "adadelta": Adadelta,
        "adamw": AdamW,
        "adagrad": Adagrad,
        "adamax": Adamax,
        "sgd": SGD
    }

    if opt_name not in optimizers.keys():
        raise ValueError(f"Wrong optimizer: {opt_name} is not a valid optimizer name. ")

    return optimizers[opt_name]


def get_activation(act_name: str) -> Callable[[Tensor], Tensor]:
    """Gets an activation function by name"""
    activations = {
        "relu": F.relu,
        "relu6": F.relu6,
        "elu": F.elu,
        "leaky_relu": F.leaky_relu,
        "sigmoid": F.sigmoid,
        "tanh": F.tanh,
        "softmax": F.softmax,
        "gelu": lambda x: x * F.sigmoid(1.702 * x)
    }

    if act_name not in activations.keys():
        raise ValueError(f"Wrong activation: {act_name} is not a valid activation function name.")

    return activations[act_name]


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x: Tensor):
        return x * F.sigmoid(1.702 * x)


def get_activation_module(act_name: str) -> nn.Module:
    """Gets an activation module by name"""
    # noinspection PyTypeChecker
    activations: Dict[str, nn.Module] = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "gelu": GELU,
    }

    if act_name not in activations.keys():
        raise ValueError(f"Wrong activation: {act_name} is not a valid activation function name.")

    return activations[act_name]


def get_initializer(init_name: str) -> Callable[[Tensor], None]:
    """Gets an initializer by name"""
    initializers = {
        "kaiming_normal": nn.init.kaiming_normal_,
        "kaiming_uniform": nn.init.kaiming_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
        "xavier_uniform": nn.init.xavier_uniform_,
        "zeros": nn.init.zeros_
    }

    if init_name not in initializers.keys():
        raise ValueError(f"Wrong initialization: {init_name} is not a valid initializer name.")

    return initializers[init_name]


class Timer:
    """
    Simple timer to get temporal metrics. Upon calling .checkpoint(), returns the time since the last call of
    """

    def __init__(self):
        self.start = time.time()

    def checkpoint(self) -> float:
        now = time.time()
        diff = now - self.start
        self.start = now
        return diff


def get_episode_lens(done_batch: Tensor) -> Tuple[int]:
    """
    Based on the recorded done values, returns the length of each episode in a batch.
    Args:
        done_batch: boolean tensor which values indicate terminal episodes

    Returns:
        tuple of episode lengths
    """
    episode_indices = done_batch.cpu().cumsum(dim=0)[:-1]
    episode_indices = torch.cat([torch.tensor([0]), episode_indices])  # [0, 0, 0, ..., 1, 1, ..., 2, ..., ...]

    ep_ids, ep_lens_tensor = torch.unique(episode_indices, return_counts=True)
    ep_lens = tuple(ep_lens_tensor.cpu().numpy())

    return ep_lens


def transpose_batch(data_batch: Union[DataBatch, DataBatchT]) -> Union[DataBatchT, DataBatch]:
    """
    In a 2-nested dictionary, swap the key levels. So it turns
    {
        "observations": {"Agent0": ..., "Agent1": ...},
        "actions": {"Agent0": ..., "Agent1": ...},
        ...
    }
    into
    {
        "Agent0": {"observations": ..., "actions": ..., ...},
        "Agent1": {"observations": ..., "actions": ..., ...},
    }
    Also works the other way around.
    Doesn't copy the underlying data, so it's very efficient (~30μs)
    """
    d = defaultdict(dict)
    for key1, inner in data_batch.items():
        for key2, value in inner.items():
            d[key2][key1] = value
    return dict(d)


def masked_mean(input_: Tensor, mask: Tensor) -> Tensor:
    """Mean of elements not covered by the mask"""
    return torch.sum(input_ * mask) / torch.sum(mask)


def masked_accuracy(preds: Tensor, labels: Tensor, mask: Tensor) -> float:
    preds_thresholded = (preds > .5).to(torch.int)
    correct_preds = (preds_thresholded == labels).to(torch.float)
    accuracy = masked_mean(correct_preds.mean(-1), mask).item()

    return accuracy


def masked_logloss(preds: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    logloss: Tensor = - labels * torch.log(preds) - (1 - labels) * torch.log(1 - preds)
    return masked_mean(logloss.mean(-1), mask)


def mean_accuracy(preds: Tensor, labels: Tensor) -> float:
    preds_thresholded = (preds > .5).to(torch.int)
    correct_preds = (preds_thresholded == labels).to(torch.float)
    accuracy = correct_preds.mean().item()

    return accuracy


def concat_batches(batches: List[AgentDataBatch]) -> AgentDataBatch:
    """Concatenate multiple batches of data"""
    merged = {}
    for key in batches[0]:
        if key == 'states':
            merged[key] = tuple(
                torch.cat([batch[key][i] for batch in batches], dim=0) for i in range(len(batches[0][key])))
        else:
            merged[key] = torch.cat([batch[key] for batch in batches], dim=0)

    return merged


def write_dict(metrics: Dict[str, Union[int, float]],
               step: int,
               writer: Optional[SummaryWriter] = None):
    """Writes a dictionary to a tensorboard SummaryWriter"""
    if writer is not None:
        writer: SummaryWriter
        for key, value in metrics.items():
            writer.add_scalar(tag=key, scalar_value=value, global_step=step)


def get_episode_rewards(batch: DataBatch) -> np.ndarray:
    """Computes the total reward in each episode in a data batch"""
    batch = transpose_batch(batch)['Agent0']
    ep_lens = get_episode_lens(batch['dones'])

    ep_rewards = np.array([torch.sum(rewards) for rewards in torch.split(batch['rewards'], ep_lens)])

    return ep_rewards


def state_iterator(state: Tuple[Tensor, ...]) -> Iterator[Tuple[Tensor, ...]]:
    if len(state) == 0:
        while True:
            yield ()
    for i in range(state[0].shape[0]):
        yield tuple((s[i] for s in state))


def entropy(p: np.ndarray) -> float:
    """Computes the entropy of a discrete distribution"""
    return float(np.sum(-p * np.log(p)))


def entropy_center(p: float, n: int) -> float:
    """Computes the entropy of a distribution with a single main value p, and n-1 uniformly distributed options"""
    assert 0 < p < 1
    q = (1 - p) / (n - 1)
    probs = np.array([p] + [q for _ in range(n - 1)])
    return entropy(probs)


def get_goal_ratio(reward_batch: Tensor,
                   done_batch: Tensor,
                   goal_reward: float = 0.6,
                   step_reward: float = -0.01) -> float:
    """Computes the ratio of correctly finished episodes to all episodes in a batch"""

    last_rewards = reward_batch[
                       done_batch].cpu().numpy() - step_reward  # get all episode end rewards, add the step penalty
    last_rewards = np.round(last_rewards, 4)  # round to avoid floating point errors
    total_eps = len(last_rewards)
    correct_eps = np.sum(last_rewards >= goal_reward)

    return correct_eps / total_eps


def sample_random_partner(returns: List[float]) -> int:
    """
    Samples a random skill level, and then returns the index closest to that skill level
    Args:
        returns: list of agents' returns

    Returns:
        index of the sampled agent
    """
    returns = np.array(returns)
    min_value, max_value = np.min(returns), np.max(returns)

    random_skill = np.random.uniform(min_value, max_value)

    idx = np.argmin(np.abs(returns - random_skill))
    return int(idx)


def batch_to_gpu(data_batch: AgentDataBatch) -> AgentDataBatch:
    new_batch = {}
    for key in data_batch:
        if key == 'states':
            new_batch[key] = tuple(state_.cuda() for state_ in data_batch[key])
        else:
            new_batch[key] = data_batch[key].cuda()
    return new_batch
