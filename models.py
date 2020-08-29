from typing import Dict, Tuple, Callable, Optional, Any

import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Categorical, Normal

from layers import RelationLayer
from utils import with_default_config, get_activation, get_initializer


class BaseModel(nn.Module):
    """
    A base class for any NN-based models, stateful or not, following a common convention:
    Each model in its forward pass takes an input, the previous recurrent state and a ToM value.
    If a certain quantity (i.e. state or ToM) is not used in that specific model, it will just be discarded

    The output of each model is an action distribution, the next recurrent state,
    and a dictionary with any extra outputs like the value or SM prediction
    """
    def __init__(self, config: Dict):
        super().__init__()
        self._stateful = False
        self.config = config
        self.device = 'cpu'

    def forward(self, x: Tensor,
                state: Tuple) -> Tuple[Distribution, Tuple, Dict[str, Tensor]]:
        # Output: action_dist, state, {value, sm_pred, whatever else}
        raise NotImplementedError

    def get_initial_state(self, requires_grad=True) -> Tuple:
        raise NotImplementedError

    @property
    def stateful(self):
        return self._stateful

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.device = 'cuda'

    def cpu(self):
        super().cpu()
        self.device = 'cpu'


class MLPModel(BaseModel):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, config: Dict):
        super().__init__(config)

        default_config = {
            "input_size": 21,
            "num_actions": 5,
            "activation": "leaky_relu",

            "hidden_sizes": (64, 64),

            "initializer": "kaiming_uniform",

            "tom_params": 0,
        }
        self.config = with_default_config(config, default_config)

        input_size: int = self.config.get("input_size")
        num_actions: int = self.config.get("num_actions")
        hidden_sizes: Tuple[int] = self.config.get("hidden_sizes")
        self.activation: Callable = get_activation(self.config.get("activation"))
        self.tom_params = self.config["tom_params"]

        layer_sizes = (input_size + self.tom_params,) + hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])

        self.policy_head = nn.Linear(layer_sizes[-1], num_actions)
        self.value_head = nn.Linear(layer_sizes[-1], 1)

        if self.config["initializer"]:
            # If given an initializer, initialize all weights using it, and all biases with 0's
            initializer_ = get_initializer(self.config["initializer"])

            for layer in self.hidden_layers:
                initializer_(layer.weight)
                nn.init.zeros_(layer.bias)

            initializer_(self.policy_head.weight)
            initializer_(self.value_head.weight)

            nn.init.zeros_(self.policy_head.bias)
            nn.init.zeros_(self.value_head.bias)

    def forward(self, x: Tensor,
                state: Tuple = ()) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        extra_outputs = {
            "value": value,
        }

        return action_distribution, state, extra_outputs

    def get_initial_state(self, requires_grad=True):
        return ()


class LSTMModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        self._stateful = True

        default_config = {
            "input_size": 21,
            "num_actions": 5,
            "activation": "leaky_relu",

            "pre_lstm_sizes": (32,),
            "lstm_nodes": 32,
            "post_lstm_sizes": (32,),

            "tom_params": 0,
        }
        self.config = with_default_config(config, default_config)

        # Unpack the config
        input_size: int = self.config.get("input_size")
        num_actions: int = self.config.get("num_actions")
        pre_lstm_sizes: Tuple[int] = self.config.get("pre_lstm_sizes")
        lstm_nodes: int = self.config.get("lstm_nodes")
        post_lstm_sizes: Tuple[int] = self.config.get("post_lstm_sizes")
        self.activation: Callable = get_activation(self.config.get("activation"))
        self.tom_params = self.config["tom_params"]

        pre_layers = (input_size + self.tom_params,) + pre_lstm_sizes
        post_layers = (lstm_nodes,) + post_lstm_sizes

        self.preprocess_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(pre_layers, pre_layers[1:])
        ])

        self.lstm = nn.LSTMCell(input_size=pre_layers[-1],
                                hidden_size=lstm_nodes,
                                bias=True)

        self.postprocess_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(post_layers, post_layers[1:])
        ])

        self.policy_head = nn.Linear(post_layers[-1], num_actions)
        self.value_head = nn.Linear(post_layers[-1], 1)

    def forward(self, x: Tensor,
                state: Tuple[Tensor, Tensor],
                tom: Optional[Tensor] = None) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:

        for layer in self.preprocess_layers:
            x = layer(x)
            x = self.activation(x)

        if len(state) == 0:
            state = self.get_initial_state()

        (h_state, c_state) = self.lstm(x, state)
        x = h_state

        for layer in self.postprocess_layers:
            x = layer(x)
            x = self.activation(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        extra_outputs = {
            "value": value,
        }

        return action_distribution, (h_state, c_state), extra_outputs

    def get_initial_state(self, requires_grad=True) -> Tuple[Tensor, Tensor]:
        return torch.zeros(1, self.config['lstm_nodes'], requires_grad=requires_grad).to(self.device), \
               torch.zeros(1, self.config['lstm_nodes'], requires_grad=requires_grad).to(self.device)


class ContinuousLSTMModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        self._stateful = True

        default_config = {
            "input_size": 5,
            "num_actions": 2,
            "activation": "leaky_relu",

            "pre_lstm_sizes": (32,),
            "lstm_nodes": 32,
            "post_lstm_sizes": (32,),

            "tom_params": 0,
        }
        self.config = with_default_config(config, default_config)

        # Unpack the config
        input_size: int = self.config.get("input_size")
        num_actions: int = self.config.get("num_actions")
        pre_lstm_sizes: Tuple[int] = self.config.get("pre_lstm_sizes")
        lstm_nodes: int = self.config.get("lstm_nodes")
        post_lstm_sizes: Tuple[int] = self.config.get("post_lstm_sizes")
        self.activation: Callable = get_activation(self.config.get("activation"))
        self.tom_params = self.config["tom_params"]

        pre_layers = (input_size + self.tom_params,) + pre_lstm_sizes
        post_layers = (lstm_nodes,) + post_lstm_sizes

        self.preprocess_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(pre_layers, pre_layers[1:])
        ])

        self.lstm = nn.LSTMCell(input_size=pre_layers[-1],
                                hidden_size=lstm_nodes,
                                bias=True)

        self.postprocess_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(post_layers, post_layers[1:])
        ])

        self.policy_head = nn.Linear(post_layers[-1], num_actions)
        self.std = nn.Parameter(torch.ones(1, num_actions) * 0.1, requires_grad=True)
        self.value_head = nn.Linear(post_layers[-1], 1)

    def forward(self, x: Tensor,
                state: Tuple[Tensor, Tensor],
                tom: Optional[Tensor] = None) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:

        sm = self.get_empty_sm(x)

        if getattr(self, "tom_params", 0) == 0:  # if something unexpected is passed, ignore it
            tom = None

        if tom is not None:
            x = torch.cat([x, tom], dim=-1)

        for layer in self.preprocess_layers:
            x = layer(x)
            x = self.activation(x)

        if len(state) == 0:
            state = self.get_initial_state()

        (h_state, c_state) = self.lstm(x, state)
        x = h_state

        for layer in self.postprocess_layers:
            x = layer(x)
            x = self.activation(x)

        action_means = self.policy_head(x)
        action_means = torch.sigmoid(action_means)  # [0, 1]

        value = self.value_head(x)

        action_distribution = Normal(loc=action_means, scale=self.std)
        # action_distribution = Categorical(logits=action_logits)

        extra_outputs = {
            "value": value,
            "sm": sm,
        }

        return action_distribution, (h_state, c_state), extra_outputs

    def get_initial_state(self, requires_grad=True) -> Tuple[Tensor, Tensor]:
        return torch.zeros(1, self.config['lstm_nodes'], requires_grad=requires_grad).to(self.device), \
               torch.zeros(1, self.config['lstm_nodes'], requires_grad=requires_grad).to(self.device)


class RelationModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        default_config = {
            "input_size": 21,
            "num_actions": 5,
            "activation": "leaky_relu",

            "emb_size": 4,
            "rel_hiddens": (16, 16,),
            "mlp_hiddens": (16,),

            "initializer": "kaiming_uniform",

        }
        self.config = with_default_config(config, default_config)

        self.tom_params = self.config["tom_params"]

        self.relation_layer = RelationLayer(self.config)

        self.policy_head = nn.Linear(self.config["mlp_hiddens"][-1], self.config["num_actions"])
        self.value_head = nn.Linear(self.config["mlp_hiddens"][-1], 1)

        if self.config["initializer"]:
            initializer_ = get_initializer(self.config["initializer"])
            initializer_(self.policy_head.weight)
            initializer_(self.value_head.weight)

            nn.init.zeros_(self.policy_head.bias)
            nn.init.zeros_(self.value_head.bias)

    def forward(self, x: Tensor,
                state: Tuple = ()) -> Tuple[Distribution, Tuple, Dict[str, Tensor]]:

        x = self.relation_layer(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        extra_outputs = {
            "value": value
        }

        return action_distribution, state, extra_outputs

    def get_initial_state(self, requires_grad=True) -> Tuple:
        return ()
