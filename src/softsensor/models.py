"""Mean-variance estimation (MVEM) network architectures.

Every architecture returns ``(mean, variance)``. The variance head predicts a log-variance,
which is clamped and exponentiated so the variance is always positive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHITECTURES = ("original", "mean_informed", "mean_and_hidden", "residual", "attention")
LOG_VAR_MIN, LOG_VAR_MAX = -10.0, 15.0


def get_activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}.get(name.lower(), nn.Tanh())


class MLP(nn.Module):
    """Fully connected network that can also return its last hidden-layer features."""

    def __init__(self, n_in, n_out, hidden, activation="tanh"):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), get_activation(activation)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_out)

    def forward_with_features(self, x):
        features = self.body(x)
        return self.head(features), features

    def forward(self, x):
        return self.forward_with_features(x)[0]


def _to_variance(log_var):
    return torch.exp(torch.clamp(log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX))


class MVEMNetwork(nn.Module):
    """Original MVEM: independent mean and variance networks on the same inputs."""

    def __init__(self, n_in, n_out, hidden, activation="tanh"):
        super().__init__()
        self.mean_network = MLP(n_in, n_out, hidden, activation)
        self.variance_network = MLP(n_in, n_out, hidden, activation)

    def forward(self, x):
        return self.mean_network(x), _to_variance(self.variance_network(x))


class MVEMNetworkMeanInformed(nn.Module):
    """Variance network sees the inputs and the predicted mean."""

    def __init__(self, n_in, n_out, hidden, activation="tanh"):
        super().__init__()
        self.mean_network = MLP(n_in, n_out, hidden, activation)
        self.variance_network = MLP(n_in + n_out, n_out, hidden, activation)

    def forward(self, x):
        mean = self.mean_network(x)
        return mean, _to_variance(self.variance_network(torch.cat([x, mean], dim=-1)))


class MVEMNetworkMeanAndHidden(nn.Module):
    """Variance network sees inputs, predicted mean and the mean network's last hidden features."""

    def __init__(self, n_in, n_out, hidden, activation="tanh"):
        super().__init__()
        self.mean_network = MLP(n_in, n_out, hidden, activation)
        self.variance_network = MLP(n_in + n_out + hidden[-1], n_out, hidden, activation)

    def forward(self, x):
        mean, feats = self.mean_network.forward_with_features(x)
        return mean, _to_variance(self.variance_network(torch.cat([x, mean, feats], dim=-1)))


class ResidualBlock(nn.Module):
    def __init__(self, size, activation="tanh"):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.linear2(self.act(self.linear1(x))) + x)


class MVEMNetworkWithResiduals(nn.Module):
    """Mean-and-hidden variance inputs, processed by a residual variance network."""

    def __init__(self, n_in, n_out, hidden, activation="tanh", n_blocks=2):
        super().__init__()
        self.mean_network = MLP(n_in, n_out, hidden, activation)
        width = hidden[0]
        self.variance_network = nn.Sequential(
            nn.Linear(n_in + n_out + hidden[-1], width),
            get_activation(activation),
            *[ResidualBlock(width, activation) for _ in range(n_blocks)],
            nn.Linear(width, n_out),
        )

    def forward(self, x):
        mean, feats = self.mean_network.forward_with_features(x)
        return mean, _to_variance(self.variance_network(torch.cat([x, mean, feats], dim=-1)))


class UncertaintyAttention(nn.Module):
    """Self-attention across the scalar features of the variance input vector."""

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(1, hidden_dim)
        self.key_proj = nn.Linear(1, hidden_dim)
        self.value_proj = nn.Linear(1, hidden_dim)
        self.scale = hidden_dim**-0.5
        self.output_proj = nn.Linear(feature_dim * hidden_dim, hidden_dim)

    def forward(self, x):
        b = x.shape[0]
        x = x.unsqueeze(-1)  # [B, F, 1]
        q, k, v = self.query_proj(x), self.key_proj(x), self.value_proj(x)  # [B, F, H]
        weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)  # [B, F, F]
        attended = torch.matmul(weights, v).reshape(b, -1)  # [B, F*H]
        return self.output_proj(attended)  # [B, H]


class MVEMNetworkWithAttention(nn.Module):
    """Attention-based variance head with an error-aware scaling term."""

    def __init__(self, n_in, n_out, hidden, activation="tanh"):
        super().__init__()
        self.mean_network = MLP(n_in, n_out, hidden, activation)
        var_in, width = n_in + n_out + hidden[-1], hidden[0]
        self.uncertainty_attention = UncertaintyAttention(var_in, hidden_dim=width)
        self.variance_layers = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, n_out))
        self.error_predictor = nn.Sequential(
            nn.Linear(var_in, width), nn.ReLU(), nn.Linear(width, 1), nn.Sigmoid()
        )

    def forward(self, x):
        mean, feats = self.mean_network.forward_with_features(x)
        var_input = torch.cat([x, mean, feats], dim=-1)
        log_var = self.variance_layers(self.uncertainty_attention(var_input))
        log_var = log_var + torch.log(1.0 + 3.0 * self.error_predictor(var_input))
        return mean, _to_variance(log_var)


_REGISTRY = {
    "original": MVEMNetwork,
    "mean_informed": MVEMNetworkMeanInformed,
    "mean_and_hidden": MVEMNetworkMeanAndHidden,
    "residual": MVEMNetworkWithResiduals,
    "attention": MVEMNetworkWithAttention,
}


def build_mvem(architecture, n_in, n_out, hidden=(64, 64), activation="tanh") -> nn.Module:
    if architecture not in _REGISTRY:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose from {ARCHITECTURES}.")
    return _REGISTRY[architecture](n_in, n_out, list(hidden), activation)
