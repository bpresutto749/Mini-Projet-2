import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * act_dim))

    def forward(self, s):
        mu, log_std = self.net(s).chunk(2, dim=-1)
        return mu, torch.clamp(log_std, -5.0, 2.0)

    def sample(self, s):
        mu, log_std = self(s)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp

    @torch.no_grad()
    def act(self, s):
        mu, _ = self(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
        return torch.tanh(mu)[0].numpy()

    @torch.no_grad()
    def act_stochastic(self, s):
        a, _ = self.sample(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
        return a[0].numpy()


class EnsembleLinear(nn.Module):
    def __init__(self, n, d_in, d_out):
        super().__init__()
        bound = 1.0 / math.sqrt(d_in)
        self.w = nn.Parameter(torch.empty(n, d_in, d_out).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.empty(n, 1, d_out).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)


class Critics(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(EnsembleLinear(2, obs_dim + act_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                 EnsembleLinear(2, hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                 EnsembleLinear(2, hidden, 1))

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1).unsqueeze(0).expand(2, -1, -1)
        return self.net(x).squeeze(-1)


class Imitator:
    def __init__(self, obs_dim, act_dim, cfg):
        self.cfg = cfg
        self.actor = Actor(obs_dim, act_dim, cfg.hidden)
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.bc_lr)
        self.S = np.zeros((0, obs_dim), dtype=np.float32)
        self.A = np.zeros((0, act_dim), dtype=np.float32)

    def add(self, s, a_exp):
        self.S = np.concatenate([self.S, s])
        self.A = np.concatenate([self.A, a_exp])

    def train(self, n_steps, rng):
        S, A = torch.as_tensor(self.S), torch.as_tensor(self.A)
        for _ in range(n_steps):
            idx = torch.as_tensor(rng.integers(0, len(S), size=self.cfg.batch))
            mu, _ = self.actor(S[idx])
            loss = F.mse_loss(torch.tanh(mu), A[idx])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


class Buffer:
    def __init__(self, obs_dim, act_dim, capacity):
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.term = np.zeros(capacity, dtype=np.float32)
        self.n = 0

    def add(self, s, a, r, s2, term):
        i = self.n
        self.s[i], self.a[i], self.r[i], self.s2[i], self.term[i] = s, a, r, s2, term
        self.n += 1

    def add_batch(self, d):
        for k in ("s", "a", "r", "s2", "term"):
            getattr(self, k)[self.n: self.n + len(d["s"])] = d[k]
        self.n += len(d["s"])

    def sample(self, k, rng):
        idx = rng.integers(0, self.n, size=k)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.term[idx]


class SAC:
    def __init__(self, obs_dim, act_dim, cfg):
        self.cfg = cfg
        self.actor = Actor(obs_dim, act_dim, cfg.hidden)
        self.critics = Critics(obs_dim, act_dim, cfg.hidden)
        self.target = copy.deepcopy(self.critics).requires_grad_(False)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.target_entropy = -float(act_dim)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critics.parameters(), lr=cfg.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

    def batch(self, online, offline, rng):
        n_off = int(round(self.cfg.batch * self.cfg.offline_ratio)) if offline is not None else 0
        parts = [online.sample(self.cfg.batch - n_off, rng)]
        if n_off:
            parts.append(offline.sample(n_off, rng))
        return [torch.as_tensor(np.concatenate(x)) for x in zip(*parts)]

    def update(self, online, offline, rng):
        s, a, r, s2, term = self.batch(online, offline, rng)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q2 = self.target(s2, a2)
            y = r + self.cfg.gamma * (1.0 - term) * (q2.min(0).values - alpha * logp2)
        q = self.critics(s, a)
        critic_loss = ((q - y.unsqueeze(0)) ** 2).mean(1).sum()
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        a_new, logp = self.actor.sample(s)
        actor_loss = (alpha * logp - self.critics(s, a_new).min(0).values).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critics.parameters(), self.target.parameters()):
                pt.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p)
