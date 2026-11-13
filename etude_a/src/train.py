# train.py: DAgger vs symmetric sampling RLPD for a single SuperTuxKart kart driving one lap alone
#uv run src/train.py --algo dagger --seed 0
# uv run src/train.py --algo rlpd   --seed 0
#   uv run src/train.py --algo rlpd   --offline-ratio 0
# out: results/<algo>_<track>-<expert>_<seed>.csv & requires analysis.py
import argparse, csv, importlib, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pystk_project", "src"))  # pystk2_gymnasium + expert agents
# for Windows, the pysupertuxkart2 wheel does not ship its DLLs, we find them from SuperTuxKart's official dependencies-win-x86_64.zip 
# (+ libc++/libunwind from llvm-mingw msvcrt)
DLLS = os.path.join(ROOT, "stk_deps", "dependencies-win-x86_64", "bin")
if os.name == "nt" and os.path.isdir(DLLS):
    os.add_dll_directory(DLLS)
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec

DEV = torch.device("cpu")  # set in main()
EXPERTS = {f"team{i}": (f"agents.team{i}.agent{i}", f"Agent{i}") for i in range(1, 6)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="dagger", choices=["dagger", "rlpd"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--track", default="lighthouse")
    p.add_argument("--expert", default="team4", choices=list(EXPERTS))
    p.add_argument("--n-demos", type=int, default=5, help="expert laps given to both methods")
    p.add_argument("--total-steps", type=int, default=50_000, help="learner environment steps")
    p.add_argument("--max-steps", type=int, default=2000, help="episode time limit (steps of 0.1 s)")
    p.add_argument("--perturb", type=float, default=0.06, help="probability per step that a random push starts")
    p.add_argument("--perturb-len", type=int, default=5, help="steps of random steering during a push")
    p.add_argument("--stuck-steps", type=int, default=100, help="truncate after this many steps without progress")
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    # DAgger
    p.add_argument("--bc-updates", type=int, default=2000, help="gradient steps after each DAgger iteration")
    p.add_argument("--bc-init-updates", type=int, default=10_000, help="gradient steps of iteration 0 (= BC)")
    p.add_argument("--beta", type=float, default=0.5, help="expert mixing at iteration i is beta**i")
    # RLPD (SAC + symmetric sampling)
    p.add_argument("--offline-ratio", type=float, default=0.5, help="fraction of each batch from expert data")
    p.add_argument("--utd", type=int, default=4, help="critic updates per environment step")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--exp-name", default="")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


# observations and actions
K = 8 # number of track nodes given to the policy

def features(obs):
    paths = np.asarray(obs["paths_start"], dtype=np.float32).reshape(-1, 3)
    widths = np.asarray(obs["paths_width"], dtype=np.float32).reshape(-1)
    if len(paths) == 0:
        paths, widths = np.zeros((1, 3), np.float32), np.zeros(1, np.float32)
    idx = np.minimum(np.arange(K), len(paths) - 1)  # near the finish line, repeat the last node
    return np.concatenate([
        np.asarray(obs["velocity"]) / 10, np.asarray(obs["center_path"]) / 10, obs["center_path_distance"] / 10,
        obs["max_steer_angle"], obs["skeed_factor"], np.asarray(obs["energy"]) / 10,
        (paths[idx] / 20).ravel(), widths[idx] / 10]).astype(np.float32)

# an action is a vector of 6 numbers in [-1, 1]: acceleration, steer, then brake/drift/nitro/rescue (> 0 means on)
ACT_DIM = 6

def encode(a):
    on = lambda x: 1.0 if bool(np.ravel(x)[0]) else -1.0
    acc, steer = float(np.ravel(a["acceleration"])[0]), float(np.ravel(a["steer"])[0])
    return np.array([2 * acc - 1, np.clip(steer, -1, 1), on(a["brake"]), on(a["drift"]),
                     on(a["nitro"]), on(a["rescue"])], dtype=np.float32)

def decode(v):
    return {"acceleration": float(np.clip((v[0] + 1) / 2, 0, 1)), "steer": float(np.clip(v[1], -1, 1)),
            "brake": v[2] > 0, "drift": v[3] > 0, "nitro": v[4] > 0, "rescue": v[5] > 0, "fire": False}


def make_expert(name, env):
    module, cls = EXPERTS[name]
    return getattr(importlib.import_module(module), cls)(env, path_lookahead=3) # fresh internal state


CRASH = (EOFError, BrokenPipeError, ConnectionError)

def restart(env):
    env._stk.process.kill()
    env._stk = None
    env.initialize(with_graphics=False)


def rollout(*a, **kw):
    for _ in range(10):
        try:
            return _rollout(*a, **kw)
        except CRASH:
            print("STK crashed, restarting", flush=True)
            restart(a[0])
    raise RuntimeError("SuperTuxKart keeps crashing")


def _rollout(env, args, seed, policy=None, expert=None, beta=0.0, on_step=None):
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)[0]
    ex = make_expert(expert, env) if expert else None
    S, A, E, R, S2, D = [], [], [], [], [], []
    length = env.track.length
    prev, progress, best, last_progress, finished, push = float(obs["distance_down_track"][0]), 0.0, 0.0, 0, False, 0
    for t in range(args.max_steps):
        s = features(obs)
        e = encode(ex.choose_action(obs)) if ex else None  # the expert is queried at every step
        a = e if policy is None or rng.random() < beta else policy(s)
        if push == 0 and rng.random() < args.perturb:  # random push: full throttle, random steering
            push, push_a = args.perturb_len, np.array([1, rng.uniform(-1, 1), -1, -1, -1, -1], np.float32)
        if push > 0:  # the push replaces the driver's action (part of the environment, same for everybody)
            a, push = push_a, push - 1
        obs, _, finished, _, _ = env.step(decode(a))
        # custom reward: signed progress along the track in meters and wrapped at the start line
        # pystk2's reward uses overall_distance but jumps by a whole lap when the kart crosses the line
        # backwards: the RL agent learned to reverse behind the line and park there
        d = float(obs["distance_down_track"][0])
        delta = (d - prev + length / 2) % length - length / 2
        prev, progress = d, progress + delta
        r = delta / 10 - 0.1 + 10 * finished
        s2 = features(obs)
        S.append(s); A.append(a); E.append(e); R.append(r); S2.append(s2); D.append(float(finished))
        if on_step:
            on_step(s, a, r, s2, float(finished))
        if progress > best + 1.0:
            best, last_progress = progress, t
        if finished or t - last_progress > args.stuck_steps:
            break
    stats = {"steps": t + 1, "finished": finished, "progress": 1.0 if finished else min(best / length, 1.0)}
    return dict(S=S, A=A, E=E, R=R, S2=S2, D=D), stats


def evaluate(env, args, act):
    res = [rollout(env, args, 100_000 + k, policy=act)[1] for k in range(args.eval_episodes)]
    fin = [r["finished"] for r in res]
    return {"success": np.mean(fin), "progress": np.mean([r["progress"] for r in res]),
            # lap time in steps of 0.1 s
            # an unfinished lap counts as the time limit
            "lap_time": np.mean([r["steps"] if r["finished"] else args.max_steps for r in res])}


def expert_demos(env, args):
    path = os.path.join(ROOT, "data", f"demos_{args.expert}_{args.track}_{args.n_demos}_p{args.perturb}.npz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        parts, times = [], []
        for k in range(args.n_demos):
            tr, st = rollout(env, args, seed=k, expert=args.expert)
            parts.append(tr); times.append(st["steps"] if st["finished"] else np.nan)
            print(f"expert lap {k}: {st}")
        cat = lambda key: np.array([x for p in parts for x in p[key]], dtype=np.float32)
        np.savez(path, S=cat("S"), A=cat("A"), E=cat("E"), R=cat("R"), S2=cat("S2"), D=cat("D"), lap_times=np.array(times))
    return dict(np.load(path))


# networks 
def mlp(i, o, h, ln=False):
    norm = (lambda: nn.LayerNorm(h)) if ln else nn.Identity
    return nn.Sequential(nn.Linear(i, h), norm(), nn.ReLU(), nn.Linear(h, h), norm(), nn.ReLU(), nn.Linear(h, o))

class Actor(nn.Module):
    def __init__(self, obs_dim, h):
        super().__init__()
        self.net = mlp(obs_dim, 2 * ACT_DIM, h)

    def forward(self, s):
        mean, log_std = self.net(s).chunk(2, -1)
        return mean, log_std.clamp(-5, 2)

    def deterministic(self, s):
        return torch.tanh(self(s)[0])

    def sample(self, s):
        mean, log_std = self(s)
        u = mean + log_std.exp() * torch.randn_like(mean)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mean) / log_std.exp()) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        return a, logp - torch.log(1 - a ** 2 + 1e-6).sum(-1)

    def act(self, s, stochastic=False):
        with torch.no_grad():
            s = torch.as_tensor(s, device=DEV).unsqueeze(0)
            return (self.sample(s)[0] if stochastic else self.deterministic(s))[0].cpu().numpy()


# DAgger
def run_dagger(env, args, demos, log):
    S, E = list(demos["S"]), list(demos["E"])  # aggregated dataset: states and expert labels
    pi = Actor(S[0].shape[0], args.hidden).to(DEV)
    opt = torch.optim.Adam(pi.parameters(), lr=args.lr)

    def fit(n_updates):
        s, e = torch.as_tensor(np.array(S), device=DEV), torch.as_tensor(np.array(E), device=DEV)
        for _ in range(n_updates):
            i = torch.randint(len(s), (args.batch_size,), device=DEV)
            loss = F.mse_loss(pi.deterministic(s[i]), e[i])
            opt.zero_grad(); loss.backward(); opt.step()

    fit(args.bc_init_updates)  # iteration 0 = behaviour cloning on the demonstrations
    steps, it, next_eval = 0, 0, 0
    while True:
        if steps >= next_eval:
            log(steps, len(S) - len(demos["S"]), evaluate(env, args, pi.act)); next_eval += args.eval_freq
        if steps >= args.total_steps:
            break
        it += 1
        tr, st = rollout(env, args, seed=args.seed * 10_000 + it, policy=pi.act, expert=args.expert, beta=args.beta ** it)
        S += tr["S"]; E += tr["E"]; steps += st["steps"]  # aggregate the visited states with expert labels
        fit(args.bc_updates)
    return pi


# RLPD: SAC + symmetric sampling
class Critic(nn.Module):  # two Q-networks with LayerNorm like in RLPD
    def __init__(self, obs_dim, h):
        super().__init__()
        self.q1, self.q2 = mlp(obs_dim + ACT_DIM, 1, h, ln=True), mlp(obs_dim + ACT_DIM, 1, h, ln=True)

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

def run_rlpd(env, args, demos, log):
    obs_dim = demos["S"].shape[1]
    actor, critic, target = (m.to(DEV) for m in [Actor(obs_dim, args.hidden), Critic(obs_dim, args.hidden), Critic(obs_dim, args.hidden)])
    target.load_state_dict(critic.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True, device=DEV)  # entropy temperature, tuned automatically
    opt_a, opt_c = torch.optim.Adam(actor.parameters(), lr=args.lr), torch.optim.Adam(critic.parameters(), lr=args.lr)
    opt_t = torch.optim.Adam([log_alpha], lr=args.lr)
    offline = {k: torch.as_tensor(demos[k], device=DEV) for k in ["S", "A", "R", "S2", "D"]}
    n = args.total_steps + args.max_steps  # online replay buffer, large enough to keep everything
    online = {k: torch.zeros((n, *v.shape[1:]), device=DEV) for k, v in offline.items()}
    n_off = int(args.batch_size * args.offline_ratio)

    def batch():  # symmetric sampling: n_off transitions from the expert data, the rest from the online buffer
        i = torch.randint(len(offline["S"]), (n_off,), device=DEV)
        j = torch.randint(steps, (args.batch_size - n_off,), device=DEV)
        return [torch.cat([offline[k][i], online[k][j]]) for k in ["S", "A", "R", "S2", "D"]]

    def update():
        alpha = log_alpha.exp().detach()
        for _ in range(args.utd):
            s, a, r, s2, d = batch()
            with torch.no_grad():
                a2, logp2 = actor.sample(s2)
                y = r + args.gamma * (1 - d) * (torch.min(*target(s2, a2)) - alpha * logp2)
            q1, q2 = critic(s, a)
            loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            opt_c.zero_grad(); loss.backward(); opt_c.step()
            with torch.no_grad():
                for p, tp in zip(critic.parameters(), target.parameters()):
                    tp.lerp_(p, args.tau)
        a_new, logp = actor.sample(s)  # one actor update per environment step
        loss = (alpha * logp - torch.min(*critic(s, a_new))).mean()
        opt_a.zero_grad(); loss.backward(); opt_a.step()
        t_loss = -(log_alpha * (logp.detach() - ACT_DIM)).mean()  # target entropy = -|A|
        opt_t.zero_grad(); t_loss.backward(); opt_t.step()

    steps, ep, next_eval = 0, 0, 0
    def on_step(s, a, r, s2, d):
        nonlocal steps
        for k, v in zip(["S", "A", "R", "S2", "D"], [s, a, r, s2, d]):
            online[k][steps] = torch.as_tensor(v)
        steps += 1
        if steps >= args.learning_starts:
            update()

    while True:
        if steps >= next_eval:
            log(steps, 0, evaluate(env, args, actor.act)); next_eval += args.eval_freq
        if steps >= args.total_steps:
            break
        ep += 1
        rollout(env, args, seed=args.seed * 10_000 + ep, policy=lambda s: actor.act(s, stochastic=True), on_step=on_step)
    return actor


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    global DEV
    DEV = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    env = STKRaceEnv(agent=AgentSpec(use_ai=False, rank_start=0), track=args.track, num_kart=1, laps=1)
    demos = expert_demos(env, args)
    print(f"{len(demos['S'])} expert transitions, expert lap times: {demos['lap_times']}")

    name = args.algo + ("" if args.algo == "dagger" or args.offline_ratio == 0.5 else f"-off{args.offline_ratio}") + args.exp_name
    out = os.path.join(ROOT, "results", f"{name}_{args.track}-{args.expert}_{args.seed}")  # e.g. rlpd_lighthouse-team4_0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f = open(out + ".csv", "w", newline="")
    w = csv.writer(f); w.writerow(["env_steps", "expert_labels", "lap_time", "success", "progress", "wallclock"])
    t0 = time.time()

    def log(steps, labels, ev):
        w.writerow([steps, labels, ev["lap_time"], ev["success"], ev["progress"], round(time.time() - t0)]); f.flush()
        print(f"[{name} s{args.seed}] steps {steps:6d}  lap {ev['lap_time']:6.0f}  success {ev['success']:.2f}  progress {ev['progress']:.2f}")

    policy = (run_dagger if args.algo == "dagger" else run_rlpd)(env, args, demos, log)
    torch.save(policy.state_dict(), out + ".pt")  # final policy
    f.close(); env.close()


if __name__ == "__main__":
    main()
