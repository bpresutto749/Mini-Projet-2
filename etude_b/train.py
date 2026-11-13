import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pystk2
import torch

SRC = os.environ.get("PYSTK_SRC", str(Path(__file__).resolve().parent / "pystk_project" / "src"))
sys.path.insert(0, SRC)
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
from agents.team4.agent4 import Agent4

from agent import SAC, Buffer, Imitator

STEP_S = 0.1
T_MAX = 1000
STUCK_WINDOW = 100
STUCK_MIN = 5.0
K_PATHS = 8
K_ITEMS = 4
N_POWERUP = 11
N_ITEM = 7
ATTACH_NONE = 9
ZIPPER = pystk2.Powerup.Type.ZIPPER.value
ACT_DIM = 2
OBS_DIM = 3 + 3 + 1 + 1 + 1 + 1 + 1 + N_POWERUP + 7 * K_PATHS + (3 + N_ITEM) * K_ITEMS
NEUTRAL = {"acceleration": 0.0, "steer": 0.0, "brake": False, "drift": False, "nitro": False, "fire": False,
           "rescue": False}


def pad(rows, k, width):
    out = np.zeros((k, width), dtype=np.float32)
    rows = list(rows)[:k]
    if rows:
        out[: len(rows)] = np.asarray(rows, dtype=np.float32).reshape(len(rows), width)
    return out


def featurize(obs):
    powerup = np.zeros(N_POWERUP, dtype=np.float32)
    powerup[int(obs["powerup_type"])] = 1.0
    starts = pad(obs["paths_start"], K_PATHS, 3) / 20.0
    ends = pad(obs["paths_end"], K_PATHS, 3) / 20.0
    widths = pad([[w] for w in np.ravel(obs["paths_width"])], K_PATHS, 1) / 10.0
    items_pos = np.clip(pad(obs["items_position"], K_ITEMS, 3) / 30.0, -3.0, 3.0)
    items_type = np.zeros((K_ITEMS, N_ITEM), dtype=np.float32)
    for i, t in enumerate(list(obs["items_type"])[:K_ITEMS]):
        items_type[i, int(t)] = 1.0
    x = np.concatenate([
        np.asarray(obs["velocity"], dtype=np.float32) / 20.0,
        np.asarray(obs["center_path"], dtype=np.float32) / 5.0,
        np.ravel(obs["center_path_distance"]).astype(np.float32) / 5.0,
        np.ravel(obs["max_steer_angle"]).astype(np.float32),
        np.ravel(obs["energy"]).astype(np.float32),
        np.array([float(obs["jumping"])], dtype=np.float32),
        np.array([float(int(obs["attachment"]) != ATTACH_NONE)], dtype=np.float32),
        powerup,
        np.concatenate([starts, ends, widths], axis=1).ravel(),
        np.concatenate([items_pos, items_type], axis=1).ravel(),
    ])
    return np.clip(x, -10.0, 10.0)


def to_env(a, obs):
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    return {"acceleration": float((a[0] + 1.0) / 2.0), "steer": float(a[1]), "brake": False, "drift": False,
            "nitro": bool(np.ravel(obs["energy"])[0] > 0), "fire": int(obs["powerup_type"]) == ZIPPER,
            "rescue": False}


def from_expert(d):
    acc = float(np.clip(np.ravel(d.get("acceleration", 0.0))[0], 0.0, 1.0))
    steer = float(np.clip(np.ravel(d.get("steer", 0.0))[0], -1.0, 1.0))
    return np.array([2.0 * acc - 1.0, steer], dtype=np.float32)


class Runner:
    def __init__(self, track, course_reward=False):
        self.env = STKRaceEnv(agent=AgentSpec(name="solo", rank_start=0), track=track, num_kart=1,
                              use_subprocess=False)
        self.course_reward = course_reward
        self.expert = None

    def reset(self, expert=False):
        self.obs, _ = self.env.reset(seed=0)
        while self.obs["phase"] < 2:
            self.obs, _, _, _, _ = self.env.step(NEUTRAL)
        self.t = 0
        self.dist = [0.0]
        self.ddt = float(np.ravel(self.obs["distance_down_track"])[0])
        self.finished = False
        self.expert = Agent4(self.env) if expert else None
        return featurize(self.obs)

    def expert_action(self):
        return from_expert(self.expert.choose_action(self.obs))

    def step(self, a):
        self.obs, r, finished, _, info = self.env.step(to_env(a, self.obs))
        self.t += 1
        self.finished = bool(finished)
        if self.course_reward:
            self.dist.append(float(info["distance"]))
        else:
            length = self.env.track.length
            ddt = float(np.ravel(self.obs["distance_down_track"])[0])
            delta = (ddt - self.ddt + length / 2) % length - length / 2
            self.ddt = ddt
            self.dist.append(self.dist[-1] + delta)
            r = delta / 10.0 - 0.1 + (10.0 if self.finished else 0.0)
        stuck = self.t > STUCK_WINDOW and self.dist[-1] - self.dist[-1 - STUCK_WINDOW] < STUCK_MIN
        term = self.finished or stuck
        trunc = (not term) and self.t >= T_MAX
        return featurize(self.obs), float(r), bool(term), bool(trunc)

    def progress(self):
        if self.course_reward:
            return min(self.dist[-1] / self.env.track.length, 1.0)
        return 1.0 if self.finished else float(np.clip(self.dist[-1] / self.env.track.length, 0.0, 1.0))


def expert_demos(runner, n_episodes, noise, rng):
    keys = ("s", "a", "r", "s2", "term", "a_exp")
    data = {k: [] for k in keys}
    laps, lengths = [], []
    for _ in range(n_episodes):
        s = runner.reset(expert=True)
        while True:
            a_exp = runner.expert_action()
            a = a_exp.copy()
            a[:2] = np.clip(a[:2] + noise * rng.normal(size=2), -1.0, 1.0)
            s2, r, term, trunc = runner.step(a)
            for k, v in zip(keys, (s, a, r, s2, term, a_exp)):
                data[k].append(v)
            s = s2
            if term or trunc:
                laps.append(round(runner.t * STEP_S, 1) if runner.finished else np.nan)
                lengths.append(runner.t)
                break
    data = {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}
    return data, laps, lengths


def evaluate(runner, act, n_eps, noise, use_expert=False):
    rows = []
    for j in range(n_eps):
        rng = np.random.default_rng(10_000 + j)
        s = runner.reset(expert=use_expert)
        while True:
            a = runner.expert_action() if use_expert else act(s)
            a = np.asarray(a, dtype=np.float32).copy()
            a[:2] = np.clip(a[:2] + noise * rng.normal(size=2), -1.0, 1.0)
            s, _, term, trunc = runner.step(a)
            if term or trunc:
                break
        rows.append({"episode": j, "finished": int(runner.finished),
                     "lap_time": round(runner.t * STEP_S if runner.finished else T_MAX * STEP_S, 1),
                     "progress": round(runner.progress(), 4)})
    return rows


class Logger:
    def __init__(self, path, fields):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()

    def write(self, **row):
        self.w.writerow(row)
        self.f.flush()


def run(cfg):
    if cfg.algo == "sac":
        cfg.offline_ratio = 0.0
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out) / f"{cfg.algo}-s{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(cfg), indent=1))
    runner = Runner(cfg.track, bool(cfg.course_reward))
    ev = Logger(out / "eval.csv", ["online_steps", "expert_labels", "grad_steps", "wall_s", "episode",
                                   "finished", "lap_time", "progress"])
    t0 = time.time()

    def evaluate_now(act, online, labels, grads, use_expert=False):
        n_eps = max(cfg.eval_eps, 30) if use_expert else cfg.eval_eps
        for row in evaluate(runner, act, n_eps, cfg.eval_noise, use_expert):
            ev.write(online_steps=online, expert_labels=labels, grad_steps=grads,
                     wall_s=round(time.time() - t0, 1), **row)

    if cfg.algo == "expert":
        evaluate_now(None, 0, 0, 0, use_expert=True)
        (out / "done").write_text("ok")
        return

    if cfg.algo == "bc":
        checkpoints = [int(x) for x in cfg.bc_demos.split(",")]
        demos, _, lengths = expert_demos(runner, max(checkpoints), cfg.demo_noise, rng)
        bounds = np.cumsum(lengths)
        for k in checkpoints:
            n = int(bounds[k - 1])
            learner = Imitator(OBS_DIM, ACT_DIM, cfg)
            learner.add(demos["s"][:n], demos["a_exp"][:n])
            n_steps = int(round(cfg.bc_steps * max(1.0, n / bounds[cfg.n_demos - 1])))
            learner.train(n_steps, rng)
            evaluate_now(learner.actor.act, 0, n, n_steps)
        (out / "done").write_text("ok")
        return

    demos, laps, _ = expert_demos(runner, cfg.n_demos, cfg.demo_noise, rng)
    log = Logger(out / "train_log.csv", ["online_steps", "episode_return", "episode_len", "finished", "lap_time",
                                         "extra"])
    log.write(online_steps=0, episode_return=float("nan"), episode_len=len(demos["s"]), finished=-1,
              lap_time=float(np.nanmean(laps)), extra="demos")

    def log_episode(online, ret):
        log.write(online_steps=online, episode_return=ret, episode_len=runner.t,
                  finished=int(runner.finished),
                  lap_time=round(runner.t * STEP_S, 1) if runner.finished else float("nan"), extra="")

    if cfg.algo == "dagger":
        learner = Imitator(OBS_DIM, ACT_DIM, cfg)
        learner.add(demos["s"], demos["a_exp"])
        learner.train(cfg.bc_steps, rng)
        grads, online, labels = cfg.bc_steps, 0, len(demos["s"])
        evaluate_now(learner.actor.act, online, labels, grads)
        for _ in range(cfg.dagger_iters):
            S, A = [], []
            n_iter = 0
            while n_iter < cfg.dagger_steps:
                s = runner.reset(expert=True)
                ret = 0.0
                while True:
                    S.append(s)
                    A.append(runner.expert_action())
                    a = learner.actor.act(s)
                    a[:2] = np.clip(a[:2] + cfg.demo_noise * rng.normal(size=2), -1.0, 1.0)
                    s, r, term, trunc = runner.step(a)
                    ret += r
                    n_iter += 1
                    if term or trunc:
                        break
                log_episode(online + n_iter, ret)
            online += n_iter
            labels += n_iter
            learner.add(np.asarray(S, dtype=np.float32), np.asarray(A, dtype=np.float32))
            learner.train(cfg.dagger_grad_steps, rng)
            grads += cfg.dagger_grad_steps
            evaluate_now(learner.actor.act, online, labels, grads)
        (out / "done").write_text("ok")
        return

    agent = SAC(OBS_DIM, ACT_DIM, cfg)
    offline = None
    if cfg.offline_ratio > 0:
        offline = Buffer(OBS_DIM, ACT_DIM, len(demos["s"]))
        offline.add_batch(demos)
    online_buf = Buffer(OBS_DIM, ACT_DIM, cfg.total_steps)
    labels = len(demos["s"]) if offline is not None else 0
    evaluate_now(agent.actor.act, 0, labels, 0)
    s = runner.reset()
    ret, grads = 0.0, 0
    for t in range(1, cfg.total_steps + 1):
        a = agent.actor.act_stochastic(s)
        s2, r, term, trunc = runner.step(a)
        online_buf.add(s, a, r, s2, float(term))
        ret += r
        s = s2
        if term or trunc:
            log_episode(t, round(ret, 3))
            s, ret = runner.reset(), 0.0
        if t >= cfg.learning_starts:
            for _ in range(cfg.utd):
                agent.update(online_buf, offline, rng)
            grads += cfg.utd
        if t % cfg.eval_every == 0:
            evaluate_now(agent.actor.act, t, labels, grads)
            s, ret = runner.reset(), 0.0
    (out / "done").write_text("ok")


def run_all(cfg, extra):
    out = Path(cfg.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for algo in cfg.algos.split(","):
        for seed in ([0] if algo == "expert" else range(cfg.seeds)):
            name = f"{algo}-s{seed}"
            if not (out / name / "done").exists():
                jobs.append((name, [sys.executable, __file__, "--algo", algo, "--seed", str(seed), "--out", str(out)]
                             + extra))
    jobs.sort(key=lambda j: j[0].split("-")[0] not in ("rlpd", "sac"))
    print(f"{len(jobs)} runs to do, {cfg.jobs} in parallel -> {out.resolve()}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    t0, failed = time.time(), []

    def launch(job):
        name, cmd = job
        start = time.time()
        with open(out / "logs" / f"{name}.log", "w") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        return name, code, (time.time() - start) / 60

    with ThreadPoolExecutor(cfg.jobs) as pool:
        for k, fut in enumerate(as_completed([pool.submit(launch, j) for j in jobs]), 1):
            name, code, minutes = fut.result()
            if code != 0:
                failed.append(name)
            elapsed = (time.time() - t0) / 60
            print(f"[{k}/{len(jobs)}] {'OK    ' if code == 0 else 'FAILED'} {name} ({minutes:.1f} min) | "
                  f"elapsed {elapsed:.0f} min | remaining ~{elapsed / k * (len(jobs) - k):.0f} min", flush=True)
    print(f"All done in {(time.time() - t0) / 60:.1f} min. Failed: {', '.join(failed) or 'none'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--algos", default="expert,bc,dagger,rlpd,sac")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--algo", choices=["dagger", "rlpd", "sac", "bc", "expert"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--track", default="zengarden")
    p.add_argument("--out", default="results")
    p.add_argument("--course-reward", type=int, default=0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n-demos", type=int, default=20)
    p.add_argument("--demo-noise", type=float, default=0.1)
    p.add_argument("--eval-eps", type=int, default=10)
    p.add_argument("--eval-noise", type=float, default=0.05)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    p.add_argument("--bc-steps", type=int, default=5000)
    p.add_argument("--dagger-iters", type=int, default=16)
    p.add_argument("--dagger-steps", type=int, default=2500)
    p.add_argument("--dagger-grad-steps", type=int, default=2000)
    p.add_argument("--bc-demos", default="20,40,80,130")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--utd", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--offline-ratio", type=float, default=0.5)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=10_000)
    cfg, extra = p.parse_known_args()
    if cfg.all:
        passthrough = [a for a in sys.argv[1:] if a not in ("--all",)]
        drop = {"--algos", "--seeds", "--jobs", "--out"}
        clean, skip = [], False
        for a in passthrough:
            if skip:
                skip = False
                continue
            if a in drop:
                skip = True
                continue
            clean.append(a)
        run_all(cfg, clean)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
