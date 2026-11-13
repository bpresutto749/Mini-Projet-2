import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

B = 10_000


def boot_ci(x, seed=0):
    m = np.random.default_rng(seed).choice(np.asarray(x, float), size=(B, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_diff_ci(a, b, seed=1):
    rng = np.random.default_rng(seed)
    d = rng.choice(a, size=(B, len(a))).mean(1) - rng.choice(b, size=(B, len(b))).mean(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def holm(p):
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load(results):
    frames = []
    for d in sorted(Path(results).iterdir()):
        if (d / "done").exists():
            algo, seed = d.name.rsplit("-s", 1)
            df = pd.read_csv(d / "eval.csv")
            df["algo"], df["seed"] = algo, int(seed)
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["lap_time"] *= 10
    key = ["algo", "seed", "online_steps", "expert_labels"]
    per_eval = df.groupby(key, as_index=False).agg(lap=("lap_time", "mean"), completion=("finished", "mean"))
    per_eval["idx"] = per_eval.groupby(["algo", "seed"]).cumcount()
    return df, per_eval


def final(df, per_eval, algo, at=None):
    d = per_eval[per_eval.algo == algo]
    if at is not None:
        d = d[d.online_steps == at]
    elif algo == "bc":
        d = d[d.idx == d.idx.max()]
    else:
        d = d[d.idx >= d.groupby("seed").idx.transform("max") - 2]
    e = df[df.algo == algo].merge(d[["seed", "online_steps", "expert_labels"]])
    s = d.groupby("seed").agg(lap=("lap", "mean"), completion=("completion", "mean"), online=("online_steps", "max"))
    s["lap_finished"] = e[e.finished == 1].groupby("seed").lap_time.mean()
    return s


def report(results, fig_path):
    df, per_eval = load(results)
    expert = df[df.algo == "expert"]
    exp_lap = expert.lap_time.mean()
    f = {a: final(df, per_eval, a) for a in ("bc", "dagger", "rlpd", "sac")}
    f["rlpd@40k"] = final(df, per_eval, "rlpd", at=40_000)

    print(f"expert: {exp_lap:.0f} steps, success {expert.finished.mean():.2f} ({len(expert)} laps)")
    for name, s in f.items():
        lo, hi = boot_ci(s.lap)
        print(f"{name:9s} {s.lap.mean():5.0f} [{lo:.0f}, {hi:.0f}]  median {s.lap.median():.0f}  "
              f"success {s.completion.mean():.2f}  converged {(s.completion >= 0.9).sum()}/{len(s)}  "
              f"online steps {s.online.mean() / 1000:.0f}k")

    beat = {}
    for algo in ("rlpd", "sac"):
        d = per_eval[per_eval.algo == algo]
        ok = d[(d.completion == 1.0) & (d.lap < exp_lap)]
        beat[algo] = ok.groupby("seed").online_steps.min().reindex(sorted(d.seed.unique())) / 1000
        b = beat[algo]
        print(f"{algo}: first beats the expert after median {b.median():.0f}k steps ({b.min():.0f}-{b.max():.0f}k)")

    tests = []

    def two(name, a, b):
        lo, hi = boot_diff_ci(a, b)
        tests.append((name, a.mean() - b.mean(), lo, hi, stats.ttest_ind(a, b, equal_var=False).pvalue))

    def one(name, x):
        x = x[~np.isnan(x)]
        lo, hi = boot_ci(x - exp_lap)
        tests.append((name, x.mean() - exp_lap, lo, hi, stats.ttest_1samp(x, exp_lap).pvalue))

    lap = {k: v.lap.values for k, v in f.items()}
    two("DAgger - RLPD", lap["dagger"], lap["rlpd"])
    two("DAgger - RLPD at 40k", lap["dagger"], lap["rlpd@40k"])
    two("RLPD - SAC", lap["rlpd"], lap["sac"])
    two("DAgger - BC", lap["dagger"], lap["bc"])
    one("DAgger - expert", lap["dagger"])
    one("RLPD - expert", lap["rlpd"])
    conv = f["rlpd"][f["rlpd"].completion >= 0.9]
    one("RLPD finished laps - expert", conv.lap_finished.values)
    two("steps to beat the expert (k): RLPD - SAC", beat["rlpd"].values, beat["sac"].values)
    p_holm = holm([t[4] for t in tests])
    for (name, diff, lo, hi, p), ph in zip(tests, p_holm):
        print(f"{name}: delta = {diff:+.1f} [{lo:+.1f}, {hi:+.1f}]  p = {p:.2g}  p_Holm = {ph:.2g}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]
    for algo, name, col in (("dagger", "DAgger", "tab:blue"), ("sac", "SAC (no demos)", "tab:orange"),
                            ("rlpd", "RLPD (50/50)", "tab:green")):
        g = per_eval[per_eval.algo == algo].groupby("idx")
        x, y = g.online_steps.mean().values / 1000, g.lap.mean().values
        ci = np.array([boot_ci(v.values) for _, v in g.lap])
        ax.plot(x, y, marker="o", ms=3, color=col, label=name)
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=col, alpha=0.2)
    ax.axhline(exp_lap, color="k", ls="--", label="expert")
    ax.set(xlabel="Learner steps (x1000)", ylabel="Lap time (failure = 1000)", ylim=(0, 1050),
           title="(a) zengarden-team4, no pushes")
    ax = axes[1]
    for k, (algo, col) in enumerate((("bc", "tab:purple"), ("dagger", "tab:blue"), ("rlpd", "tab:green"),
                                     ("sac", "tab:orange"))):
        y = f[algo].lap.values
        ax.scatter(k + np.random.default_rng(k).uniform(-0.15, 0.15, len(y)), y, s=10, color=col, zorder=3)
        ax.plot([k - 0.25, k + 0.25], [np.median(y)] * 2, color="k", lw=1)
    ax.axhline(exp_lap, color="k", ls="--")
    ax.set_xticks(range(4), ["BC", "DAgger", "RLPD", "SAC"], fontsize=8)
    ax.set(ylabel="Final lap time", title="(b) final, per seed", ylim=(280, 400))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(fig_path)


def exploit(results):
    rows = []
    for d in sorted(Path(results).glob("*-s*")):
        algo, seed = d.name.rsplit("-s", 1)
        if algo not in ("rlpd", "sac"):
            continue
        t = pd.read_csv(d / "train_log.csv")
        t = t[t.extra != "demos"]
        hack = (t.finished == 0) & (t.episode_return > 30)
        last = t[t.online_steps > t.online_steps.max() - 20_000]
        e = pd.read_csv(d / "eval.csv")
        fe = e[e.online_steps >= sorted(e.online_steps.unique())[-3]]
        rows.append((algo, int(seed), ((last.finished == 0) & (last.episode_return > 30)).mean() > 0.3
                     and fe.finished.mean() == 0, t.online_steps[hack].min() if hack.any() else np.nan,
                     fe.finished.mean() >= 0.9))
    df = pd.DataFrame(rows, columns=["algo", "seed", "exploit", "first_exploit_step", "converged"])
    for algo, g in df.groupby("algo"):
        print(f"{algo}: {g.exploit.sum()}/{len(g)} runs end in the start-line exploit "
              f"(first exploit episode at {g.first_exploit_step.min():.0f} steps), {g.converged.sum()}/{len(g)} converged")


def simulate():
    import importlib
    import train
    runner = train.Runner("zengarden")
    turn = lambda t: np.array([1.0, 1.0], np.float32) if t < 40 else np.array([-1.0, 0.0], np.float32)
    for name, policy, use_expert in (("expert lap", None, True), ("U-turn, then stop", turn, False)):
        runner.reset(expert=use_expert)
        rews = []
        while True:
            before = max(0.0, float(runner.env.world.karts[0].overall_distance))
            a = runner.expert_action() if use_expert else policy(runner.t)
            _, _, term, trunc = runner.step(a)
            after = max(0.0, float(runner.env.world.karts[0].overall_distance))
            rews.append((after - before) / 10.0 - 0.1 + (10.0 if runner.finished else 0.0))
            if term or trunc:
                break
        rews = np.array(rews)
        print(f"{name}: default-reward return {rews.sum():.1f}, discounted {(0.99 ** np.arange(len(rews)) * rews).sum():.1f}")
    import os
    os.chdir(Path(train.SRC) / "main")
    env = runner.env
    laps = {}
    for track in env.TRACKS:
        env.default_track = track
        for team in (2, 3, 4, 5):
            agent = getattr(importlib.import_module(f"agents.team{team}.agent{team}"), f"Agent{team}")(env)
            obs, _ = env.reset(seed=0)
            steps, done = 0, False
            while not done and steps < 3000:
                obs, _, done, _, _ = env.step(agent.choose_action(obs))
                steps += 1
            laps[track, team] = steps
    best = sum(laps[t, 4] == min(laps[t, k] for k in (2, 3, 4, 5)) for t in env.TRACKS)
    near = sum(laps[t, 4] <= 1.02 * min(laps[t, k] for k in (2, 3, 4, 5)) for t in env.TRACKS)
    print(f"team 4 fastest (or tied) on {best}/{len(env.TRACKS)} tracks, within 2% on {near}/{len(env.TRACKS)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--fig", default="studyA.pdf")
    p.add_argument("--course-results", default=None)
    p.add_argument("--simulate", action="store_true")
    args = p.parse_args()
    plt.rcParams.update({"pdf.fonttype": 42})
    report(args.results, args.fig)
    if args.course_results:
        exploit(args.course_results)
    if args.simulate:
        simulate()


if __name__ == "__main__":
    main()
