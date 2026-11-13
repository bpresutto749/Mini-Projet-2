# analysis.py -> results/: learning curves, final scores with
# 95% bootstrap CI, Welch t-tests
# uv run src/analysis.py --last 2
import argparse, glob, itertools, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--last", type=int, default=2, help="final score = mean of the last evaluations")
args = p.parse_args()
NAMES = {"dagger": "DAgger", "rlpd": "RLPD (50/50)", "rlpd-off0.0": "SAC (no demos)"}

# results/<algo>_<track>-<expert>_<seed>.csv -> runs[config][method] = list of runs
runs = {}
for path in sorted(glob.glob(os.path.join(ROOT, "results", "*_*-team*_*.csv"))):
    algo, config, _ = os.path.basename(path).split("_", 1)[0], *os.path.basename(path).split("_", 1)[1].rsplit("_", 1)
    if algo in NAMES:
        runs.setdefault(config, {}).setdefault(NAMES[algo], []).append(np.genfromtxt(path, delimiter=",", names=True))

def boot_ci(x, n=10_000, rng=np.random.default_rng(0)):
    return np.percentile(rng.choice(x, (n, len(x))).mean(1), [2.5, 97.5])

def expert_time(config):
    track, expert = config.rsplit("-", 1)
    demos = glob.glob(os.path.join(ROOT, "data", f"demos_{expert}_{track}_*.npz"))
    return np.nanmean(np.load(demos[0])["lap_times"]) if demos else None

out = []
fig, axes = plt.subplots(1, len(runs), figsize=(2.5 * len(runs), 2.5), squeeze=False)
for ax, (config, res) in zip(axes[0], runs.items()):
    # final scores: mean of the last evaluations
    # BC = evaluation of DAgger before any DAgger iteration
    final, success = {}, {}
    if "DAgger" in res:
        final["BC"] = np.array([d["lap_time"][0] for d in res["DAgger"]])
        success["BC"] = np.array([d["success"][0] for d in res["DAgger"]])
    for name, ds in res.items():
        final[name] = np.array([d["lap_time"][-args.last:].mean() for d in ds])
        success[name] = np.array([d["success"][-args.last:].mean() for d in ds])
        grid = np.arange(0, min(d["env_steps"][-1] for d in ds) + 1, 10_000)
        ys = np.array([np.interp(grid, d["env_steps"], d["lap_time"]) for d in ds])  # DAgger evaluates between episodes
        ci = np.array([boot_ci(y) for y in ys.T])
        ax.plot(grid / 1000, ys.mean(0), marker="o", ms=3, label=name)
        ax.fill_between(grid / 1000, ci[:, 0], ci[:, 1], alpha=0.2)
    expert = expert_time(config)
    if expert:
        ax.axhline(expert, color="k", ls="--", label="expert")
    ax.set(xlabel="Learner steps (x1000)", title=config, ylim=(0, 2050))

    out += ["", f"== {config}, expert lap time {expert:.0f}",
            f"{'method':16s} n   lap time [95% CI]   median   success [95% CI]   expert labels"]
    for name, x in final.items():
        (lo, hi), (slo, shi) = boot_ci(x), boot_ci(success[name])
        labels = np.mean([d["expert_labels"][-1] for d in res[name]]) if name in res else 0
        out.append(f"{name:16s} {len(x):2d}  {x.mean():6.0f} [{lo:4.0f}, {hi:4.0f}]   {np.median(x):5.0f}   "
                   f"{success[name].mean():.2f} [{slo:.2f}, {shi:.2f}]   {labels:.0f}")
    # pair wise Welch t-tests on the per-run final lap times (paired t-test for BC vs DAgger: same runs),
    # Holm correction over the tests of this configuration
    pairs = list(itertools.combinations(final, 2))
    pv = [(stats.ttest_rel if {a, b} == {"BC", "DAgger"} else lambda x, y: stats.ttest_ind(x, y, equal_var=False))(
          final[a], final[b]).pvalue for a, b in pairs]
    order, holm, m = np.argsort(pv), np.empty(len(pv)), len(pv)
    for rank, i in enumerate(order):
        holm[i] = min(1.0, max(pv[i] * (m - rank), holm[order[rank - 1]] if rank else 0))
    for (a, b), p_raw, p_h in zip(pairs, pv, holm):
        out.append(f"{a} vs {b}: delta lap time = {final[b].mean() - final[a].mean():+.0f}  p = {p_raw:.4f}  p_Holm = {p_h:.4f}")

axes[0, 0].set_ylabel("Lap time (failure = 2000)")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.9))
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "figures", "curves.pdf"))
print("\n".join(out))
open(os.path.join(ROOT, "figures", "results.txt"), "w").write("\n".join(out).strip() + "\n")
