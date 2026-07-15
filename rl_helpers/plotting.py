"""Plotting utilities shared across the RL notebooks.

Everything here is thin matplotlib wrapping so that notebooks stay focused on the
RL algorithm rather than on plotting boilerplate.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .gridworld import ACTION_ARROWS, GridWorld


def moving_average(x, window: int = 50):
    """Simple trailing moving average used to smooth noisy learning curves."""
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_learning_curve(returns, window: int = 50, title: str = "Learning curve",
                        ax=None, label: str | None = None):
    """Plot raw episode returns plus a smoothed moving average."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    returns = np.asarray(returns, dtype=float)
    ax.plot(returns, alpha=0.25, color="tab:blue")
    smooth = moving_average(returns, window)
    xs = np.arange(len(smooth)) + window - 1
    ax.plot(xs, smooth, color="tab:blue", lw=2, label=label or f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_state_values(env: GridWorld, V, title: str = "State values V(s)",
                      ax=None, annotate: bool = True):
    """Heatmap of a state-value function laid out on the grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(1.4 * env.n_cols, 1.4 * env.n_rows))
    grid = np.array(V, dtype=float).reshape(env.n_rows, env.n_cols)
    masked = np.ma.array(grid, mask=[[(r, c) in env.walls
                                      for c in range(env.n_cols)]
                                     for r in range(env.n_rows)])
    im = ax.imshow(masked, cmap="RdYlGn", origin="upper")
    if annotate:
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                if (r, c) in env.walls:
                    ax.text(c, r, "WALL", ha="center", va="center", fontsize=8)
                else:
                    ax.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center",
                            fontsize=9)
    ax.set_xticks(range(env.n_cols))
    ax.set_yticks(range(env.n_rows))
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_policy(env: GridWorld, policy, title: str = "Greedy policy", ax=None):
    """Draw arrows for a deterministic policy (array of action indices)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(1.4 * env.n_cols, 1.4 * env.n_rows))
    ax.set_xlim(-0.5, env.n_cols - 0.5)
    ax.set_ylim(env.n_rows - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, env.n_cols, 1))
    ax.set_yticks(np.arange(-0.5, env.n_rows, 1))
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    policy = np.asarray(policy).reshape(env.n_rows, env.n_cols)
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if (r, c) in env.walls:
                ax.text(c, r, "WALL", ha="center", va="center", fontsize=8)
            elif (r, c) in env.terminals:
                mark = "G" if env.terminals[(r, c)] > 0 else "H"
                ax.text(c, r, mark, ha="center", va="center", fontsize=14,
                        fontweight="bold")
            else:
                ax.text(c, r, ACTION_ARROWS[int(policy[r, c])], ha="center",
                        va="center", fontsize=18)
    ax.set_title(title)
    return ax


def plot_action_values(env: GridWorld, Q, title: str = "Action values Q(s,a)",
                       ax=None):
    """Show each cell split into 4 triangles coloured by Q(s, a)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(1.6 * env.n_cols, 1.6 * env.n_rows))
    Q = np.asarray(Q).reshape(env.n_states, env.n_actions)
    vmin, vmax = np.min(Q), np.max(Q)
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin, vmax)
    # triangle vertices (in cell-local coords) for UP, RIGHT, DOWN, LEFT
    tris = {
        0: [(-.5, -.5), (.5, -.5), (0, 0)],
        1: [(.5, -.5), (.5, .5), (0, 0)],
        2: [(.5, .5), (-.5, .5), (0, 0)],
        3: [(-.5, .5), (-.5, -.5), (0, 0)],
    }
    for s in range(env.n_states):
        r, c = env.s_to_rc(s)
        if (r, c) in env.walls:
            continue
        for a in range(env.n_actions):
            pts = np.array([(c + dx, r + dy) for dx, dy in tris[a]])
            ax.fill(pts[:, 0], pts[:, 1], color=cmap(norm(Q[s, a])),
                    edgecolor="gray", lw=0.5)
    ax.set_xlim(-0.5, env.n_cols - 0.5)
    ax.set_ylim(env.n_rows - 0.5, -0.5)
    ax.set_xticks(range(env.n_cols))
    ax.set_yticks(range(env.n_rows))
    ax.set_title(title)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 fraction=0.046, pad=0.04)
    return ax
