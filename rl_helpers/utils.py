"""Generic RL utilities: seeding, rollouts, epsilon-greedy, GIF recording."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def set_seed(seed: int = 0):
    """Seed python, numpy (and torch if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def epsilon_greedy(Q_s, epsilon: float, n_actions: int, rng=None) -> int:
    """Pick an action epsilon-greedily from an array of action values Q_s."""
    rng = rng or np.random
    if rng.random() < epsilon:
        return int(rng.integers(n_actions) if hasattr(rng, "integers")
                   else rng.randint(n_actions))
    q = np.asarray(Q_s, dtype=float)
    # break ties randomly so the agent doesn't get stuck on the first action
    return int(rng.choice(np.flatnonzero(q == q.max())))


def linear_epsilon(step: int, start: float = 1.0, end: float = 0.05,
                   decay_steps: int = 10_000) -> float:
    """Linearly anneal epsilon from ``start`` to ``end`` over ``decay_steps``."""
    frac = min(1.0, step / decay_steps)
    return start + frac * (end - start)


def run_episode(env, policy_fn, max_steps: int = 500, render: bool = False):
    """Roll out one episode following ``policy_fn(state) -> action``.

    Works with both Gymnasium envs (5-tuple step) and the course GridWorld
    (4-tuple step). Returns (total_reward, n_steps, frames).
    """
    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    total, frames = 0.0, []
    for t in range(max_steps):
        if render and hasattr(env, "render"):
            frames.append(env.render())
        action = policy_fn(state)
        out = env.step(action)
        if len(out) == 5:  # Gymnasium: obs, reward, terminated, truncated, info
            state, reward, terminated, truncated, _ = out
            done = terminated or truncated
        else:              # GridWorld: s', reward, done, info
            state, reward, done, _ = out
        total += reward
        if done:
            break
    return total, t + 1, frames


def greedy_policy_from_Q(Q):
    """Return a deterministic greedy policy array from a Q-table (dict or array)."""
    if isinstance(Q, dict):
        return {s: int(np.argmax(qs)) for s, qs in Q.items()}
    return np.argmax(np.asarray(Q), axis=-1)


def make_q_table(n_states: int, n_actions: int, kind: str = "array"):
    """Create an empty Q representation.

    kind='array' -> numpy zeros; kind='dict' -> defaultdict for large/unknown
    state spaces.
    """
    if kind == "dict":
        return defaultdict(lambda: np.zeros(n_actions))
    return np.zeros((n_states, n_actions))


def save_frames_as_gif(frames, path: str, fps: int = 15):
    """Save a list of RGB frames (H, W, 3 uint8) to an animated GIF."""
    import imageio
    valid = [f for f in frames if f is not None]
    if not valid:
        raise ValueError("No frames to save (did you pass render_mode='rgb_array'?)")
    imageio.mimsave(path, valid, fps=fps, loop=0)
    return path


def record_policy_gif(env, policy_fn, path: str = "agent.gif", max_steps: int = 200,
                      fps: int = 8, episodes: int = 1):
    """Roll out a (trained) policy and save the episode(s) as an animated GIF.

    This lets students *watch the learned agent move* in the environment.
    ``env`` must render RGB frames — for Gymnasium pass ``render_mode='rgb_array'``
    when creating it; the course GridWorld renders RGB by default.

    Returns the path to the saved GIF.
    """
    all_frames = []
    for _ in range(episodes):
        _, _, frames = run_episode(env, policy_fn, max_steps=max_steps, render=True)
        # capture the final frame too so the terminal state is visible
        if hasattr(env, "render"):
            try:
                all_frames.extend(frames + [env.render()])
            except Exception:
                all_frames.extend(frames)
        else:
            all_frames.extend(frames)
    return save_frames_as_gif(all_frames, path, fps=fps)


def show_gif(path: str, width: int | None = None):
    """Display a saved GIF inline in a Jupyter notebook.

    Usage:
        record_policy_gif(env, policy, "agent.gif")
        show_gif("agent.gif")
    """
    from IPython.display import Image, display
    display(Image(filename=path, width=width))


def animate_policy(env, policy_fn, max_steps: int = 200, fps: int = 8,
                   path: str = "agent.gif", width: int | None = 320):
    """One-liner: roll out ``policy_fn`` and show it inline in the notebook."""
    record_policy_gif(env, policy_fn, path=path, max_steps=max_steps, fps=fps)
    show_gif(path, width=width)
    return path
