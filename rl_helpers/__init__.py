"""rl_helpers: shared utilities for the DIT Reinforcement Learning course.

Importing this package gives notebooks a small, consistent toolbox:

    from rl_helpers import GridWorld, make_default_gridworld
    from rl_helpers import set_seed, epsilon_greedy, run_episode
    from rl_helpers import plot_state_values, plot_policy, plot_learning_curve
    from rl_helpers import animate_policy, record_policy_gif, show_gif   # watch the agent!
"""

from .gridworld import (
    GridWorld,
    make_default_gridworld,
    ACTIONS,
    ACTION_NAMES,
    ACTION_ARROWS,
    UP,
    RIGHT,
    DOWN,
    LEFT,
)
from .utils import (
    set_seed,
    epsilon_greedy,
    linear_epsilon,
    run_episode,
    greedy_policy_from_Q,
    make_q_table,
    save_frames_as_gif,
    record_policy_gif,
    show_gif,
    animate_policy,
)
from .plotting import (
    moving_average,
    plot_learning_curve,
    plot_state_values,
    plot_policy,
    plot_action_values,
)

__all__ = [
    "GridWorld",
    "make_default_gridworld",
    "ACTIONS",
    "ACTION_NAMES",
    "ACTION_ARROWS",
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "set_seed",
    "epsilon_greedy",
    "linear_epsilon",
    "run_episode",
    "greedy_policy_from_Q",
    "make_q_table",
    "save_frames_as_gif",
    "record_policy_gif",
    "show_gif",
    "animate_policy",
    "moving_average",
    "plot_learning_curve",
    "plot_state_values",
    "plot_policy",
    "plot_action_values",
]
