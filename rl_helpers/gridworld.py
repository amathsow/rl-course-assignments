"""A small, fully-observable GridWorld MDP.

This environment is deliberately simple and *white-box*: it exposes the full
transition model ``P[s][a] -> [(prob, next_state, reward, done), ...]`` so that
students can implement Dynamic Programming (policy / value iteration) directly,
and can also step through it like a Gymnasium environment for model-free methods.

Grid layout (row 0 is the TOP):

    +----+----+----+----+
    | S  |    |    | G  |   G = goal   (reward +1, terminal)
    +----+----+----+----+
    |    | ## |    | H  |   H = hole   (reward -1, terminal)
    +----+----+----+----+
    |    |    |    |    |   ## = wall  (impassable)
    +----+----+----+----+

Actions: 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT.

The environment supports a ``slip`` probability that makes actions stochastic
(the agent moves in a perpendicular direction with probability ``slip``), which
is what makes DP and RL interesting rather than trivial path-finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Action constants -----------------------------------------------------------
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTION_ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}
# (row, col) deltas for each action
_DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


@dataclass
class GridWorld:
    """A tabular GridWorld MDP with an explicit transition model.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions.
    start : (int, int)
        Starting cell (row, col).
    terminals : dict[(int, int), float]
        Mapping from terminal cell to the reward received on entering it.
    walls : set[(int, int)]
        Impassable cells. Trying to move into a wall keeps the agent in place.
    step_reward : float
        Reward received on every non-terminal transition (usually negative to
        encourage short paths).
    slip : float
        Probability that the agent slips. With probability ``1 - slip`` it moves
        in the intended direction; the remaining ``slip`` is split equally
        between the two perpendicular directions.
    gamma : float
        Discount factor (stored here for convenience; DP code reads it).
    """

    n_rows: int = 3
    n_cols: int = 4
    start: tuple[int, int] = (0, 0)
    terminals: dict = field(default_factory=lambda: {(0, 3): 1.0, (1, 3): -1.0})
    walls: set = field(default_factory=lambda: {(1, 1)})
    step_reward: float = -0.04
    slip: float = 0.0
    gamma: float = 0.99

    def __post_init__(self):
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        self._build_model()
        # RNG for the Gym-style step() interface
        self._rng = np.random.default_rng()
        self._state = self.rc_to_s(*self.start)

    # -- index helpers --------------------------------------------------------
    def rc_to_s(self, r: int, c: int) -> int:
        return r * self.n_cols + c

    def s_to_rc(self, s: int) -> tuple[int, int]:
        return divmod(s, self.n_cols)

    def is_terminal(self, s: int) -> bool:
        return self.s_to_rc(s) in self.terminals

    def is_wall(self, s: int) -> bool:
        return self.s_to_rc(s) in self.walls

    # -- transition model -----------------------------------------------------
    def _move(self, r: int, c: int, a: int) -> tuple[int, int]:
        """Deterministic move that respects walls and grid boundaries."""
        dr, dc = _DELTAS[a]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= self.n_rows or nc < 0 or nc >= self.n_cols:
            return r, c  # bumped into the outer boundary -> stay
        if (nr, nc) in self.walls:
            return r, c  # bumped into a wall -> stay
        return nr, nc

    def _build_model(self):
        """Populate ``self.P``: P[s][a] -> list of (prob, s', reward, done)."""
        self.P: dict[int, dict[int, list]] = {}
        for s in range(self.n_states):
            self.P[s] = {a: [] for a in ACTIONS}
            r, c = self.s_to_rc(s)
            # Terminal / wall states are absorbing (self-loop, no reward).
            if (r, c) in self.terminals or (r, c) in self.walls:
                for a in ACTIONS:
                    self.P[s][a] = [(1.0, s, 0.0, True)]
                continue
            for a in ACTIONS:
                outcomes: dict[int, float] = {}
                # intended direction + the two perpendicular slips
                intended = 1.0 - self.slip
                perp = self.slip / 2.0
                dirs = [(a, intended), ((a - 1) % 4, perp), ((a + 1) % 4, perp)]
                for move_a, prob in dirs:
                    if prob == 0.0:
                        continue
                    nr, nc = self._move(r, c, move_a)
                    ns = self.rc_to_s(nr, nc)
                    outcomes[ns] = outcomes.get(ns, 0.0) + prob
                for ns, prob in outcomes.items():
                    reward = self.terminals.get(self.s_to_rc(ns), self.step_reward)
                    done = self.s_to_rc(ns) in self.terminals
                    self.P[s][a].append((prob, ns, reward, done))
        return self.P

    # -- Gymnasium-style interface (for model-free methods) -------------------
    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self.rc_to_s(*self.start)
        return self._state

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """Sample one transition from the model. Returns (s', reward, done, info)."""
        transitions = self.P[self._state][action]
        probs = [t[0] for t in transitions]
        idx = self._rng.choice(len(transitions), p=probs)
        prob, ns, reward, done = transitions[idx]
        self._state = ns
        return ns, reward, done, {"prob": prob}

    # -- rendering ------------------------------------------------------------
    def render(self, mode: str = "rgb_array"):
        """Render the current state.

        mode='rgb_array' returns an (H, W, 3) uint8 image (so episodes can be
        turned into GIFs and shown inside the notebook, letting students *watch*
        the learned agent move). mode='ansi' returns a text grid.
        """
        if mode == "ansi":
            return self._render_ansi_state()
        return self._render_rgb()

    def _render_ansi_state(self) -> str:
        rows = []
        for r in range(self.n_rows):
            cells = []
            for c in range(self.n_cols):
                s = self.rc_to_s(r, c)
                if (r, c) in self.walls:
                    cells.append("##")
                elif s == self._state:
                    cells.append(" A")  # the agent
                elif (r, c) in self.terminals:
                    cells.append(" G" if self.terminals[(r, c)] > 0 else " H")
                else:
                    cells.append(" .")
            rows.append(" ".join(cells))
        return "\n".join(rows)

    def _render_rgb(self, cell_px: int = 64):
        """Draw the grid + agent to an RGB numpy array using matplotlib."""
        import matplotlib
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = Figure(figsize=(self.n_cols, self.n_rows), dpi=cell_px)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-0.5, self.n_cols - 0.5)
        ax.set_ylim(self.n_rows - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if (r, c) in self.walls:
                    color = "#444444"
                elif (r, c) in self.terminals:
                    color = "#4caf50" if self.terminals[(r, c)] > 0 else "#e53935"
                else:
                    color = "#f5f5f5"
                ax.add_patch(matplotlib.patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, facecolor=color,
                    edgecolor="#999999", lw=1))
        ar, ac = self.s_to_rc(self._state)
        ax.add_patch(matplotlib.patches.Circle((ac, ar), 0.3, facecolor="#1e88e5",
                                                edgecolor="black", lw=1.5, zorder=5))
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        return buf[..., :3].copy()

    # -- pretty printing ------------------------------------------------------
    def render_ascii(self) -> str:
        rows = []
        for r in range(self.n_rows):
            cells = []
            for c in range(self.n_cols):
                if (r, c) in self.walls:
                    cells.append("##")
                elif (r, c) in self.terminals:
                    cells.append(" G" if self.terminals[(r, c)] > 0 else " H")
                elif (r, c) == self.start:
                    cells.append(" S")
                else:
                    cells.append(" .")
            rows.append(" ".join(cells))
        return "\n".join(rows)


def make_default_gridworld(slip: float = 0.1, gamma: float = 0.99) -> GridWorld:
    """The 3x4 'Russell & Norvig' style grid used throughout the course."""
    return GridWorld(slip=slip, gamma=gamma)
