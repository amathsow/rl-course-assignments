# Reinforcement Learning — Hands-on Course

**Dakar Institute of Technology (DIT)**

A practical, notebook-based introduction to Reinforcement Learning. Each module
pairs clear explanations with code you complete yourself, and — because RL is best
understood by *watching* — every algorithm ends by **rendering the trained agent
acting in its environment**.

Built on [**Gymnasium**](https://gymnasium.farama.org/), NumPy, and PyTorch.

---

## Course structure

The course is a ladder — each module builds on the previous one:

| # | Module | Key idea | Environments |
|---|--------|----------|--------------|
| 01 | [Gymnasium hands-on](01_gymnasium_handson/) | The agent–environment loop | CartPole, FrozenLake |
| 02 | [Dynamic Programming](02_dynamic_programming/) | Plan with a **known** model (policy & value iteration) | GridWorld, FrozenLake, Taxi |
| 03 | [Model-free RL](03_model_free/) | Learn from experience (MC, TD, SARSA, Q-learning) | GridWorld, CliffWalking, Blackjack |
| 04 | [Function approximation](04_function_approximation/) | Scale to big/continuous states (tile coding, **DQN**) | MountainCar, CartPole |
| 05 | [Model-based RL](05_model_based/) | **Learn** a model and plan with it (Dyna-Q) | Maze GridWorld, FrozenLake |
| 06 | [Actor-Critic](06_actor_critic/) | Optimise the policy directly (REINFORCE, **A2C**) | CartPole, LunarLander |

Each module folder contains:
- `README.md` — objectives, tasks, and estimated time.
- `assignment/` — the notebook **you** work in (with `# TODO` gaps to fill).

---

## Setup

This project uses [**uv**](https://docs.astral.sh/uv/) for fast, reproducible
environments.

### Quick start (recommended)
Clone the repository, then run the one-shot installer from the repo root:
```bash
git clone https://github.com/amathsow/rl-course-assignments.git
cd rl-course-assignments
./install.sh
```
`install.sh` installs `uv` (if missing), creates the virtual environment with every
dependency, and registers the Jupyter kernel — everything below, in one command.

### Manual setup (if you prefer)
```bash
# 1. install uv (skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. create the environment and install all dependencies
uv sync

# 3. register the Jupyter kernel
uv run python -m ipykernel install --user --name rl-course --display-name "Python (rl-course)"
```

### Launch Jupyter
```bash
uv run jupyter lab # or: uv run jupyter notebook
```
Open any notebook and select the **`Python (rl-course)`** kernel (top-right).

> **Tip:** you can also run a notebook end-to-end from the terminal:
> ```bash
> uv run jupyter nbconvert --to notebook --execute --inplace <notebook.ipynb>
> ```

---

## The `rl_helpers` package

A small shared toolbox so notebooks focus on the RL, not the boilerplate:

```python
from rl_helpers import GridWorld, make_default_gridworld # a white-box MDP
from rl_helpers import set_seed, epsilon_greedy, run_episode # RL utilities
from rl_helpers import plot_state_values, plot_policy, plot_learning_curve
from rl_helpers import animate_policy # <-- watch a trained agent as a GIF!
```

The star helper is **`animate_policy(env, policy_fn)`**: it rolls out a policy,
records frames, and displays the episode inline as an animation — used at the end
of every module.

Notebooks locate this package automatically (a small bootstrap cell walks up the
directory tree), so they work from any subfolder.

---

## How to use this course

1. Read the module `README.md`.
2. Open the notebook in `assignment/` and select the `Python (rl-course)` kernel.
3. Fill in every `# TODO` / `raise NotImplementedError`. The assignment is the same
   notebook as the reference, with the key lines left blank for you to implement.
4. Run all cells top-to-bottom and watch your trained agent act at the end.

---

## Requirements
- Python 3.10–3.12
- ~2–3 GB disk (mostly PyTorch)
- A GPU is optional; the deep-RL modules (04, 06) run on CPU in a few minutes and
  use a GPU automatically if one is present.

---

## Suggested schedule (12-week course)
| Weeks | Modules |
|---|---|
| 1–2 | 01 Gymnasium + RL foundations |
| 3–4 | 02 Dynamic Programming |
| 5–7 | 03 Model-free RL |
| 8–9 | 04 Function approximation |
| 10 | 05 Model-based RL |
| 11–12 | 06 Actor-Critic + mini-project |

---

## References
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018) — the
  canonical text this course follows.
- [Gymnasium documentation](https://gymnasium.farama.org/)
- David Silver's RL course (UCL).

---

*Happy learning — and don't forget to watch your agents move!*
