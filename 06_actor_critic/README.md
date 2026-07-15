# Module 06 — Policy Gradients & Actor-Critic

Instead of learning values and acting greedily, **parameterize the policy directly**
and improve it by gradient ascent on expected return. This is how modern deep RL
handles **stochastic** and **continuous** action spaces.

## What you'll learn
- The **policy gradient theorem** and the `log π · Ψ` update.
- **REINFORCE** (Monte Carlo policy gradient) and why it's high-variance.
- **Actor-Critic (A2C)**: a critic baseline / advantage to cut variance.
- The **entropy bonus** for exploration and **gradient clipping** for stability.
- That the *same* Actor-Critic code scales to a much harder task just by changing
  the environment id.
- How the method extends to **continuous** actions with a Gaussian policy
  (covered conceptually).

## Environments used
- `CartPole-v1` — REINFORCE and Actor-Critic.
- `LunarLander-v3` — the same Actor-Critic applied to a harder control task
  (requires Box2D, installed automatically by `install.sh` / `uv sync`).

## How to work through it
1. Open `assignment/06_actor_critic_assignment.ipynb` and select the
   **`Python (rl-course)`** kernel.
2. Fill in every `# TODO` / `raise NotImplementedError`.
3. Run all cells; Actor-Critic should learn faster and more stably than REINFORCE,
   and you should be able to watch the trained agent balance the pole.

## Tasks
1. Complete the **REINFORCE** update and solve CartPole.
2. Complete the **Actor-Critic** update (`train_a2c`); plot it against REINFORCE and
   explain the variance reduction.
3. Run the *same* `train_a2c` on **LunarLander-v3** and watch the learning curve
   climb; render the trained lander.
4. Read the continuous-actions note and describe what changes (and what stays the
   same) when moving from a categorical to a Gaussian policy.

## Prerequisites
Modules 01, 03, and 04 (PyTorch basics from the DQN section).

## Note
The CartPole runs finish in a couple of minutes on CPU; the LunarLander run trains
for ~600 episodes and takes several more minutes. It shows a clear upward learning
trend rather than a fully "solved" agent — increase `n_episodes` for a better lander.
A GPU is used automatically if available.

## Estimated time
2.5–3 hours.
