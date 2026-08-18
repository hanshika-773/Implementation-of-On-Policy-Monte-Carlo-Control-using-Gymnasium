# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm



## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode_idx in range(num_episodes):
    episode = generate_episode(epsilon)

    # Calculate returns (G) for each state-action pair in the episode
    G = 0
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = reward + gamma * G

        # Check if the (state, action) pair is encountered for the first time in the episode
        # (First-visit Monte Carlo)
        first_visit = True
        for prev_t in range(t):
            prev_state, prev_action, _ = episode[prev_t]
            if state == prev_state and action == prev_action:
                first_visit = False
                break

        if first_visit:
            # Update Q-value
            Q[state, action] = Q[state, action] + alpha * (G - Q[state, action])

    # Store total reward for the episode
    total_episode_reward = sum([reward for _, _, reward in episode])
    episode_rewards.append(total_episode_reward)

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


```

---

## Output

```

Final Q-table:
[[0.748 0.811 0.766 0.773]
 [0.709 0.    0.84  0.659]
 [0.713 0.858 0.748 0.703]
 [0.757 0.    0.44  0.363]
 [0.749 0.819 0.    0.73 ]
 [0.    0.    0.    0.   ]
 [0.    0.967 0.    0.849]
 [0.    0.    0.    0.   ]
 [0.791 0.    0.902 0.688]
 [0.863 0.971 0.87  0.   ]
 [0.888 0.989 0.    0.961]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.867 0.988 0.964]
 [0.98  0.99  1.    0.979]
 [0.    0.    0.    0.   ]]

Estimated State-Value Function:
[[0.811 0.84  0.858 0.757]
 [0.819 0.    0.967 0.   ]
 [0.902 0.971 0.989 0.   ]
 [0.    0.988 1.    0.   ]]
Name: Hanshika Varthini R
Register Number:   212223240046   

Learned Policy:
[['D' 'R' 'D' 'L']
 ['D' 'L' 'D' 'L']
 ['R' 'D' 'D' 'L']
 ['L' 'R' 'R' 'L']]

Average reward over last 1000 episodes: 0.945
```
<img width="452" height="646" alt="image" src="https://github.com/user-attachments/assets/fb990d71-876a-46aa-b640-9d75f085ee3b" />

<img width="779" height="532" alt="image" src="https://github.com/user-attachments/assets/712340c4-3ded-4db2-a972-90025cd06362" />

<img width="388" height="474" alt="image" src="https://github.com/user-attachments/assets/e9488f2d-1136-46c3-9e92-f3c5309ebb2f" />

<img width="554" height="380" alt="image" src="https://github.com/user-attachments/assets/98c2bed9-8697-4095-9745-2d8c24f8209c" />

## Result
Thus, the On-Policy Monte Carlo Control algorithm was successfully implemented, and the optimal policy and value function were obtained using the Gymnasium environment.

## Inference
The on-policy Monte Carlo control method effectively enabled the agent to learn from episodic experiences. By maintaining an epsilon-greedy policy, the agent successfully balanced exploration and exploitation, leading to convergence toward the optimal policy and improved cumulative rewards over time.


