"""
2x2x2 Rubik's Cube Solver — Deep Q-Network (DQN)
A full DQN implementation in pure numpy:
  - Epsilon-greedy exploration
  - Experience replay buffer
  - Target network with periodic hard updates
  - Curriculum learning (increasing scramble depth)
  - Hindsight Experience Replay for sample efficiency
  - Greedy + beam search solvers at test time

The agent learns Q(s, a) = expected future reward for taking action a in state s.
At solve time, it greedily picks argmax_a Q(s,a) at each step.
"""

import numpy as np
import time

# 1. CUBE ENVIRONMENT

class Cube2x2:
    """
    2x2x2 Rubik's Cube — 6 face turns: R, R', U, U', F, F'.
    State: 24-element int8 array (sticker colors 0-5).
    Faces (4 stickers each): U(0-3) D(4-7) F(8-11) B(12-15) L(16-19) R(20-23)
    DBL corner is fixed to remove rotational symmetry.
    """

    NUM_ACTIONS = 6
    ACTION_NAMES = ["R", "R'", "U", "U'", "F", "F'"]
    INVERSE = [1, 0, 3, 2, 5, 4]

    MOVE_CYCLES = [
        [(20,21,23,22), (3,12,7,11), (9,1,14,5)],   # R
        [(20,22,23,21), (3,11,7,12), (9,5,14,1)],   # R'
        [(0,2,3,1),   (8,20,12,16), (17,9,21,13)],  # U
        [(0,1,3,2),   (8,16,12,20), (17,13,21,9)],  # U'
        [(8,9,11,10), (2,20,5,19),  (17,3,22,4)],   # F
        [(8,10,11,9), (2,19,5,20),  (17,4,22,3)],   # F'
    ]

    SOLVED = np.array([i // 4 for i in range(24)], dtype=np.int8)

    @staticmethod
    def apply(state, action):
        s = state.copy()
        for a, b, c, d in Cube2x2.MOVE_CYCLES[action]:
            s[a], s[b], s[c], s[d] = s[d], s[a], s[b], s[c]
        return s

    @staticmethod
    def is_solved(state):
        return np.array_equal(state, Cube2x2.SOLVED)

    @staticmethod
    def scramble(n, rng):
        state = Cube2x2.SOLVED.copy()
        last = -1
        moves = []
        for _ in range(n):
            valid = [a for a in range(6) if last < 0 or a != Cube2x2.INVERSE[last]]
            a = rng.choice(valid)
            state = Cube2x2.apply(state, a)
            moves.append(a)
            last = a
        return state, moves

    @staticmethod
    def featurize(state):
        """One-hot: 24 stickers × 6 colors = 144."""
        f = np.zeros(144, dtype=np.float32)
        f[np.arange(24) * 6 + state.astype(np.int32)] = 1.0
        return f


# 2. REPLAY BUFFER

class ReplayBuffer:
    """Fixed-size ring buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity=200_000):
        self.cap = capacity
        self.buf = []
        self.pos = 0

    def push(self, s, a, r, s2, done):
        item = (s, a, r, s2, done)
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
        self.pos = (self.pos + 1) % self.cap

    def sample(self, n, rng):
        idx = rng.choice(len(self.buf), size=min(n, len(self.buf)), replace=False)
        batch = [self.buf[i] for i in idx]
        return (
            np.array([b[0] for b in batch]),
            np.array([b[1] for b in batch], dtype=np.int32),
            np.array([b[2] for b in batch], dtype=np.float32),
            np.array([b[3] for b in batch]),
            np.array([b[4] for b in batch], dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# 3. Q-NETWORK (Numpy MLP)

class QNetwork:
    """
    Two-hidden-layer MLP: 144 → h1 → h2 → 6
    ReLU activations, He init, Adam optimizer.
    """

    def __init__(self, h1=256, h2=128, lr=1e-3):
        self.lr = lr
        rng = np.random.default_rng(42)
        self.W1 = (rng.normal(0, 1, (144, h1)) * np.sqrt(2/144)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = (rng.normal(0, 1, (h1, h2)) * np.sqrt(2/h1)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = (rng.normal(0, 1, (h2, 6)) * np.sqrt(2/h2)).astype(np.float32)
        self.b3 = np.zeros(6, dtype=np.float32)
        self._params = ['W1','b1','W2','b2','W3','b3']
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self.t = 0

    def forward(self, x):
        """x: (batch,144) or (144,). Returns Q-values shape (batch,6) or (6,)."""
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis]
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._a1 = np.maximum(0, self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = np.maximum(0, self._z2)
        out = self._a2 @ self.W3 + self.b3
        return out[0] if single else out

    def train_on_batch(self, states, actions, targets):
        """
        MSE loss only on the Q-value of the taken action.
        states:  (B,144)
        actions: (B,) int
        targets: (B,) float — the TD target for Q(s,a)
        """
        B = states.shape[0]
        q_all = self.forward(states)                       # (B,6)
        q_sa = q_all[np.arange(B), actions]                # (B,)
        td_err = q_sa - targets                            # (B,)
        loss = np.mean(td_err ** 2)

        # Backprop — gradient only flows through the taken action
        dq = np.zeros_like(q_all)
        dq[np.arange(B), actions] = (2.0 / B) * td_err

        dW3 = self._a2.T @ dq
        db3 = dq.sum(0)
        da2 = dq @ self.W3.T
        dz2 = da2 * (self._z2 > 0)
        dW2 = self._a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self._z1 > 0)
        dW1 = self._x.T @ dz1
        db1 = dz1.sum(0)

        grads = {'W1':dW1,'b1':db1,'W2':dW2,'b2':db2,'W3':dW3,'b3':db3}
        for k in grads:
            np.clip(grads[k], -1, 1, out=grads[k])

        self.t += 1
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        for k in self._params:
            g = grads[k]
            self.m[k] = b1_ * self.m[k] + (1 - b1_) * g
            self.v[k] = b2_ * self.v[k] + (1 - b2_) * g ** 2
            mh = self.m[k] / (1 - b1_ ** self.t)
            vh = self.v[k] / (1 - b2_ ** self.t)
            p = getattr(self, k)
            p -= self.lr * mh / (np.sqrt(vh) + eps)

        return loss

    def copy_to(self, other):
        for k in self._params:
            setattr(other, k, getattr(self, k).copy())


# 4. DQN AGENT

class DQNAgent:
    def __init__(self, h1=256, h2=128, lr=5e-4, gamma=0.97,
                 batch_size=64, target_update=400, seed=42):
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.rng = np.random.default_rng(seed)

        self.q = QNetwork(h1, h2, lr)
        self.q_target = QNetwork(h1, h2, lr)
        self.q.copy_to(self.q_target)

        self.buffer = ReplayBuffer(200_000)
        self.steps = 0

    def act(self, feat, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, 6))
        return int(np.argmax(self.q.forward(feat)))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        s, a, r, s2, d = self.buffer.sample(self.batch_size, self.rng)
        q_next = self.q_target.forward(s2)
        targets = r + self.gamma * np.max(q_next, axis=1) * (1 - d)
        loss = self.q.train_on_batch(s, a, targets)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.q.copy_to(self.q_target)
        return loss


# 5. HINDSIGHT EXPERIENCE REPLAY (HER)

def fill_buffer_with_hindsight(agent, n_scrambles=3000, max_depth=8):
    """
    Pre-fill the replay buffer with expert-quality transitions.
    Scramble from solved, then reverse the path — each reverse step
    is a transition that leads closer to solved.
    This gives the DQN a massive head start.
    """
    rng = agent.rng
    for _ in range(n_scrambles):
        depth = rng.integers(1, max_depth + 1)
        state, moves = Cube2x2.scramble(depth, rng)

        # Walk backwards along the scramble (reverse = inverse moves)
        path_states = [Cube2x2.SOLVED.copy()]
        s = Cube2x2.SOLVED.copy()
        for m in moves:
            s = Cube2x2.apply(s, m)
            path_states.append(s.copy())

        # path_states[depth] = scrambled, path_states[0] = solved
        # Transition: from state[i], take inverse of moves[i-1] → state[i-1]
        for i in range(len(moves), 0, -1):
            s_cur  = path_states[i]
            action = Cube2x2.INVERSE[moves[i - 1]]
            s_next = path_states[i - 1]
            done   = float(Cube2x2.is_solved(s_next))
            reward = 10.0 if done else -0.1

            feat_cur  = Cube2x2.featurize(s_cur)
            feat_next = Cube2x2.featurize(s_next)
            agent.buffer.push(feat_cur, action, reward, feat_next, done)


# 6. TRAINING

def train(max_depth=7, episodes_per_depth=800, max_ep_steps=20,
          eps_start=0.8, eps_end=0.02, her_scrambles=8000, her_depth=9,
          update_freq=3):
    """
    Curriculum DQN training:
    1. Pre-fill buffer with hindsight replay
    2. For each curriculum depth (1..max_depth):
       - Run episodes with epsilon-greedy exploration
       - Train Q-network on replay batches after each step
    """
    agent = DQNAgent(h1=256, h2=128, lr=5e-4, gamma=0.97,
                     batch_size=64, target_update=300)

    print(f"  Curriculum: depth 1 → {max_depth}")
    print(f"  Episodes/depth: {episodes_per_depth}  |  Max steps: {max_ep_steps}")
    print(f"  HER pre-fill: {her_scrambles} scrambles (depth 1-{her_depth})")

    # ── Phase 1: Hindsight Experience Replay pre-fill ──
    t0 = time.time()
    fill_buffer_with_hindsight(agent, her_scrambles, her_depth)
    print(f"  Buffer filled: {len(agent.buffer)} transitions")

    # Pre-train on the HER data
    for _ in range(1000):
        agent.update()
    print(f"  Pre-training done ({time.time()-t0:.1f}s)\n")

    # ── Phase 2: Curriculum RL ──
    total_episodes = 0

    for depth in range(1, max_depth + 1):
        # Refresh buffer with HER at current depth
        fill_buffer_with_hindsight(agent, 1000, depth + 1)
        for _ in range(200):
            agent.update()

        solves = 0
        solve_steps = []
        ep_rewards = []
        loss_accum = 0.0
        loss_count = 0

        ep_start = max(eps_start - 0.08 * (depth - 1), 0.15)
        ep_end_  = eps_end

        t_depth = time.time()
        env_step = 0

        for ep in range(episodes_per_depth):
            frac = ep / max(episodes_per_depth - 1, 1)
            epsilon = ep_start + (ep_end_ - ep_start) * frac

            # Random scramble depth in [1, depth]
            d = agent.rng.integers(1, depth + 1)
            state, _ = Cube2x2.scramble(d, agent.rng)
            feat = Cube2x2.featurize(state)
            ep_reward = 0.0
            solved = False

            for step in range(max_ep_steps):
                action = agent.act(feat, epsilon)
                next_state = Cube2x2.apply(state, action)
                next_feat = Cube2x2.featurize(next_state)

                if Cube2x2.is_solved(next_state):
                    reward, done, solved = 10.0, 1.0, True
                else:
                    reward, done = -0.1, 0.0

                agent.buffer.push(feat, action, reward, next_feat, done)
                ep_reward += reward
                env_step += 1

                if env_step % update_freq == 0:
                    loss = agent.update()
                    if loss > 0:
                        loss_accum += loss
                        loss_count += 1

                state, feat = next_state, next_feat
                if done:
                    solve_steps.append(step + 1)
                    break

            if solved:
                solves += 1
            ep_rewards.append(ep_reward)
            total_episodes += 1

            # Mid-depth progress
            if (ep + 1) % 400 == 0:
                sr = solves / (ep + 1) * 100
                print(f"    depth {depth} | ep {ep+1:4d}/{episodes_per_depth} | "
                      f"solve={sr:5.1f}% | eps={epsilon:.2f} | "
                      f"buf={len(agent.buffer)}", flush=True)

        elapsed = time.time() - t_depth
        rate = solves / episodes_per_depth * 100
        avg_r = np.mean(ep_rewards)
        avg_steps = np.mean(solve_steps) if solve_steps else float('nan')
        avg_loss = loss_accum / max(loss_count, 1)

        bar_len = 40
        filled = int(bar_len * rate / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"\n  ── Depth {depth} Summary ──")
        print(f"  Solve rate:  [{bar}] {rate:.1f}%")
        print(f"  Avg moves when solved: {avg_steps:.2f}")
        print(f"  Avg reward: {avg_r:.2f}  |  Avg loss: {avg_loss:.4f}")
        print(f"  Time: {elapsed:.1f}s  |  Total episodes: {total_episodes}\n")

    total = time.time() - t0
    print(f"  Training complete in {total:.1f}s  ({total_episodes} episodes)")

    return agent


# 7. SOLVERS

def greedy_solve(q_net, state, max_steps=30):
    """Pick argmax Q at each step. Track visited states to avoid loops."""
    if Cube2x2.is_solved(state):
        return []
    cur = state.copy()
    moves = []
    visited = {cur.tobytes()}

    for _ in range(max_steps):
        feat = Cube2x2.featurize(cur)
        q_vals = q_net.forward(feat)

        # Sort actions by Q-value (descending) and pick best unvisited child
        order = np.argsort(q_vals)[::-1]
        moved = False
        for a in order:
            child = Cube2x2.apply(cur, a)
            if Cube2x2.is_solved(child):
                moves.append(int(a))
                return moves
            key = child.tobytes()
            if key not in visited:
                visited.add(key)
                moves.append(int(a))
                cur = child
                moved = True
                break
        if not moved:
            break
    return None


def beam_solve(q_net, state, beam_width=128, max_steps=20):
    """Beam search guided by Q-values."""
    if Cube2x2.is_solved(state):
        return []
    beam = [(state.copy(), [])]
    visited = {state.tobytes()}

    for _ in range(max_steps):
        candidates = []
        for s, path in beam:
            feat = Cube2x2.featurize(s)
            q_vals = q_net.forward(feat)
            for a in range(6):
                child = Cube2x2.apply(s, a)
                if Cube2x2.is_solved(child):
                    return path + [a]
                key = child.tobytes()
                if key not in visited:
                    visited.add(key)
                    candidates.append((q_vals[a], child, path + [a]))
        if not candidates:
            break
        candidates.sort(key=lambda x: -x[0])  # highest Q first
        beam = [(c[1], c[2]) for c in candidates[:beam_width]]
    return None


# 8. TESTING

def test_agent(agent, num_tests=60, max_depth=11, max_steps=25):
    rng = np.random.default_rng(999)

    for name, solver in [("Greedy",
                           lambda s: greedy_solve(agent.q, s, max_steps)),
                          ("Beam Search (w=128)",
                           lambda s: beam_solve(agent.q, s, 128, max_steps))]:
        print(f"  TESTING: {name}")

        total_s, total_t = 0, 0
        rng = np.random.default_rng(999)

        for depth in range(1, max_depth + 1):
            solved, mc = 0, []
            for _ in range(num_tests):
                state, _ = Cube2x2.scramble(depth, rng)
                result = solver(state)
                if result is not None:
                    solved += 1
                    mc.append(len(result))
            total_s += solved
            total_t += num_tests
            pct = solved / num_tests * 100
            avg = np.mean(mc) if mc else float('nan')
            med = np.median(mc) if mc else float('nan')
            mx  = max(mc) if mc else 0
            print(f"  Depth {depth:2d}: {solved:3d}/{num_tests} ({pct:5.1f}%) | "
                  f"Avg={avg:5.1f}  Med={med:4.1f}  Max={mx:2d}")

        print(f"\n  Overall: {total_s}/{total_t} ({total_s/total_t*100:.1f}%)")


def demo_solves(agent, n=12, depth=6, use_beam=False, bw=128):
    rng = np.random.default_rng(77)
    tag = f"Beam(w={bw})" if use_beam else "Greedy"

    print(f"  DEMO: {tag} — {n} cubes, scramble depth {depth}")

    move_list = []
    for i in range(n):
        state, sc = Cube2x2.scramble(depth, rng)
        sc_str = " ".join(Cube2x2.ACTION_NAMES[a] for a in sc)

        if use_beam:
            result = beam_solve(agent.q, state, bw)
        else:
            result = greedy_solve(agent.q, state)

        if result is not None:
            sol_str = " ".join(Cube2x2.ACTION_NAMES[a] for a in result)
            nm = len(result)
            move_list.append(nm)
            print(f"  #{i+1:2d} | Scramble: {sc_str}")
            print(f"       | Solution ({nm:2d} moves): {sol_str}  ✓")
        else:
            print(f"  #{i+1:2d} | Scramble: {sc_str}")
            print(f"       | ✗ FAILED")

    if move_list:
        print(f"\n  Results: {len(move_list)}/{n} solved | "
              f"Avg={np.mean(move_list):.1f} | "
              f"Min={min(move_list)} | Max={max(move_list)} moves")


def main():
    print()

    agent = train(
        max_depth=7,
        episodes_per_depth=800,
        max_ep_steps=20,
        eps_start=0.8,
        eps_end=0.02,
        her_scrambles=8000,
        her_depth=9,
    )

    test_agent(agent, num_tests=300, max_depth=11, max_steps=25)

    demo_solves(agent, n=12, depth=4, use_beam=False)
    demo_solves(agent, n=12, depth=7, use_beam=True, bw=128)


if __name__ == "__main__":
    main()
