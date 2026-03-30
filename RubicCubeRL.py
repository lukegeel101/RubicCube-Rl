"""
2x2x2 Rubik's Cube Solver using Reinforcement Learning
========================================================
Uses Autodidactic Iteration (ADI). The same family of algorithms as DeepCubeA, but adapted for a 2x2x2 cube with a pure-numpy neural net.

How it works:
  1. Generate training states by scrambling from SOLVED state
  2. For each state, compute target value = 1 + min(value(child)) over all moves (the solved state has value 0)
  3. Train a neural network to predict this value function
  4. At solve time, greedily pick moves that minimize predicted distance

This is a form of approximate value iteration.
"""

import numpy as np
from collections import deque
import time

# 1. CUBE ENVIRONMENT

class Cube2x2:
    """
    2x2x2 Rubik's Cube with 6 face moves: R, R', U, U', F, F'.
    State: 24-element int8 array (sticker colors 0-5).
    Faces: U(0-3), D(4-7), F(8-11), B(12-15), L(16-19), R(20-23)
    We fix DBL corner → only R, U, F and inverses needed.
    """
    
    NUM_ACTIONS = 6
    ACTION_NAMES = ["R", "R'", "U", "U'", "F", "F'"]
    INVERSE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
    
    # Permutation cycles for each move: (a,b,c,d) means a→b→c→d→a
    MOVE_CYCLES = [
        # 0: R   
        [(20,21,23,22), (3,12,7,11), (9,1,14,5)],
        # 1: R'
        [(20,22,23,21), (3,11,7,12), (9,5,14,1)],
        # 2: U   
        [(0,2,3,1), (8,20,12,16), (17,9,21,13)],
        # 3: U'
        [(0,1,3,2), (8,16,12,20), (17,13,21,9)],
        # 4: F   
        [(8,9,11,10), (2,20,5,19), (17,3,22,4)],
        # 5: F'
        [(8,10,11,9), (2,19,5,20), (17,4,22,3)],
    ]
    
    SOLVED = np.array([i // 4 for i in range(24)], dtype=np.int8)
    
    @staticmethod
    def apply_move(state, action):
        """Apply move to state, return new state."""
        s = state.copy()
        for cycle in Cube2x2.MOVE_CYCLES[action]:
            a, b, c, d = cycle
            s[a], s[b], s[c], s[d] = s[d], s[a], s[b], s[c]
        return s
    
    @staticmethod
    def is_solved(state):
        return np.array_equal(state, Cube2x2.SOLVED)
    
    @staticmethod
    def scramble(n_moves, rng):
        """Return state after n random moves from solved (no immediate reversals)."""
        state = Cube2x2.SOLVED.copy()
        last = -1
        moves = []
        for _ in range(n_moves):
            valid = [a for a in range(6) if last < 0 or a != Cube2x2.INVERSE[last]]
            action = rng.choice(valid)
            state = Cube2x2.apply_move(state, action)
            moves.append(action)
            last = action
        return state, moves
    
    @staticmethod
    def state_to_features(state):
        """One-hot encode: 24 stickers x 6 colors = 144 features."""
        feat = np.zeros(144, dtype=np.float32)
        feat[np.arange(24) * 6 + state.astype(np.int32)] = 1.0
        return feat
    
    @staticmethod
    def batch_to_features(states):
        """Vectorized one-hot encoding for a batch of states."""
        n = len(states)
        feat = np.zeros((n, 144), dtype=np.float32)
        idx = np.arange(24) * 6  # base indices
        for i, s in enumerate(states):
            feat[i, idx + s.astype(np.int32)] = 1.0
        return feat


# 2. NEURAL NETWORK (Numpy)

class Network:
    """
    MLP with shared trunk + value head (predicts distance) + policy head (best move).
    Architecture: 144 -> hidden -> hidden -> value(1) + policy(6)
    Uses ReLU, He init, Adam optimizer.
    """
    
    def __init__(self, hidden=128, lr=1e-3):
        self.lr = lr
        rng = np.random.default_rng(42)
        
        # Shared layers
        self.W1 = rng.normal(0, np.sqrt(2/144), (144, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2/hidden), (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        
        # Value head
        self.Wv = rng.normal(0, np.sqrt(2/hidden), (hidden, 1)).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)
        
        # Policy head
        self.Wp = rng.normal(0, np.sqrt(2/hidden), (hidden, 6)).astype(np.float32)
        self.bp = np.zeros(6, dtype=np.float32)
        
        # Adam state
        self.params = ['W1','b1','W2','b2','Wv','bv','Wp','bp']
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.t = 0
    
    def forward(self, x):
        """x: (batch, 144) or (144,). Returns (value, policy_logits)."""
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        
        self._x = x
        self._h1 = x @ self.W1 + self.b1
        self._a1 = np.maximum(0, self._h1)
        self._h2 = self._a1 @ self.W2 + self.b2
        self._a2 = np.maximum(0, self._h2)
        
        val = self._a2 @ self.Wv + self.bv      # (batch, 1)
        pol = self._a2 @ self.Wp + self.bp       # (batch, 6)
        
        if single:
            return val[0, 0], pol[0]
        return val[:, 0], pol
    
    def train_step(self, x, target_values, target_policies):
        """
        One gradient step.
        target_values: (batch,) — distance to solved
        target_policies: (batch, 6) — one-hot best move
        """
        batch = x.shape[0]
        val, pol = self.forward(x)
        
        # Value loss (MSE)
        val_err = val - target_values
        val_loss = np.mean(val_err ** 2)
        
        # Policy loss (cross-entropy via softmax)
        pol_max = pol - pol.max(axis=1, keepdims=True)
        exp_pol = np.exp(pol_max)
        softmax = exp_pol / exp_pol.sum(axis=1, keepdims=True)
        pol_loss = -np.mean(np.sum(target_policies * np.log(softmax + 1e-8), axis=1))
        
        # ── Backward pass ──
        d_pol = (softmax - target_policies) / batch
        d_val = (2 * val_err / batch).reshape(-1, 1)
        
        # Head gradients
        dWp = self._a2.T @ d_pol
        dbp = d_pol.sum(axis=0)
        dWv = self._a2.T @ d_val
        dbv = d_val.sum(axis=0).ravel()
        
        # Backprop through trunk
        d_a2 = d_pol @ self.Wp.T + d_val @ self.Wv.T
        d_h2 = d_a2 * (self._h2 > 0)
        dW2 = self._a1.T @ d_h2
        db2 = d_h2.sum(axis=0)
        
        d_a1 = d_h2 @ self.W2.T
        d_h1 = d_a1 * (self._h1 > 0)
        dW1 = self._x.T @ d_h1
        db1 = d_h1.sum(axis=0)
        
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,
                 'Wv': dWv, 'bv': dbv, 'Wp': dWp, 'bp': dbp}
        for k in grads:
            np.clip(grads[k], -5, 5, out=grads[k])
        
        # Adam update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k in self.params:
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * g ** 2
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            param = getattr(self, k)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        
        return val_loss, pol_loss


# 3. TRAINING (Autodidactic Iteration) 

def generate_training_data(net, batch_size, max_depth, rng):
    """
    Generate training data:
    1. Scramble from solved to random depth (1..max_depth)
    2. For each state, try all 6 moves
    3. Value target = 0 if solved, else 1 + min(net_value(child))
    4. Policy target = one-hot of best move
    """
    states = []
    target_vals = []
    target_pols = []
    
    for _ in range(batch_size):
        depth = rng.integers(1, max_depth + 1)
        state, _ = Cube2x2.scramble(depth, rng)
        
        # Evaluate all children
        child_vals = np.zeros(6)
        for a in range(6):
            child = Cube2x2.apply_move(state, a)
            if Cube2x2.is_solved(child):
                child_vals[a] = 0.0
            else:
                feat = Cube2x2.state_to_features(child)
                v, _ = net.forward(feat)
                child_vals[a] = v
        
        best_action = np.argmin(child_vals)
        target_val = 1.0 + child_vals[best_action]
        
        if child_vals[best_action] == 0.0:
            target_val = 1.0
        
        target_pol = np.zeros(6, dtype=np.float32)
        target_pol[best_action] = 1.0
        
        states.append(state)
        target_vals.append(target_val)
        target_pols.append(target_pol)
    
    # Add solved state (value = 0)
    states.append(Cube2x2.SOLVED.copy())
    target_vals.append(0.0)
    target_pols.append(np.ones(6, dtype=np.float32) / 6)
    
    features = Cube2x2.batch_to_features(states)
    return features, np.array(target_vals, dtype=np.float32), np.array(target_pols)


def train(num_iterations=200, batch_size=256, max_depth=8, lr=5e-4, hidden=128):
    """Train value/policy network using Autodidactic Iteration."""
    net = Network(hidden=hidden, lr=lr)
    rng = np.random.default_rng(42)
    
    print("  2x2x2 RUBIK'S CUBE — RL TRAINING (Autodidactic Iteration)")
    print(f"  Iterations: {num_iterations}  |  Batch size: {batch_size}")
    print(f"  Max scramble depth: {max_depth}")
    print(f"  Network: 144 -> {hidden} -> {hidden} -> value(1) + policy(6)")
    
    t_start = time.time()
    
    for it in range(1, num_iterations + 1):
        features, target_vals, target_pols = \
            generate_training_data(net, batch_size, max_depth, rng)
        
        # Multiple gradient passes on this data
        for _ in range(3):
            perm = rng.permutation(len(features))
            features = features[perm]
            target_vals = target_vals[perm]
            target_pols = target_pols[perm]
            
            for start in range(0, len(features), 64):
                end = min(start + 64, len(features))
                v_loss, p_loss = net.train_step(
                    features[start:end],
                    target_vals[start:end],
                    target_pols[start:end]
                )
        
        if it % 25 == 0 or it == 1:
            elapsed = time.time() - t_start
            # Quick test
            n_test = 40
            solved = 0
            for _ in range(n_test):
                d = rng.integers(1, max_depth + 1)
                s, _ = Cube2x2.scramble(d, rng)
                if greedy_solve(net, s, max_steps=20) is not None:
                    solved += 1
            rate = solved / n_test * 100
            print(f"  Iter {it:4d}/{num_iterations} | "
                  f"VLoss={v_loss:7.3f} | PLoss={p_loss:6.3f} | "
                  f"SolveRate={rate:5.1f}% | {elapsed:.1f}s", flush=True)
    
    total_time = time.time() - t_start
    print(f"\n  Training complete in {total_time:.1f}s")
    return net


# 4. SOLVERS

def greedy_solve(net, state, max_steps=30):
    """Greedy: always pick the move with lowest predicted value."""
    if Cube2x2.is_solved(state):
        return []
    
    current = state.copy()
    moves = []
    visited = set()
    visited.add(current.tobytes())
    
    for _ in range(max_steps):
        best_val = float('inf')
        best_action = 0
        best_child = None
        
        for a in range(6):
            child = Cube2x2.apply_move(current, a)
            if Cube2x2.is_solved(child):
                moves.append(a)
                return moves
            
            key = child.tobytes()
            if key in visited:
                continue
            
            feat = Cube2x2.state_to_features(child)
            v, _ = net.forward(feat)
            if v < best_val:
                best_val = v
                best_action = a
                best_child = child
        
        if best_child is None:
            break
        
        moves.append(best_action)
        current = best_child
        visited.add(current.tobytes())
    
    return None


def beam_solve(net, state, beam_width=64, max_steps=20):
    """Beam search: explore top-k states by predicted value."""
    if Cube2x2.is_solved(state):
        return []
    
    beam = [(state.copy(), [])]
    visited = {state.tobytes()}
    
    for step in range(max_steps):
        candidates = []
        
        for s, moves in beam:
            for a in range(6):
                child = Cube2x2.apply_move(s, a)
                if Cube2x2.is_solved(child):
                    return moves + [a]
                
                key = child.tobytes()
                if key in visited:
                    continue
                visited.add(key)
                
                feat = Cube2x2.state_to_features(child)
                v, _ = net.forward(feat)
                candidates.append((v, child, moves + [a]))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: x[0])
        beam = [(c[1], c[2]) for c in candidates[:beam_width]]
    
    return None


# 5. TESTING & DEMOS

def test_agent(net, num_tests=50, max_depth=12, max_steps=25):
    """Test on random scrambles, report solve rates and move counts."""
    rng = np.random.default_rng(999)
    
    for method_name, solver in [("Greedy", lambda s: greedy_solve(net, s, max_steps)),
                                 ("Beam Search (w=128)", lambda s: beam_solve(net, s, 128, max_steps))]:
        print(f"  TESTING: {method_name}")
        
        total_solved = 0
        total_tests = 0
        rng = np.random.default_rng(999)  # reset for fair comparison
        
        for depth in range(1, max_depth + 1):
            solved = 0
            move_counts = []
            
            for _ in range(num_tests):
                state, _ = Cube2x2.scramble(depth, rng)
                result = solver(state)
                if result is not None:
                    solved += 1
                    move_counts.append(len(result))
            
            total_solved += solved
            total_tests += num_tests
            pct = solved / num_tests * 100
            avg = np.mean(move_counts) if move_counts else float('nan')
            med = np.median(move_counts) if move_counts else float('nan')
            mx = max(move_counts) if move_counts else 0
            
            print(f"  Depth {depth:2d}: {solved:3d}/{num_tests} solved ({pct:5.1f}%) | "
                  f"Avg={avg:5.1f}  Med={med:4.1f}  Max={mx:2d}")
        
        print(f"\n  Overall: {total_solved}/{total_tests} "
              f"({total_solved/total_tests*100:.1f}%)")


def demo_solves(net, n=12, scramble_depth=6, use_beam=False, beam_w=128):
    """Show individual solve attempts with full move sequences."""
    rng = np.random.default_rng(77)
    method = f"Beam(w={beam_w})" if use_beam else "Greedy"
    
    print(f"  DEMO: {method} — {n} cubes, scramble depth {scramble_depth}")
    
    solved_moves = []
    
    for i in range(n):
        state, sc_moves = Cube2x2.scramble(scramble_depth, rng)
        sc_str = " ".join(Cube2x2.ACTION_NAMES[a] for a in sc_moves)
        
        result = beam_solve(net, state, beam_w) if use_beam else greedy_solve(net, state)
        
        if result is not None:
            sol_str = " ".join(Cube2x2.ACTION_NAMES[a] for a in result)
            nm = len(result)
            solved_moves.append(nm)
            print(f"  #{i+1:2d} | Scramble: {sc_str}")
            print(f"       | Solution ({nm:2d} moves): {sol_str}  ✓")
        else:
            print(f"  #{i+1:2d} | Scramble: {sc_str}")
            print(f"       | ✗ FAILED")
    
    if solved_moves:
        print(f"\n  Results: {len(solved_moves)}/{n} solved | "
              f"Avg={np.mean(solved_moves):.1f} | "
              f"Min={min(solved_moves)} | Max={max(solved_moves)} moves")

def main():
    print()
    
    # ── TRAIN ──
    net = train(
        num_iterations=250,
        batch_size=400,
        max_depth=8,
        lr=5e-4,
        hidden=200,
    )
    
    # ── TEST ──
    test_agent(net, num_tests=60, max_depth=11, max_steps=25)
    
    # ── DEMO ──
    demo_solves(net, n=12, scramble_depth=4, use_beam=False)
    demo_solves(net, n=12, scramble_depth=7, use_beam=True, beam_w=128)


if __name__ == "__main__":
    main()
