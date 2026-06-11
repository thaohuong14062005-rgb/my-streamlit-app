# bai11_q_learning_policy.py
# Bài 11 - Học tăng cường Q-learning cho chính sách kinh tế thích nghi
# Module Streamlit: cần có hàm render()

import numpy as np
import pandas as pd
import streamlit as st

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except Exception:
    GYM_AVAILABLE = False

    class _FakeDiscrete:
        def __init__(self, n):
            self.n = int(n)

        def sample(self):
            return int(np.random.randint(self.n))

    class _FakeMultiDiscrete:
        def __init__(self, nvec):
            self.nvec = np.array(nvec, dtype=int)

        def sample(self):
            return np.array([np.random.randint(n) for n in self.nvec], dtype=int)

    class spaces:
        Discrete = _FakeDiscrete
        MultiDiscrete = _FakeMultiDiscrete

    class _FakeEnv:
        pass

    class gym:
        Env = _FakeEnv


# ============================================================
# 1. Cấu hình chung của bài 11
# ============================================================

STATE_NAMES = [
    "GDP growth",
    "Digital index",
    "AI capacity",
    "Unemployment risk",
]

LEVEL_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High",
}

ACTION_NAMES = {
    0: "a0 - Truyền thống",
    1: "a1 - Cân bằng",
    2: "a2 - Số hóa nhanh",
    3: "a3 - AI dẫn dắt",
    4: "a4 - Bao trùm",
}

ACTION_SHORT = {
    0: "Truyền thống",
    1: "Cân bằng",
    2: "Số hóa nhanh",
    3: "AI dẫn dắt",
    4: "Bao trùm",
}

ACTION_EXPLANATION = {
    0: "70% K, 10% D, 10% AI, 10% H",
    1: "40% K, 25% D, 15% AI, 20% H",
    2: "25% K, 45% D, 15% AI, 15% H",
    3: "20% K, 20% D, 45% AI, 15% H",
    4: "30% K, 20% D, 10% AI, 40% H",
}

ACTIONS = {
    0: np.array([0.70, 0.10, 0.10, 0.10], dtype=float),
    1: np.array([0.40, 0.25, 0.15, 0.20], dtype=float),
    2: np.array([0.25, 0.45, 0.15, 0.15], dtype=float),
    3: np.array([0.20, 0.20, 0.45, 0.15], dtype=float),
    4: np.array([0.30, 0.20, 0.10, 0.40], dtype=float),
}

ACTION_COMPONENTS = ["K", "D", "AI", "H"]
N_ACTIONS = 5
STATE_SHAPE = (3, 3, 3, 3)

INITIAL_STATES = {
    "Việt Nam 2026 thực tế": np.array([1, 1, 0, 1], dtype=int),
    "Khủng hoảng: GDP thấp, D thấp, AI thấp, thất nghiệp cao": np.array([0, 0, 0, 2], dtype=int),
    "Tăng trưởng cao: GDP cao, D cao, AI cao, thất nghiệp thấp": np.array([2, 2, 2, 0], dtype=int),
    "AI nhanh nhưng rủi ro lao động cao": np.array([1, 1, 2, 2], dtype=int),
    "Số hóa yếu nhưng thất nghiệp trung bình": np.array([1, 0, 0, 1], dtype=int),
}

WELFARE_WEIGHTS = {
    "GDP growth": 0.40,
    "Unemployment": 0.25,
    "Cyber risk": 0.20,
    "Emission": 0.15,
}


# ============================================================
# 2. Hàm tiện ích
# ============================================================

def state_to_label(state):
    return (
        f"GDP={LEVEL_LABELS[int(state[0])]}, "
        f"D={LEVEL_LABELS[int(state[1])]}, "
        f"AI={LEVEL_LABELS[int(state[2])]}, "
        f"U={LEVEL_LABELS[int(state[3])]}"
    )


def state_to_tuple(state):
    return tuple(int(x) for x in state)


def discretize_state(gdp_growth_pct, digital_index, ai_capacity, unemployment_risk):
    """
    Chuyển trạng thái liên tục sang trạng thái rời rạc 3 mức.

    Quy ước:
    - GDP growth:
        low    < 4%
        medium < 7%
        high   >= 7%
    - Digital index:
        low    < 20
        medium < 50
        high   >= 50
    - AI capacity:
        low    < 100 nghìn DN / năng lực tương đương
        medium < 160
        high   >= 160
    - Unemployment risk:
        low    < 8%
        medium < 16%
        high   >= 16%
    """
    g = 0 if gdp_growth_pct < 4 else (1 if gdp_growth_pct < 7 else 2)
    d = 0 if digital_index < 20 else (1 if digital_index < 50 else 2)
    ai = 0 if ai_capacity < 100 else (1 if ai_capacity < 160 else 2)
    u = 0 if unemployment_risk < 0.08 else (1 if unemployment_risk < 0.16 else 2)
    return np.array([g, d, ai, u], dtype=int)


def state_code_to_continuous(state_code):
    """
    Gán giá trị đại diện liên tục cho từng trạng thái rời rạc.
    Dùng khi reset môi trường từ một trạng thái giả định.
    """
    state_code = np.array(state_code, dtype=int)

    gdp_growth_rep = np.array([3.0, 6.0, 8.5])
    digital_rep = np.array([15.0, 30.0, 65.0])
    ai_rep = np.array([80.0, 125.0, 190.0])
    unemployment_rep = np.array([0.05, 0.12, 0.22])

    g_level, d_level, ai_level, u_level = state_code

    K = np.array([24500.0, 27500.0, 31500.0])[g_level]
    D = digital_rep[d_level]
    AI = ai_rep[ai_level]
    H = np.array([25.0, 32.0, 42.0])[d_level] + 2.0 * ai_level - 2.0 * u_level
    H = float(np.clip(H, 20.0, 55.0))
    U = float(unemployment_rep[u_level])
    last_growth = float(gdp_growth_rep[g_level])

    return {
        "K": float(K),
        "D": float(D),
        "AI": float(AI),
        "H": float(H),
        "U": float(U),
        "last_growth_pct": float(last_growth),
    }


def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    if len(values) < window:
        return pd.Series(values).rolling(max(1, len(values)), min_periods=1).mean().to_numpy()
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def make_action_table():
    rows = []
    for action_id, alloc in ACTIONS.items():
        row = {
            "Mã hành động": action_id,
            "Tên hành động": ACTION_NAMES[action_id],
            "Diễn giải": ACTION_EXPLANATION[action_id],
            "K": alloc[0],
            "D": alloc[1],
            "AI": alloc[2],
            "H": alloc[3],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def make_state_space_table():
    rows = []
    for g in range(3):
        for d in range(3):
            for ai in range(3):
                for u in range(3):
                    state = np.array([g, d, ai, u])
                    rows.append({
                        "GDP growth": LEVEL_LABELS[g],
                        "Digital index": LEVEL_LABELS[d],
                        "AI capacity": LEVEL_LABELS[ai],
                        "Unemployment risk": LEVEL_LABELS[u],
                        "State tuple": str(tuple(state)),
                    })
    return pd.DataFrame(rows)


# ============================================================
# 3. Môi trường MDP cho nền kinh tế Việt Nam
# ============================================================

class VietnamEconomyEnv(gym.Env):
    """
    Môi trường mô phỏng kinh tế Việt Nam dạng MDP đơn giản.

    State:
        s = [GDP growth level, Digital level, AI level, Unemployment risk level]
        Mỗi thành phần có 3 mức: 0=low, 1=medium, 2=high.
        Tổng số trạng thái: 3^4 = 81.

    Action:
        5 chính sách phân bổ ngân sách:
        a0 truyền thống, a1 cân bằng, a2 số hóa nhanh, a3 AI dẫn dắt, a4 bao trùm.

    Reward:
        R = 0.40*GDP_growth_norm
            - 0.25*delta_unemployment_norm
            - 0.20*cyber_risk
            - 0.15*emission
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        annual_budget=1000.0,
        horizon=10,
        shock_std=0.01,
        seed=42,
    ):
        if GYM_AVAILABLE:
            super().__init__()

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])

        self.annual_budget = float(annual_budget)
        self.T = int(horizon)
        self.shock_std = float(shock_std)
        self.rng = np.random.default_rng(seed)

        self.alpha = 0.33
        self.beta = 0.42
        self.gamma_d = 0.10
        self.delta_ai = 0.08
        self.theta_h = 0.07

        self.L = 53.9
        self.A = 1.0

        self.t = 0
        self.K = 27500.0
        self.D = 20.3
        self.AI = 86.0
        self.H = 30.0
        self.unemployment_risk = 0.12
        self.last_Y = None
        self.last_growth_pct = 6.0
        self.state = None

    def production(self):
        K = max(self.K, 1e-6)
        L = max(self.L, 1e-6)
        D = max(self.D, 1e-6)
        AI = max(self.AI, 1e-6)
        H = max(self.H, 1e-6)

        return (
            self.A
            * (K ** self.alpha)
            * (L ** self.beta)
            * (D ** self.gamma_d)
            * (AI ** self.delta_ai)
            * (H ** self.theta_h)
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0

        if options is None:
            options = {}

        state_code = options.get("state_code", None)

        if state_code is None:
            # Trạng thái thực tế Việt Nam 2026 theo đề gợi ý:
            # GDP medium, D medium, AI low, U medium.
            self.K = 27500.0
            self.D = 20.3
            self.AI = 86.0
            self.H = 30.0
            self.unemployment_risk = 0.12
            self.last_growth_pct = 6.0
        else:
            cont = state_code_to_continuous(state_code)
            self.K = cont["K"]
            self.D = cont["D"]
            self.AI = cont["AI"]
            self.H = cont["H"]
            self.unemployment_risk = cont["U"]
            self.last_growth_pct = cont["last_growth_pct"]

        self.A = 1.0
        self.L = 53.9
        self.last_Y = self.production()
        self.state = discretize_state(
            self.last_growth_pct,
            self.D,
            self.AI,
            self.unemployment_risk,
        )

        return self.state.copy(), {}

    def step(self, action):
        action = int(action)
        alloc = ACTIONS[action]
        old_Y = self.last_Y
        old_U = self.unemployment_risk

        budget = self.annual_budget
        invest_K = alloc[0] * budget
        invest_D = alloc[1] * budget
        invest_AI = alloc[2] * budget
        invest_H = alloc[3] * budget

        # Động học vốn và năng lực
        # K: khấu hao thấp, tích lũy trực tiếp
        self.K = (1.0 - 0.045) * self.K + invest_K

        # D: hạ tầng số có khấu hao công nghệ, đầu tư D tăng digital index
        self.D = (1.0 - 0.035) * self.D + 0.0065 * invest_D + 0.008 * self.H
        self.D = float(np.clip(self.D, 5.0, 100.0))

        # AI: năng lực AI tăng nhờ đầu tư AI, nhưng cần D làm nền
        self.AI = (1.0 - 0.055) * self.AI + 0.022 * invest_AI + 0.035 * self.D
        self.AI = float(np.clip(self.AI, 30.0, 300.0))

        # H: vốn nhân lực tăng nhờ đầu tư H, có suy giảm do brain drain
        self.H = (1.0 - 0.018) * self.H + 0.0042 * invest_H
        self.H = float(np.clip(self.H, 15.0, 80.0))

        # TFP nội sinh: D, AI, H làm tăng năng suất nhưng có hiệu ứng nhỏ
        tfp_growth = (
            0.0015 * (self.D / 100.0)
            + 0.0010 * (self.AI / 200.0)
            + 0.0018 * (self.H / 60.0)
        )
        self.A = self.A * (1.0 + tfp_growth)

        # Shock ngẫu nhiên nhỏ
        macro_shock = self.rng.normal(0.0, self.shock_std)

        new_Y_raw = self.production()
        new_Y = max(new_Y_raw * (1.0 + macro_shock), 1e-6)

        gdp_growth_pct = (new_Y - old_Y) / max(old_Y, 1e-6) * 100.0

        # Rủi ro thất nghiệp:
        # AI tăng nhanh có thể làm tăng rủi ro dịch chuyển lao động.
        # H và D giúp giảm rủi ro do đào tạo lại và hấp thụ công nghệ.
        delta_U = (
            0.020 * alloc[2]
            + 0.006 * alloc[0]
            - 0.022 * alloc[3]
            - 0.010 * alloc[1]
            - 0.004 * np.tanh((gdp_growth_pct - 6.0) / 4.0)
            + self.rng.normal(0.0, self.shock_std * 0.20)
        )
        self.unemployment_risk = float(np.clip(self.unemployment_risk + delta_U, 0.02, 0.35))

        # Cyber risk: AI cao nhưng H/D chưa đủ sẽ làm tăng rủi ro
        cyber_risk = (
            0.20
            + 0.58 * alloc[2]
            + 0.12 * (self.AI / 250.0)
            - 0.22 * (self.H / 70.0)
            - 0.10 * (self.D / 100.0)
        )
        cyber_risk = float(np.clip(cyber_risk, 0.0, 1.0))

        # Emission: đầu tư K và AI/hạ tầng dữ liệu tăng phát thải gián tiếp
        emission = (
            0.18
            + 0.42 * alloc[0]
            + 0.24 * alloc[2]
            + 0.12 * alloc[1]
            - 0.10 * alloc[3]
        )
        emission = float(np.clip(emission, 0.0, 1.0))

        # Chuẩn hóa reward
        gdp_growth_norm = float(np.clip(gdp_growth_pct / 10.0, -1.0, 1.5))
        delta_unemployment_norm = float(np.clip((self.unemployment_risk - old_U) / 0.05, -1.0, 1.0))

        reward = (
            WELFARE_WEIGHTS["GDP growth"] * gdp_growth_norm
            - WELFARE_WEIGHTS["Unemployment"] * delta_unemployment_norm
            - WELFARE_WEIGHTS["Cyber risk"] * cyber_risk
            - WELFARE_WEIGHTS["Emission"] * emission
        )

        self.t += 1
        self.last_Y = new_Y
        self.last_growth_pct = gdp_growth_pct

        self.state = discretize_state(
            gdp_growth_pct,
            self.D,
            self.AI,
            self.unemployment_risk,
        )

        terminated = self.t >= self.T
        truncated = False

        info = {
            "t": self.t,
            "K": self.K,
            "D": self.D,
            "AI": self.AI,
            "H": self.H,
            "Y": new_Y,
            "GDP growth (%)": gdp_growth_pct,
            "Unemployment risk": self.unemployment_risk,
            "Delta unemployment": self.unemployment_risk - old_U,
            "Cyber risk": cyber_risk,
            "Emission": emission,
            "Reward": reward,
            "Action": action,
            "Action name": ACTION_SHORT[action],
        }

        return self.state.copy(), float(reward), terminated, truncated, info


# ============================================================
# 4. Q-learning
# ============================================================

def epsilon_by_episode(ep, eps_start, eps_min, decay_episodes):
    if decay_episodes <= 0:
        return eps_min
    frac = min(1.0, ep / decay_episodes)
    return float(max(eps_min, eps_start - (eps_start - eps_min) * frac))


def choose_action_from_policy(policy_name, state, Q=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    if policy_name == "q_learning":
        if Q is None:
            return 1
        return int(np.argmax(Q[state_to_tuple(state)]))

    if policy_name == "always_a1":
        return 1

    if policy_name == "always_a3":
        return 3

    if policy_name == "always_a4":
        return 4

    if policy_name == "random":
        return int(rng.integers(0, N_ACTIONS))

    return 1


def train_q_learning(
    episodes=10000,
    alpha=0.10,
    gamma=0.95,
    eps_start=1.0,
    eps_min=0.05,
    eps_decay_episodes=5000,
    annual_budget=1000.0,
    shock_std=0.01,
    random_start=True,
    seed=42,
):
    """
    Huấn luyện tabular Q-learning.

    Q có shape:
        (3, 3, 3, 3, 5)

    Công thức:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma*max_a' Q(s',a') - Q(s,a)]
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros(STATE_SHAPE + (N_ACTIONS,), dtype=float)

    episode_rewards = []
    episode_eps = []
    action_counts = np.zeros(N_ACTIONS, dtype=int)

    env = VietnamEconomyEnv(
        annual_budget=annual_budget,
        horizon=10,
        shock_std=shock_std,
        seed=seed,
    )

    for ep in range(int(episodes)):
        if random_start:
            init_state = rng.integers(0, 3, size=4)
            state, _ = env.reset(
                seed=int(seed + ep),
                options={"state_code": init_state},
            )
        else:
            state, _ = env.reset(seed=int(seed + ep))

        eps = epsilon_by_episode(ep, eps_start, eps_min, eps_decay_episodes)
        total_reward = 0.0

        while True:
            s_tuple = state_to_tuple(state)

            if rng.random() < eps:
                action = int(rng.integers(0, N_ACTIONS))
            else:
                action = int(np.argmax(Q[s_tuple]))

            next_state, reward, done, _, _ = env.step(action)
            ns_tuple = state_to_tuple(next_state)

            td_target = reward
            if not done:
                td_target += gamma * np.max(Q[ns_tuple])

            td_error = td_target - Q[s_tuple + (action,)]
            Q[s_tuple + (action,)] += alpha * td_error

            total_reward += reward
            action_counts[action] += 1
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        episode_eps.append(eps)

    history = pd.DataFrame({
        "Episode": np.arange(1, int(episodes) + 1),
        "Reward": episode_rewards,
        "Rolling mean reward": moving_average(episode_rewards, window=max(10, int(episodes / 100))),
        "Epsilon": episode_eps,
    })

    action_count_df = pd.DataFrame({
        "Hành động": [ACTION_NAMES[i] for i in range(N_ACTIONS)],
        "Số lần được chọn khi train": action_counts,
        "Tỷ trọng (%)": action_counts / max(action_counts.sum(), 1) * 100,
    })

    return Q, history, action_count_df


@st.cache_data(show_spinner=False)
def cached_train_q_learning(
    episodes,
    alpha,
    gamma,
    eps_start,
    eps_min,
    eps_decay_episodes,
    annual_budget,
    shock_std,
    random_start,
    seed,
):
    return train_q_learning(
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        eps_start=eps_start,
        eps_min=eps_min,
        eps_decay_episodes=eps_decay_episodes,
        annual_budget=annual_budget,
        shock_std=shock_std,
        random_start=random_start,
        seed=seed,
    )


# ============================================================
# 5. Đánh giá chính sách
# ============================================================

def simulate_policy(
    policy_name,
    Q=None,
    start_state=None,
    annual_budget=1000.0,
    shock_std=0.01,
    seed=42,
):
    rng = np.random.default_rng(seed)
    env = VietnamEconomyEnv(
        annual_budget=annual_budget,
        horizon=10,
        shock_std=shock_std,
        seed=seed,
    )

    if start_state is None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset(seed=seed, options={"state_code": np.array(start_state, dtype=int)})

    rows = []

    for year in range(2026, 2036):
        action = choose_action_from_policy(policy_name, state, Q=Q, rng=rng)
        next_state, reward, done, _, info = env.step(action)

        rows.append({
            "Năm": year,
            "Chính sách": policy_name,
            "State trước hành động": state_to_label(state),
            "State sau hành động": state_to_label(next_state),
            "Action id": action,
            "Hành động": ACTION_SHORT[action],
            "Reward": reward,
            "K": info["K"],
            "D": info["D"],
            "AI": info["AI"],
            "H": info["H"],
            "Y": info["Y"],
            "GDP growth (%)": info["GDP growth (%)"],
            "Unemployment risk (%)": info["Unemployment risk"] * 100.0,
            "Cyber risk": info["Cyber risk"],
            "Emission": info["Emission"],
        })

        state = next_state

        if done:
            break

    return pd.DataFrame(rows)


def evaluate_policy_many_episodes(
    policy_name,
    Q=None,
    n_eval=200,
    annual_budget=1000.0,
    shock_std=0.01,
    random_start=True,
    seed=123,
):
    rng = np.random.default_rng(seed)

    rows = []

    for i in range(int(n_eval)):
        if random_start:
            start_state = rng.integers(0, 3, size=4)
        else:
            start_state = INITIAL_STATES["Việt Nam 2026 thực tế"]

        traj = simulate_policy(
            policy_name=policy_name,
            Q=Q,
            start_state=start_state,
            annual_budget=annual_budget,
            shock_std=shock_std,
            seed=int(seed + i),
        )

        total_reward = float(traj["Reward"].sum())
        final_row = traj.iloc[-1]

        rows.append({
            "Episode": i + 1,
            "Chính sách": policy_name,
            "Total reward": total_reward,
            "Final GDP growth (%)": final_row["GDP growth (%)"],
            "Final unemployment risk (%)": final_row["Unemployment risk (%)"],
            "Final D": final_row["D"],
            "Final AI": final_row["AI"],
            "Final H": final_row["H"],
            "Final cyber risk": final_row["Cyber risk"],
            "Final emission": final_row["Emission"],
        })

    return pd.DataFrame(rows)


def compare_policies(
    Q,
    n_eval=200,
    annual_budget=1000.0,
    shock_std=0.01,
    random_start=True,
    seed=123,
):
    policy_map = {
        "Q-learning π*": "q_learning",
        "Rule a1 - Cân bằng": "always_a1",
        "Rule a3 - AI dẫn dắt": "always_a3",
        "Rule random": "random",
    }

    all_eval = []
    summary_rows = []

    for display_name, policy_name in policy_map.items():
        df_eval = evaluate_policy_many_episodes(
            policy_name=policy_name,
            Q=Q,
            n_eval=n_eval,
            annual_budget=annual_budget,
            shock_std=shock_std,
            random_start=random_start,
            seed=seed + len(all_eval) * 1000,
        )
        df_eval["Tên chính sách"] = display_name
        all_eval.append(df_eval)

        summary_rows.append({
            "Chính sách": display_name,
            "Reward TB": df_eval["Total reward"].mean(),
            "Reward std": df_eval["Total reward"].std(),
            "Final GDP growth (%)": df_eval["Final GDP growth (%)"].mean(),
            "Final unemployment risk (%)": df_eval["Final unemployment risk (%)"].mean(),
            "Final D": df_eval["Final D"].mean(),
            "Final AI": df_eval["Final AI"].mean(),
            "Final H": df_eval["Final H"].mean(),
            "Final cyber risk": df_eval["Final cyber risk"].mean(),
            "Final emission": df_eval["Final emission"].mean(),
        })

    all_eval_df = pd.concat(all_eval, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("Reward TB", ascending=False)

    return summary_df, all_eval_df


def extract_policy_for_initial_states(Q):
    rows = []
    for name, state in INITIAL_STATES.items():
        q_values = Q[state_to_tuple(state)]
        best_action = int(np.argmax(q_values))
        rows.append({
            "Trạng thái kiểm tra": name,
            "State tuple": str(tuple(state)),
            "Diễn giải state": state_to_label(state),
            "Action π*(s)": ACTION_NAMES[best_action],
            "Phân bổ": ACTION_EXPLANATION[best_action],
            "Q-value tốt nhất": float(np.max(q_values)),
        })
    return pd.DataFrame(rows)


def make_q_values_table(Q, state):
    q_values = Q[state_to_tuple(state)]
    return pd.DataFrame({
        "Action id": list(range(N_ACTIONS)),
        "Hành động": [ACTION_NAMES[i] for i in range(N_ACTIONS)],
        "Phân bổ": [ACTION_EXPLANATION[i] for i in range(N_ACTIONS)],
        "Q-value": q_values,
    }).sort_values("Q-value", ascending=False)


def make_policy_grid(Q, fixed_ai_level=1, fixed_u_level=1):
    """
    Tạo lưới chính sách theo GDP growth x Digital index,
    giữ cố định AI capacity và Unemployment risk.
    """
    rows = []
    for g in range(3):
        row = {}
        for d in range(3):
            state = np.array([g, d, fixed_ai_level, fixed_u_level], dtype=int)
            action = int(np.argmax(Q[state_to_tuple(state)]))
            row[LEVEL_LABELS[d]] = ACTION_SHORT[action]
        rows.append({
            "GDP growth": LEVEL_LABELS[g],
            **row,
        })

    return pd.DataFrame(rows)


def make_policy_action_distribution(Q):
    best_actions = []
    for g in range(3):
        for d in range(3):
            for ai in range(3):
                for u in range(3):
                    state = (g, d, ai, u)
                    best_actions.append(int(np.argmax(Q[state])))

    counts = pd.Series(best_actions).value_counts().reindex(range(N_ACTIONS), fill_value=0)
    return pd.DataFrame({
        "Hành động": [ACTION_NAMES[i] for i in range(N_ACTIONS)],
        "Số trạng thái chọn hành động này": [counts[i] for i in range(N_ACTIONS)],
        "Tỷ trọng trong 81 trạng thái (%)": [counts[i] / 81 * 100 for i in range(N_ACTIONS)],
    })


# ============================================================
# 6. Streamlit UI
# ============================================================

def render():
    st.title("Bài 11. Học tăng cường Q-learning cho chính sách kinh tế thích nghi")
    st.caption("MDP 81 trạng thái | 5 hành động chính sách | Tabular Q-learning | So sánh rule-based")

    st.markdown(
        """
        Bài này mô phỏng nền kinh tế Việt Nam như một **Markov Decision Process (MDP)**.
        Mỗi năm, agent chọn một chính sách phân bổ ngân sách giữa **K, D, AI, H**.
        Sau 10 năm, Q-learning học được chính sách thích nghi `π*(s) = argmax_a Q(s,a)`.
        """
    )

    st.sidebar.header("Thiết lập huấn luyện Bài 11")

    episodes = st.sidebar.number_input(
        "Số episode huấn luyện",
        min_value=500,
        max_value=50000,
        value=10000,
        step=500,
    )

    alpha = st.sidebar.slider(
        "Learning rate α",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
    )

    gamma = st.sidebar.slider(
        "Discount factor γ",
        min_value=0.50,
        max_value=0.99,
        value=0.95,
        step=0.01,
    )

    eps_start = st.sidebar.slider(
        "Epsilon ban đầu",
        min_value=0.10,
        max_value=1.00,
        value=1.00,
        step=0.05,
    )

    eps_min = st.sidebar.slider(
        "Epsilon tối thiểu",
        min_value=0.00,
        max_value=0.30,
        value=0.05,
        step=0.01,
    )

    eps_decay_episodes = st.sidebar.number_input(
        "Số episode để epsilon giảm dần",
        min_value=100,
        max_value=50000,
        value=5000,
        step=500,
    )

    annual_budget = st.sidebar.number_input(
        "Ngân sách mỗi năm, nghìn tỷ VND",
        min_value=100.0,
        max_value=5000.0,
        value=1000.0,
        step=100.0,
    )

    shock_std = st.sidebar.slider(
        "Độ nhiễu cú sốc kinh tế",
        min_value=0.00,
        max_value=0.10,
        value=0.01,
        step=0.005,
    )

    random_start = st.sidebar.checkbox(
        "Huấn luyện từ nhiều trạng thái khởi đầu ngẫu nhiên",
        value=True,
    )

    seed = st.sidebar.number_input(
        "Random seed",
        min_value=1,
        max_value=999999,
        value=42,
        step=1,
    )

    with st.spinner("Đang huấn luyện Q-learning..."):
        Q, history, action_count_df = cached_train_q_learning(
            int(episodes),
            float(alpha),
            float(gamma),
            float(eps_start),
            float(eps_min),
            int(eps_decay_episodes),
            float(annual_budget),
            float(shock_std),
            bool(random_start),
            int(seed),
        )

    policy_state_df = extract_policy_for_initial_states(Q)
    action_dist_df = make_policy_action_distribution(Q)

    vn_state = INITIAL_STATES["Việt Nam 2026 thực tế"]
    vn_action = int(np.argmax(Q[state_to_tuple(vn_state)]))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Số trạng thái", "81")
    col2.metric("Số hành động", "5")
    col3.metric("Reward TB 100 ep cuối", f"{history['Reward'].tail(100).mean():.3f}")
    col4.metric("π*(VN 2026)", ACTION_SHORT[vn_action])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. MDP",
        "2. Huấn luyện",
        "3. Chính sách π*",
        "4. So sánh chính sách",
        "5. Diễn giải & DQN",
    ])

    # ========================================================
    # Tab 1 - MDP
    # ========================================================
    with tab1:
        st.subheader("1.1. Không gian trạng thái")

        st.markdown(
            """
            State gồm 4 thành phần rời rạc, mỗi thành phần có 3 mức:

            - GDP growth: low / medium / high
            - Digital index: low / medium / high
            - AI capacity: low / medium / high
            - Unemployment risk: low / medium / high

            Tổng số trạng thái: `3^4 = 81`.
            """
        )

        state_space_df = make_state_space_table()
        st.dataframe(state_space_df, use_container_width=True)

        st.subheader("1.2. Không gian hành động")

        action_table = make_action_table()
        st.dataframe(
            action_table.style.format({
                "K": "{:.2f}",
                "D": "{:.2f}",
                "AI": "{:.2f}",
                "H": "{:.2f}",
            }),
            use_container_width=True,
        )

        st.subheader("1.3. Hàm reward")

        reward_df = pd.DataFrame({
            "Thành phần welfare": list(WELFARE_WEIGHTS.keys()),
            "Trọng số": list(WELFARE_WEIGHTS.values()),
            "Vai trò trong reward": [
                "Tăng trưởng GDP cao làm reward tăng",
                "Tăng thất nghiệp làm reward giảm",
                "Rủi ro an ninh mạng làm reward giảm",
                "Phát thải làm reward giảm",
            ],
        })

        st.dataframe(reward_df, use_container_width=True)

        st.code(
            """
Reward = 0.40 * GDP_growth_norm
         - 0.25 * delta_unemployment_norm
         - 0.20 * cyber_risk
         - 0.15 * emission
            """,
            language="text",
        )

        if not GYM_AVAILABLE:
            st.warning(
                "Môi trường hiện tại chưa có gymnasium. Code vẫn chạy nhờ lớp fallback nội bộ. "
                "Nếu muốn đúng chuẩn Gymnasium, thêm `gymnasium` vào requirements.txt."
            )

    # ========================================================
    # Tab 2 - Training
    # ========================================================
    with tab2:
        st.subheader("2.1. Learning curve")

        chart_history = history.set_index("Episode")[["Reward", "Rolling mean reward"]]
        st.line_chart(chart_history)

        st.subheader("2.2. Epsilon decay")

        eps_chart = history.set_index("Episode")[["Epsilon"]]
        st.line_chart(eps_chart)

        st.subheader("2.3. Tần suất chọn hành động trong quá trình train")

        st.dataframe(
            action_count_df.style.format({
                "Số lần được chọn khi train": "{:,.0f}",
                "Tỷ trọng (%)": "{:.2f}",
            }),
            use_container_width=True,
        )

        st.bar_chart(action_count_df.set_index("Hành động")[["Số lần được chọn khi train"]])

        st.subheader("2.4. Nhận xét kỹ thuật")

        st.markdown(
            """
            - Nếu learning curve tăng dần rồi ổn định, Q-learning đã học được chính sách tương đối tốt.
            - Nếu reward dao động mạnh, tăng số episode hoặc giảm `shock_std`.
            - Nếu agent quá thiên về một hành động, thử bật huấn luyện từ nhiều trạng thái khởi đầu hoặc tăng số episode.
            - Nếu muốn chính sách bớt cực đoan, có thể thêm penalty cho hành động thay đổi quá mạnh giữa các năm.
            """
        )

    # ========================================================
    # Tab 3 - Policy
    # ========================================================
    with tab3:
        st.subheader("3.1. Chính sách π*(s) ở 5 trạng thái kiểm tra")

        st.dataframe(
            policy_state_df.style.format({"Q-value tốt nhất": "{:.4f}"}),
            use_container_width=True,
        )

        st.subheader("3.2. Q-value tại một trạng thái cụ thể")

        selected_state_name = st.selectbox(
            "Chọn trạng thái để xem Q-value",
            list(INITIAL_STATES.keys()),
        )
        selected_state = INITIAL_STATES[selected_state_name]

        q_table = make_q_values_table(Q, selected_state)
        st.dataframe(q_table.style.format({"Q-value": "{:.4f}"}), use_container_width=True)

        st.subheader("3.3. Ma trận chính sách theo GDP growth và Digital index")

        col_a, col_b = st.columns(2)
        fixed_ai_level = col_a.selectbox(
            "Giữ cố định AI capacity",
            options=[0, 1, 2],
            format_func=lambda x: LEVEL_LABELS[x],
            index=1,
        )
        fixed_u_level = col_b.selectbox(
            "Giữ cố định Unemployment risk",
            options=[0, 1, 2],
            format_func=lambda x: LEVEL_LABELS[x],
            index=1,
        )

        policy_grid = make_policy_grid(
            Q,
            fixed_ai_level=int(fixed_ai_level),
            fixed_u_level=int(fixed_u_level),
        )
        st.dataframe(policy_grid, use_container_width=True)

        st.subheader("3.4. Phân bố hành động tối ưu trên toàn bộ 81 trạng thái")

        st.dataframe(
            action_dist_df.style.format({
                "Số trạng thái chọn hành động này": "{:,.0f}",
                "Tỷ trọng trong 81 trạng thái (%)": "{:.2f}",
            }),
            use_container_width=True,
        )

        st.bar_chart(action_dist_df.set_index("Hành động")[["Số trạng thái chọn hành động này"]])

    # ========================================================
    # Tab 4 - Compare
    # ========================================================
    with tab4:
        st.subheader("4.1. So sánh π* với các chính sách rule-based")

        n_eval = st.number_input(
            "Số episode dùng để đánh giá chính sách",
            min_value=20,
            max_value=2000,
            value=300,
            step=20,
        )

        with st.spinner("Đang đánh giá các chính sách..."):
            summary_df, all_eval_df = compare_policies(
                Q=Q,
                n_eval=int(n_eval),
                annual_budget=float(annual_budget),
                shock_std=float(shock_std),
                random_start=True,
                seed=int(seed) + 999,
            )

        st.dataframe(
            summary_df.style.format({
                "Reward TB": "{:.4f}",
                "Reward std": "{:.4f}",
                "Final GDP growth (%)": "{:.2f}",
                "Final unemployment risk (%)": "{:.2f}",
                "Final D": "{:.2f}",
                "Final AI": "{:.2f}",
                "Final H": "{:.2f}",
                "Final cyber risk": "{:.3f}",
                "Final emission": "{:.3f}",
            }),
            use_container_width=True,
        )

        st.markdown("#### Reward trung bình theo chính sách")
        st.bar_chart(summary_df.set_index("Chính sách")[["Reward TB"]])

        st.markdown("#### Rủi ro thất nghiệp cuối kỳ theo chính sách")
        st.bar_chart(summary_df.set_index("Chính sách")[["Final unemployment risk (%)"]])

        st.subheader("4.2. Mô phỏng quỹ đạo 2026–2035")

        selected_start_name = st.selectbox(
            "Chọn trạng thái khởi đầu để mô phỏng quỹ đạo",
            list(INITIAL_STATES.keys()),
            key="trajectory_start",
        )

        selected_policies = st.multiselect(
            "Chọn chính sách để vẽ quỹ đạo",
            options=[
                "Q-learning π*",
                "Rule a1 - Cân bằng",
                "Rule a3 - AI dẫn dắt",
                "Rule random",
            ],
            default=["Q-learning π*", "Rule a1 - Cân bằng", "Rule a3 - AI dẫn dắt"],
        )

        policy_name_map = {
            "Q-learning π*": "q_learning",
            "Rule a1 - Cân bằng": "always_a1",
            "Rule a3 - AI dẫn dắt": "always_a3",
            "Rule random": "random",
        }

        traj_rows = []
        for display_name in selected_policies:
            policy_internal = policy_name_map[display_name]
            traj = simulate_policy(
                policy_name=policy_internal,
                Q=Q,
                start_state=INITIAL_STATES[selected_start_name],
                annual_budget=float(annual_budget),
                shock_std=float(shock_std),
                seed=int(seed) + 2026 + len(traj_rows),
            )
            traj["Tên chính sách"] = display_name
            traj_rows.append(traj)

        if traj_rows:
            traj_all = pd.concat(traj_rows, ignore_index=True)

            st.markdown("#### Bảng quỹ đạo")
            st.dataframe(
                traj_all.style.format({
                    "Reward": "{:.4f}",
                    "K": "{:,.2f}",
                    "D": "{:.2f}",
                    "AI": "{:.2f}",
                    "H": "{:.2f}",
                    "Y": "{:,.2f}",
                    "GDP growth (%)": "{:.2f}",
                    "Unemployment risk (%)": "{:.2f}",
                    "Cyber risk": "{:.3f}",
                    "Emission": "{:.3f}",
                }),
                use_container_width=True,
            )

            metrics_to_plot = [
                "Reward",
                "GDP growth (%)",
                "Unemployment risk (%)",
                "D",
                "AI",
                "H",
                "Cyber risk",
                "Emission",
            ]

            selected_metric = st.selectbox(
                "Chọn chỉ tiêu để vẽ",
                metrics_to_plot,
            )

            pivot_traj = traj_all.pivot_table(
                index="Năm",
                columns="Tên chính sách",
                values=selected_metric,
                aggfunc="mean",
            )
            st.line_chart(pivot_traj)

    # ========================================================
    # Tab 5 - Discussion + DQN
    # ========================================================
    with tab5:
        st.subheader("5.1. Gợi ý trả lời câu hỏi thảo luận chính sách")

        crisis_state = np.array([0, 0, 0, 2], dtype=int)
        high_state = np.array([2, 2, 2, 0], dtype=int)

        crisis_action = int(np.argmax(Q[state_to_tuple(crisis_state)]))
        high_action = int(np.argmax(Q[state_to_tuple(high_state)]))

        st.markdown(
            f"""
            **a) Khi GDP thấp, D thấp, thất nghiệp cao, chính sách π*(s) chọn gì?**

            Với mô hình hiện tại, tại trạng thái `{state_to_label(crisis_state)}`,
            Q-learning chọn **{ACTION_NAMES[crisis_action]}**,
            tức **{ACTION_EXPLANATION[crisis_action]}**.

            Có thể diễn giải đây là phản ứng kiểu “quick win” nếu chính sách ưu tiên phục hồi nhanh,
            hoặc là phản ứng bao trùm nếu agent ưu tiên giảm rủi ro thất nghiệp.

            **b) Khi GDP cao, AI cao, thất nghiệp thấp, chính sách chọn gì?**

            Tại trạng thái `{state_to_label(high_state)}`,
            Q-learning chọn **{ACTION_NAMES[high_action]}**,
            tức **{ACTION_EXPLANATION[high_action]}**.

            Nếu hành động nghiêng về `a1` hoặc `a4`, có thể hiểu là giai đoạn củng cố:
            duy trì tăng trưởng nhưng kiểm soát rủi ro xã hội, an ninh mạng và phát thải.

            **c) AI không thay thế quyết định chính trị - xã hội**

            Chính sách `π*` nên được dùng như một công cụ hỗ trợ ra quyết định, không phải cơ chế tự động ra chính sách.
            Quy trình phù hợp là:

            1. Mô hình đề xuất chính sách theo trạng thái kinh tế.
            2. Chuyên gia kiểm tra giả định, dữ liệu và tham số reward.
            3. Cơ quan chính sách thảo luận đánh đổi: tăng trưởng, lao động, an ninh dữ liệu, môi trường.
            4. Người ra quyết định chịu trách nhiệm chính trị cuối cùng.
            """
        )

        st.subheader("5.2. Mở rộng DQN")

        st.markdown(
            """
            Phần DQN là mở rộng. Trên Streamlit Cloud, `stable-baselines3` và `torch` có thể làm app nặng,
            nên mình không chạy trực tiếp trong module này. Bạn có thể để đoạn dưới vào notebook riêng hoặc file train riêng.
            """
        )

        dqn_code = """
# Gợi ý mở rộng DQN cho Bài 11
# Chạy riêng trong notebook hoặc file train_dqn.py

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Tái sử dụng VietnamEconomyEnv trong file bai11_q_learning_policy.py
from bai11_q_learning_policy import VietnamEconomyEnv

env = VietnamEconomyEnv(
    annual_budget=1000.0,
    horizon=10,
    shock_std=0.01,
    seed=42,
)

check_env(env, warn=True)

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.95,
    exploration_fraction=0.30,
    exploration_final_eps=0.05,
    train_freq=1,
    target_update_interval=500,
    verbose=1,
)

model.learn(total_timesteps=100000)
model.save("dqn_vietnam_economy")

# Đánh giá nhanh
obs, _ = env.reset()
total_reward = 0.0
for t in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    print(t, action, reward, info)
    if done:
        break

print("Total reward:", total_reward)
        """
        st.code(dqn_code, language="python")

        st.subheader("5.3. Ghi chú kỹ thuật quan trọng")

        st.warning(
            "Q-learning trong bài này là mô phỏng sư phạm. Kết quả phụ thuộc mạnh vào cách thiết kế reward, "
            "ngưỡng rời rạc hóa state và phương trình chuyển trạng thái. Khi viết báo cáo, bạn nên giải thích rõ "
            "đây là mô hình minh họa, không phải dự báo chính sách thực tế."
        )
