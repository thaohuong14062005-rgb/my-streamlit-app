# bai11_q_learning_rl.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except Exception:
    gym = None
    spaces = None
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import DQN
    from gymnasium.wrappers import FlattenObservation
    DQN_AVAILABLE = True and GYM_AVAILABLE
except Exception:
    DQN_AVAILABLE = False


BRAND = "#053151"

MULTI_COLORS = {
    "Q-learning π*": "#053151",
    "Luôn chọn a1": "#E76F51",
    "Luôn chọn a3": "#2A9D8F",
    "Random": "#F4A261",
    "a1 - Hạ tầng nhanh": "#053151",
    "a2 - Cân bằng": "#E76F51",
    "a3 - Dữ liệu/nền tảng": "#2A9D8F",
    "a4 - AI tăng tốc": "#F4A261",
    "a5 - Nhân lực/an sinh": "#8E44AD",
}


# =========================================================
# BÀI 11 — Q-LEARNING CHO CHÍNH SÁCH CHUYỂN ĐỔI SỐ
# =========================================================


T = 10
N_ACTIONS = 5

STATE_LABELS = {
    0: {
        0: "GDP thấp",
        1: "GDP trung bình",
        2: "GDP cao",
    },
    1: {
        0: "D thấp",
        1: "D trung bình",
        2: "D cao",
    },
    2: {
        0: "AI thấp",
        1: "AI trung bình",
        2: "AI cao",
    },
    3: {
        0: "U thấp",
        1: "U trung bình",
        2: "U cao",
    },
}

ACTION_NAMES = {
    0: "a1 - Hạ tầng nhanh",
    1: "a2 - Cân bằng",
    2: "a3 - Dữ liệu/nền tảng",
    3: "a4 - AI tăng tốc",
    4: "a5 - Nhân lực/an sinh",
}

ACTION_SHORT = {
    0: "a1",
    1: "a2",
    2: "a3",
    3: "a4",
    4: "a5",
}

ALLOCATION = {
    0: np.array([0.70, 0.10, 0.10, 0.10], dtype=float),
    1: np.array([0.40, 0.25, 0.15, 0.20], dtype=float),
    2: np.array([0.25, 0.45, 0.15, 0.15], dtype=float),
    3: np.array([0.20, 0.20, 0.45, 0.15], dtype=float),
    4: np.array([0.30, 0.20, 0.10, 0.40], dtype=float),
}

ALLOCATION_LABELS = ["K", "D", "AI", "H"]

REWARD_WEIGHTS = np.array([0.40, 0.25, 0.20, 0.15], dtype=float)


class SimpleDiscrete:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return int(np.random.randint(self.n))


class SimpleMultiDiscrete:
    def __init__(self, nvec):
        self.nvec = np.array(nvec, dtype=int)


BaseEnv = gym.Env if GYM_AVAILABLE else object


class VietnamEconomyEnv(BaseEnv):
    metadata = {"render_modes": []}

    def __init__(self, start_state=None):
        if GYM_AVAILABLE:
            super().__init__()
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        else:
            self.action_space = SimpleDiscrete(N_ACTIONS)
            self.observation_space = SimpleMultiDiscrete([3, 3, 3, 3])

        self.T = T
        self.budget = 1000.0

        self.default_state = np.array([1, 1, 0, 1], dtype=int)

        if start_state is None:
            self.start_state = self.default_state.copy()
        else:
            self.start_state = np.asarray(start_state, dtype=int)

        self.state = self.start_state.copy()
        self.t = 0

        self.K = 27500.0
        self.D = 20.3
        self.AI = 86.0
        self.H = 30.0
        self.U = 4.5
        self.prev_Y = None

    def _init_macro_from_state(self, state):
        g_level, d_level, ai_level, u_level = state

        k_map = {
            0: 25500.0,
            1: 27500.0,
            2: 30500.0,
        }

        d_map = {
            0: 16.0,
            1: 20.3,
            2: 30.0,
        }

        ai_map = {
            0: 86.0,
            1: 115.0,
            2: 145.0,
        }

        h_map = {
            0: 26.0,
            1: 30.0,
            2: 37.0,
        }

        u_map = {
            0: 3.2,
            1: 4.5,
            2: 6.5,
        }

        self.K = k_map[int(g_level)]
        self.D = d_map[int(d_level)]
        self.AI = ai_map[int(ai_level)]
        self.H = h_map[int(max(0, min(2, 2 - u_level)))]
        self.U = u_map[int(u_level)]

        self.prev_Y = self._production()

    def _production(self):
        return (
            (self.K ** 0.33)
            * (54.0 ** 0.42)
            * (max(self.D, 1e-9) ** 0.10)
            * (max(self.AI, 1e-9) ** 0.08)
            * (max(self.H, 1e-9) ** 0.07)
        )

    @staticmethod
    def _discretize_growth(growth_pct):
        if growth_pct < 4.0:
            return 0
        if growth_pct < 7.0:
            return 1
        return 2

    @staticmethod
    def _discretize_D(D):
        if D < 20.0:
            return 0
        if D < 32.0:
            return 1
        return 2

    @staticmethod
    def _discretize_AI(AI):
        if AI < 95.0:
            return 0
        if AI < 135.0:
            return 1
        return 2

    @staticmethod
    def _discretize_U(U):
        if U < 4.0:
            return 0
        if U < 6.0:
            return 1
        return 2

    def _state_from_macro(self, growth_pct):
        return np.array(
            [
                self._discretize_growth(growth_pct),
                self._discretize_D(self.D),
                self._discretize_AI(self.AI),
                self._discretize_U(self.U),
            ],
            dtype=int,
        )

    def _reward(self, growth_pct, action):
        growth_score = np.clip((growth_pct + 1.0) / 9.0, 0, 1)
        d_score = np.clip((self.D - 15.0) / 25.0, 0, 1)
        ai_score = np.clip((self.AI - 80.0) / 70.0, 0, 1)
        u_score = np.clip((8.0 - self.U) / 5.5, 0, 1)

        base = 100.0 * (
            REWARD_WEIGHTS[0] * growth_score
            + REWARD_WEIGHTS[1] * d_score
            + REWARD_WEIGHTS[2] * ai_score
            + REWARD_WEIGHTS[3] * u_score
        )

        allocation = ALLOCATION[int(action)]

        imbalance_penalty = 0.0

        if allocation[2] >= 0.40 and self.H < 31.0:
            imbalance_penalty += 7.5

        if self.U > 6.0 and allocation[3] < 0.20:
            imbalance_penalty += 8.0

        if self.D < 20.0 and allocation[1] < 0.20:
            imbalance_penalty += 4.0

        return float(base - imbalance_penalty)

    def reset(self, seed=None, options=None):
        if GYM_AVAILABLE:
            super().reset(seed=seed)

        if options is not None and "start_state" in options:
            self.start_state = np.asarray(options["start_state"], dtype=int)
        else:
            self.start_state = self.default_state.copy()

        self.state = self.start_state.copy()
        self.t = 0

        self._init_macro_from_state(self.state)

        return self.state.copy(), {}

    def step(self, action):
        action = int(action)
        allocation = ALLOCATION[action]

        old_Y = self.prev_Y

        self.K += allocation[0] * self.budget
        self.D += allocation[1] * self.budget / 100.0
        self.AI += allocation[2] * self.budget / 20.0
        self.H += allocation[3] * self.budget / 200.0

        new_Y = self._production()
        growth_pct = (new_Y - old_Y) / max(old_Y, 1e-9) * 100.0

        automation_pressure = 1.25 * allocation[2]
        training_relief = 1.35 * allocation[3]
        growth_relief = 0.04 * max(growth_pct - 5.0, -3.0)

        self.U = self.U + automation_pressure - training_relief - growth_relief
        self.U = float(np.clip(self.U, 2.5, 8.5))

        reward = self._reward(growth_pct, action)

        self.state = self._state_from_macro(growth_pct)
        self.prev_Y = new_Y
        self.t += 1

        terminated = self.t >= self.T
        truncated = False

        info = {
            "K": self.K,
            "D": self.D,
            "AI": self.AI,
            "H": self.H,
            "U": self.U,
            "Y": new_Y,
            "growth_pct": growth_pct,
            "action_name": ACTION_NAMES[action],
        }

        return self.state.copy(), reward, terminated, truncated, info


def state_to_text(state):
    state = np.asarray(state, dtype=int)

    parts = []

    for idx, value in enumerate(state):
        parts.append(STATE_LABELS[idx][int(value)])

    return " | ".join(parts)


def action_allocation_table():
    rows = []

    for a in range(N_ACTIONS):
        row = {
            "Hành động": ACTION_SHORT[a],
            "Tên hành động": ACTION_NAMES[a],
            "K": ALLOCATION[a][0],
            "D": ALLOCATION[a][1],
            "AI": ALLOCATION[a][2],
            "H": ALLOCATION[a][3],
        }
        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def train_q_learning_cached(episodes=10000, alpha=0.1, gamma_discount=0.95, seed=42):
    rng = np.random.default_rng(int(seed))
    env = VietnamEconomyEnv()

    Q = np.zeros((3, 3, 3, 3, N_ACTIONS), dtype=float)

    rewards = []
    eps_values = []

    default_start = np.array([1, 1, 0, 1], dtype=int)

    for ep in range(int(episodes)):
        if ep % 5 == 0:
            start_state = default_start.copy()
        else:
            start_state = rng.integers(0, 3, size=4)

        s, _ = env.reset(options={"start_state": start_state})

        eps = max(0.05, 1.0 - 0.95 * ep / max(1, int(episodes) - 1))
        total_reward = 0.0

        while True:
            if rng.random() < eps:
                a = int(rng.integers(0, N_ACTIONS))
            else:
                a = int(np.argmax(Q[tuple(s)]))

            s2, r, done, _, _ = env.step(a)

            old_q = Q[tuple(s) + (a,)]
            target = r + gamma_discount * np.max(Q[tuple(s2)]) * (0 if done else 1)

            Q[tuple(s) + (a,)] = old_q + alpha * (target - old_q)

            s = s2
            total_reward += r

            if done:
                break

        rewards.append(float(total_reward))
        eps_values.append(float(eps))

    learning_df = pd.DataFrame(
        {
            "Episode": np.arange(1, int(episodes) + 1),
            "Reward": rewards,
            "Epsilon": eps_values,
        }
    )

    learning_df["Reward rolling mean"] = learning_df["Reward"].rolling(200, min_periods=1).mean()

    return {
        "Q": Q,
        "learning_records": learning_df.to_dict("records"),
        "episodes": int(episodes),
        "alpha": float(alpha),
        "gamma": float(gamma_discount),
        "seed": int(seed),
    }


def policy_action(Q, state):
    state = np.asarray(state, dtype=int)
    return int(np.argmax(Q[tuple(state)]))


def evaluate_policy(Q=None, policy_type="q", fixed_action=None, n_episodes=300, seed=123):
    rng = np.random.default_rng(int(seed))
    env = VietnamEconomyEnv()

    rewards = []

    for ep in range(int(n_episodes)):
        if ep % 4 == 0:
            start_state = np.array([1, 1, 0, 1], dtype=int)
        else:
            start_state = rng.integers(0, 3, size=4)

        s, _ = env.reset(options={"start_state": start_state})
        total = 0.0

        while True:
            if policy_type == "q":
                a = int(np.argmax(Q[tuple(s)]))
            elif policy_type == "fixed":
                a = int(fixed_action)
            else:
                a = int(rng.integers(0, N_ACTIONS))

            s2, r, done, _, _ = env.step(a)

            total += r
            s = s2

            if done:
                break

        rewards.append(total)

    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "records": rewards,
    }


def policy_report_table(Q):
    states = [
        ("Việt Nam 2026 thực tế", np.array([1, 1, 0, 1])),
        ("Quick win: GDP thấp, D thấp, U cao", np.array([0, 0, 0, 2])),
        ("Consolidation: GDP cao, AI cao, U thấp", np.array([2, 2, 2, 0])),
        ("D trung bình, AI trung bình, U cao", np.array([1, 1, 1, 2])),
        ("GDP cao nhưng AI thấp", np.array([2, 1, 0, 1])),
    ]

    rows = []

    for label, state in states:
        a = policy_action(Q, state)

        rows.append(
            {
                "Trạng thái": label,
                "Mã trạng thái": str(tuple(state.tolist())),
                "Diễn giải trạng thái": state_to_text(state),
                "Hành động π*(s)": ACTION_SHORT[a],
                "Tên hành động": ACTION_NAMES[a],
                "Tỷ trọng K": ALLOCATION[a][0],
                "Tỷ trọng D": ALLOCATION[a][1],
                "Tỷ trọng AI": ALLOCATION[a][2],
                "Tỷ trọng H": ALLOCATION[a][3],
            }
        )

    return pd.DataFrame(rows)


def simulate_one_episode(policy_name, Q=None, fixed_action=None, start_state=None, seed=7):
    rng = np.random.default_rng(int(seed))
    env = VietnamEconomyEnv()

    if start_state is None:
        start_state = np.array([1, 1, 0, 1], dtype=int)

    s, _ = env.reset(options={"start_state": start_state})

    rows = []

    for t in range(T):
        if policy_name == "Q-learning π*":
            a = int(np.argmax(Q[tuple(s)]))
        elif policy_name == "Random":
            a = int(rng.integers(0, N_ACTIONS))
        else:
            a = int(fixed_action)

        s2, r, done, _, info = env.step(a)

        rows.append(
            {
                "Năm": 2026 + t,
                "Chính sách": policy_name,
                "State trước": state_to_text(s),
                "Hành động": ACTION_NAMES[a],
                "Reward": r,
                "K": info["K"],
                "D": info["D"],
                "AI": info["AI"],
                "H": info["H"],
                "U": info["U"],
                "Y": info["Y"],
                "GDP growth (%)": info["growth_pct"],
                "State sau": state_to_text(s2),
            }
        )

        s = s2

        if done:
            break

    return pd.DataFrame(rows)


def comparison_table(Q):
    q_eval = evaluate_policy(Q=Q, policy_type="q", n_episodes=300, seed=100)
    a1_eval = evaluate_policy(policy_type="fixed", fixed_action=0, n_episodes=300, seed=101)
    a3_eval = evaluate_policy(policy_type="fixed", fixed_action=2, n_episodes=300, seed=102)
    random_eval = evaluate_policy(policy_type="random", n_episodes=300, seed=103)

    rows = []

    for name, res in [
        ("Q-learning π*", q_eval),
        ("Luôn chọn a1", a1_eval),
        ("Luôn chọn a3", a3_eval),
        ("Random", random_eval),
    ]:
        rows.append(
            {
                "Chính sách": name,
                "Reward trung bình": res["mean"],
                "Độ lệch chuẩn": res["std"],
                "Reward thấp nhất": res["min"],
                "Reward cao nhất": res["max"],
            }
        )

    return pd.DataFrame(rows).sort_values("Reward trung bình", ascending=False).reset_index(drop=True)


def full_q_policy_table(Q):
    rows = []

    for g in range(3):
        for d in range(3):
            for ai in range(3):
                for u in range(3):
                    s = np.array([g, d, ai, u], dtype=int)
                    a = policy_action(Q, s)

                    rows.append(
                        {
                            "GDP": STATE_LABELS[0][g],
                            "D": STATE_LABELS[1][d],
                            "AI": STATE_LABELS[2][ai],
                            "U": STATE_LABELS[3][u],
                            "Mã trạng thái": str((g, d, ai, u)),
                            "Hành động": ACTION_SHORT[a],
                            "Tên hành động": ACTION_NAMES[a],
                        }
                    )

    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def train_dqn_cached(total_timesteps=5000, seed=42):
    if not DQN_AVAILABLE:
        return {
            "success": False,
            "message": "Chưa cài stable-baselines3 hoặc gymnasium.",
            "mean_reward": np.nan,
        }

    try:
        env = FlattenObservation(VietnamEconomyEnv())

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=5000,
            learning_starts=300,
            batch_size=32,
            gamma=0.95,
            exploration_fraction=0.40,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=0,
            seed=int(seed),
        )

        model.learn(total_timesteps=int(total_timesteps))

        rewards = []

        for ep in range(100):
            raw_env = VietnamEconomyEnv()
            s, _ = raw_env.reset()
            total = 0.0

            for _ in range(T):
                obs = np.asarray(s, dtype=np.int64)
                action, _ = model.predict(obs, deterministic=True)

                s, r, done, _, _ = raw_env.step(int(action))
                total += r

                if done:
                    break

            rewards.append(total)

        return {
            "success": True,
            "message": "DQN đã huấn luyện thử thành công.",
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "total_timesteps": int(total_timesteps),
        }

    except Exception as exc:
        return {
            "success": False,
            "message": str(exc),
            "mean_reward": np.nan,
        }


def make_styled_table(df, decimals=3):
    show_df = df.copy()

    format_dict = {}

    for col in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[col]):
            if str(col).lower() in ["năm", "episode"]:
                format_dict[col] = "{:.0f}"
            else:
                format_dict[col] = "{:." + str(decimals) + "f}"

    styler = show_df.style.format(format_dict)

    try:
        styler = styler.hide(axis="index")
    except Exception:
        try:
            styler = styler.hide_index()
        except Exception:
            pass

    styler = styler.set_properties(
        **{
            "background-color": "#ffffff",
            "color": BRAND,
            "border": f"1px solid {BRAND}",
            "padding": "10px 12px",
            "font-size": "16px",
        }
    )

    styler = styler.set_table_styles(
        [
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", BRAND),
                    ("font-weight", "800"),
                    ("border", f"1px solid {BRAND}"),
                    ("padding", "12px 12px"),
                    ("font-size", "16px"),
                    ("text-align", "left"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", BRAND),
                    ("border", f"1px solid {BRAND}"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("background-color", "#ffffff"),
                    ("color", BRAND),
                ],
            },
        ]
    )

    return styler


def show_table(df, decimals=3):
    st.table(make_styled_table(df, decimals=decimals))


def style_base_fig(fig, height=430):
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=BRAND, size=15),
        title_font=dict(color=BRAND, size=20),
        xaxis=dict(
            showline=True,
            linecolor=BRAND,
            tickfont=dict(color=BRAND),
            title_font=dict(color=BRAND),
        ),
        yaxis=dict(
            showline=True,
            linecolor=BRAND,
            gridcolor="rgba(5,49,81,0.18)",
            tickfont=dict(color=BRAND),
            title_font=dict(color=BRAND),
        ),
        legend=dict(font=dict(color=BRAND)),
    )

    return fig


def render():
    st.title("Bài 11. Reinforcement Learning cho chính sách chuyển đổi số")
    st.caption("Gymnasium Env, Q-learning tabular, epsilon-greedy, policy extraction, rule-based comparison và DQN mở rộng")

    train_res = train_q_learning_cached(
        episodes=10000,
        alpha=0.1,
        gamma_discount=0.95,
        seed=42,
    )

    Q = train_res["Q"]
    learning_df = pd.DataFrame(train_res["learning_records"])

    tabs = st.tabs(
        [
            "11.3.1 Env",
            "11.3.2 Q-learning",
            "11.3.3 Chính sách π*",
            "11.3.4 So sánh chính sách",
            "11.3.5 DQN",
            "11.4 Chính sách",
        ]
    )

    # =====================================================
    # 11.3.1
    # =====================================================
    with tabs[0]:
        st.header("11.3.1. Môi trường Gym/Gymnasium")

        c1, c2, c3 = st.columns(3)
        c1.metric("Gymnasium", "Có" if GYM_AVAILABLE else "Fallback nội bộ")
        c2.metric("Episode length", f"{T} năm")
        c3.metric("Số hành động", f"{N_ACTIONS}")

        env_info = pd.DataFrame(
            {
                "Thành phần": [
                    "observation_space",
                    "action_space",
                    "Trạng thái",
                    "Hành động",
                    "Reward",
                    "reset()",
                    "step()",
                ],
                "Mô tả": [
                    "MultiDiscrete([3, 3, 3, 3])",
                    "Discrete(5)",
                    "GDP growth, D, AI, U; mỗi chỉ số có 3 mức: thấp, trung bình, cao",
                    "5 cách phân bổ ngân sách vào K, D, AI, H",
                    "Tổng hợp tăng trưởng, số hóa, AI readiness và thất nghiệp thấp",
                    "Khởi tạo Việt Nam 2026 hoặc trạng thái giả định",
                    "Cập nhật K, D, AI, H, U, Y và trả về reward",
                ],
            }
        )

        show_table(env_info, decimals=3)

        st.subheader("Bảng hành động")
        show_table(action_allocation_table(), decimals=3)

        sample_env = VietnamEconomyEnv()
        s0, _ = sample_env.reset()

        rows = []

        for a in range(N_ACTIONS):
            env = VietnamEconomyEnv()
            s, _ = env.reset()
            s2, r, done, _, info = env.step(a)

            rows.append(
                {
                    "Hành động": ACTION_NAMES[a],
                    "State đầu": state_to_text(s),
                    "Reward năm đầu": r,
                    "GDP growth (%)": info["growth_pct"],
                    "U sau": info["U"],
                    "State sau": state_to_text(s2),
                }
            )

        st.subheader("Mô phỏng một bước từ trạng thái Việt Nam 2026")
        show_table(pd.DataFrame(rows), decimals=3)

        st.success(
            "Môi trường đã có đủ reset(), step(), action_space và observation_space. Một episode tương ứng 10 năm chính sách."
        )

    # =====================================================
    # 11.3.2
    # =====================================================
    with tabs[1]:
        st.header("11.3.2. Huấn luyện Q-learning tabular")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Episodes", f"{train_res['episodes']:,}")
        c2.metric("α", f"{train_res['alpha']:.2f}")
        c3.metric("γ", f"{train_res['gamma']:.2f}")
        c4.metric("ε cuối", f"{learning_df['Epsilon'].iloc[-1]:.2f}")

        param_df = pd.DataFrame(
            {
                "Tham số": [
                    "Q-table shape",
                    "Learning rate α",
                    "Discount γ",
                    "Epsilon start",
                    "Epsilon end",
                    "Episodes",
                    "Episode length",
                ],
                "Giá trị": [
                    str(Q.shape),
                    "0.10",
                    "0.95",
                    "1.00",
                    "0.05",
                    "10.000",
                    "10 năm",
                ],
            }
        )

        show_table(param_df, decimals=3)

        fig_learning = px.line(
            learning_df,
            x="Episode",
            y="Reward rolling mean",
            title="Learning curve - rolling mean 200 episodes",
        )
        fig_learning.update_traces(
            line=dict(color=BRAND, width=4),
        )
        fig_learning.update_layout(
            xaxis_title="Episode",
            yaxis_title="Reward rolling mean",
        )
        style_base_fig(fig_learning, height=430)
        st.plotly_chart(fig_learning, use_container_width=True)

        fig_eps = px.line(
            learning_df,
            x="Episode",
            y="Epsilon",
            title="Epsilon giảm dần từ 1.0 xuống 0.05",
        )
        fig_eps.update_traces(
            line=dict(color=BRAND, width=4),
        )
        fig_eps.update_layout(
            xaxis_title="Episode",
            yaxis_title="Epsilon",
        )
        style_base_fig(fig_eps, height=360)
        st.plotly_chart(fig_eps, use_container_width=True)

        st.success(
            "Q-learning đã được huấn luyện với epsilon-greedy giảm dần. Learning curve dùng trung bình trượt để dễ quan sát xu hướng học."
        )

    # =====================================================
    # 11.3.3
    # =====================================================
    with tabs[2]:
        st.header("11.3.3. Trích xuất chính sách π*(s) = argmax Q(s,a)")

        report_df = policy_report_table(Q)

        show_table(report_df, decimals=3)

        action_count = full_q_policy_table(Q)["Tên hành động"].value_counts().reset_index()
        action_count.columns = ["Tên hành động", "Số trạng thái chọn"]

        fig_policy = px.bar(
            action_count,
            x="Tên hành động",
            y="Số trạng thái chọn",
            color="Tên hành động",
            text="Số trạng thái chọn",
            title="Tần suất hành động được π* chọn trên toàn bộ 81 trạng thái",
            color_discrete_map=MULTI_COLORS,
        )
        fig_policy.update_traces(
            textposition="outside",
            marker_line_color="white",
            marker_line_width=1,
        )
        fig_policy.update_layout(
            xaxis_title="Hành động",
            yaxis_title="Số trạng thái",
        )
        style_base_fig(fig_policy, height=430)
        st.plotly_chart(fig_policy, use_container_width=True)

        with st.expander("Xem toàn bộ policy table 81 trạng thái"):
            show_table(full_q_policy_table(Q), decimals=3)

        st.success(
            "Đã trích xuất chính sách tối ưu tabular π*(s). Bảng trên báo cáo trạng thái Việt Nam 2026 và 4 trạng thái giả định."
        )

    # =====================================================
    # 11.3.4
    # =====================================================
    with tabs[3]:
        st.header("11.3.4. So sánh π* với 3 chính sách rule-based")

        comp_df = comparison_table(Q)
        best_policy = comp_df.iloc[0]["Chính sách"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Chính sách tốt nhất", best_policy)
        c2.metric("Reward trung bình cao nhất", f"{comp_df.iloc[0]['Reward trung bình']:.2f}")
        c3.metric("Số policy so sánh", "4")

        show_table(comp_df, decimals=3)

        fig_comp = px.bar(
            comp_df,
            x="Chính sách",
            y="Reward trung bình",
            color="Chính sách",
            text=comp_df["Reward trung bình"].round(2),
            title="So sánh reward tích lũy trung bình",
            color_discrete_map=MULTI_COLORS,
        )
        fig_comp.update_traces(
            textposition="outside",
            marker_line_color="white",
            marker_line_width=1,
        )
        fig_comp.update_layout(
            xaxis_title="Chính sách",
            yaxis_title="Reward trung bình",
        )
        style_base_fig(fig_comp, height=430)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.subheader("Mô phỏng một episode từ trạng thái Việt Nam 2026")

        episode_rows = []

        for name, fixed_action in [
            ("Q-learning π*", None),
            ("Luôn chọn a1", 0),
            ("Luôn chọn a3", 2),
            ("Random", None),
        ]:
            ep_df = simulate_one_episode(
                name,
                Q=Q,
                fixed_action=fixed_action,
                start_state=np.array([1, 1, 0, 1]),
                seed=7,
            )
            episode_rows.append(ep_df)

        episode_df = pd.concat(episode_rows, ignore_index=True)

        reward_path = episode_df.groupby(["Năm", "Chính sách"], as_index=False)["Reward"].sum()
        reward_path["Reward tích lũy"] = reward_path.groupby("Chính sách")["Reward"].cumsum()

        fig_path = px.line(
            reward_path,
            x="Năm",
            y="Reward tích lũy",
            color="Chính sách",
            markers=True,
            title="Reward tích lũy trong một episode 10 năm",
            color_discrete_map=MULTI_COLORS,
        )
        fig_path.update_traces(line=dict(width=4), marker=dict(size=8))
        fig_path.update_layout(
            xaxis_title="Năm",
            yaxis_title="Reward tích lũy",
        )
        style_base_fig(fig_path, height=450)
        st.plotly_chart(fig_path, use_container_width=True)

        with st.expander("Xem chi tiết episode mô phỏng"):
            show_table(episode_df, decimals=3)

        st.success(
            "π* được so sánh với ba chính sách rule-based: luôn chọn a1, luôn chọn a3 và random."
        )

    # =====================================================
    # 11.3.5
    # =====================================================
    with tabs[4]:
        st.header("11.3.5. Mở rộng DQN bằng stable-baselines3")

        dqn_info = pd.DataFrame(
            {
                "Thành phần": [
                    "Thuật toán",
                    "Policy network",
                    "Hidden layers",
                    "Units mỗi layer",
                    "Gamma",
                    "Exploration final eps",
                    "Trạng thái cài đặt",
                ],
                "Giá trị": [
                    "DQN",
                    "MlpPolicy",
                    "2",
                    "64",
                    "0.95",
                    "0.05",
                    "Có thể chạy" if DQN_AVAILABLE else "Chưa cài stable-baselines3/gymnasium",
                ],
            }
        )

        show_table(dqn_info, decimals=3)

        st.info(
            "DQN là phần mở rộng nặng hơn. Để tránh Streamlit chạy chậm, phần này chỉ chạy khi bạn bật tùy chọn bên dưới."
        )

        run_dqn = st.checkbox("Chạy thử DQN 5.000 timesteps", value=False)

        if run_dqn:
            dqn_res = train_dqn_cached(total_timesteps=5000, seed=42)

            if dqn_res["success"]:
                c1, c2, c3 = st.columns(3)
                c1.metric("DQN", "Thành công")
                c2.metric("Reward TB", f"{dqn_res['mean_reward']:.3f}")
                c3.metric("Timesteps", f"{dqn_res['total_timesteps']:,}")

                q_eval = evaluate_policy(Q=Q, policy_type="q", n_episodes=100, seed=99)

                dqn_compare = pd.DataFrame(
                    {
                        "Phương pháp": ["Q-learning tabular", "DQN"],
                        "Reward trung bình": [q_eval["mean"], dqn_res["mean_reward"]],
                        "Độ lệch chuẩn": [q_eval["std"], dqn_res["std_reward"]],
                    }
                )

                show_table(dqn_compare, decimals=3)

                fig_dqn = px.bar(
                    dqn_compare,
                    x="Phương pháp",
                    y="Reward trung bình",
                    color="Phương pháp",
                    text=dqn_compare["Reward trung bình"].round(2),
                    title="So sánh Q-learning tabular và DQN thử nghiệm",
                )
                fig_dqn.update_traces(
                    marker_color=BRAND,
                    textposition="outside",
                    textfont=dict(color=BRAND),
                )
                fig_dqn.update_layout(
                    xaxis_title="Phương pháp",
                    yaxis_title="Reward trung bình",
                )
                style_base_fig(fig_dqn, height=390)
                st.plotly_chart(fig_dqn, use_container_width=True)

                st.success(
                    "DQN đã chạy thử. Với state space nhỏ 3×3×3×3, Q-learning tabular thường đã đủ tốt; DQN chỉ thực sự có lợi khi trạng thái liên tục hoặc rất lớn."
                )
            else:
                st.warning(dqn_res["message"])
        else:
            st.success(
                "Với bài này, Q-learning tabular là lựa chọn ổn định nhất vì không gian trạng thái chỉ có 81 trạng thái. "
                "DQN là mở rộng phù hợp khi tăng số biến trạng thái hoặc dùng dữ liệu liên tục."
            )

    # =====================================================
    # 11.4
    # =====================================================
    with tabs[5]:
        st.header("11.4. Thảo luận chính sách")

        low_state = np.array([0, 0, 0, 2])
        high_state = np.array([2, 2, 2, 0])

        low_action = policy_action(Q, low_state)
        high_action = policy_action(Q, high_state)

        st.markdown("### a) Khi GDP thấp, D thấp, U cao, chính sách π*(s) chọn gì?")

        st.success(
            f"Ở trạng thái **GDP thấp, D thấp, AI thấp, U cao**, π*(s) chọn: "
            f"**{ACTION_NAMES[low_action]}**."
        )

        st.markdown(
            """
            Đây là trạng thái cần phản ứng chính sách nhanh. Nếu policy chọn nhóm hành động thiên về hạ tầng,
            dữ liệu hoặc nhân lực, có thể hiểu là mô hình đang tìm “quick win” để phục hồi nền tảng tăng trưởng
            và giảm thất nghiệp. Trong bối cảnh thất nghiệp cao, hành động quá thiên về AI có thể bị phạt
            nếu năng lực nhân lực chưa đủ, vì tự động hóa nhanh có thể làm tăng áp lực dịch chuyển lao động.

            Vì vậy, “quick win” hợp lý thường là kết hợp giữa hạ tầng số, dữ liệu nền tảng và đào tạo lại,
            thay vì chỉ tăng tốc AI đơn thuần.
            """
        )

        st.markdown("### b) Khi GDP cao, AI cao, U thấp, chính sách chọn gì?")

        st.success(
            f"Ở trạng thái **GDP cao, AI cao, U thấp**, π*(s) chọn: "
            f"**{ACTION_NAMES[high_action]}**."
        )

        st.markdown(
            """
            Trạng thái này giống giai đoạn “consolidation”: nền kinh tế đã có tăng trưởng và năng lực AI tốt,
            thất nghiệp thấp nên chính sách không nhất thiết phải tiếp tục bơm mạnh vào AI.
            Hành động phù hợp thường là củng cố hạ tầng, dữ liệu, nhân lực và an sinh để giữ hệ thống ổn định.

            Nếu policy chọn cân bằng hoặc nhân lực, điều đó phản ánh logic duy trì năng lực hấp thụ,
            giảm rủi ro xã hội và tránh tăng trưởng nóng dựa quá nhiều vào tự động hóa.
            """
        )

        st.markdown("### c) Tích hợp π* vào quy trình hoạch định chính sách thế nào?")

        st.markdown(
            """
            Chính sách π* không nên được dùng như một mệnh lệnh tự động. AI và reinforcement learning chỉ nên là
            công cụ hỗ trợ phân tích, không thay thế quyết định chính trị - xã hội.

            Có thể tích hợp π* vào quy trình hoạch định chính sách Việt Nam theo bốn bước:

            **Thứ nhất, dùng π* như hệ thống khuyến nghị kỹ thuật.**
            Mô hình đưa ra hành động đề xuất theo trạng thái kinh tế - xã hội, kèm reward dự kiến
            và các trade-off về tăng trưởng, số hóa, AI và thất nghiệp.

            **Thứ hai, yêu cầu giải trình.**
            Mỗi khuyến nghị cần đi kèm diễn giải: vì sao chọn hành động đó, tiêu chí nào chi phối,
            nhóm lao động nào có thể chịu tác động và rủi ro chính sách là gì.

            **Thứ ba, đưa vào hội đồng chính sách đa bên.**
            Khuyến nghị của π* cần được xem xét bởi chuyên gia kinh tế, công nghệ, lao động,
            đại diện địa phương, doanh nghiệp và tổ chức xã hội.

            **Thứ tư, giữ quyền quyết định cuối cùng cho cơ quan có thẩm quyền.**
            Chính phủ hoặc Quốc hội có thể chấp nhận, điều chỉnh hoặc bác bỏ khuyến nghị
            dựa trên mục tiêu phát triển, công bằng xã hội, an ninh dữ liệu và tính chính danh.

            Như vậy, π* là “bản đồ gợi ý chính sách”, không phải người ra quyết định.
            """
        )

        st.info(
            "Kết luận: Q-learning giúp học chính sách thích ứng theo trạng thái, nhưng quyết định cuối cùng vẫn phải qua quy trình governance minh bạch và có trách nhiệm giải trình."
        )


def run():
    render()
