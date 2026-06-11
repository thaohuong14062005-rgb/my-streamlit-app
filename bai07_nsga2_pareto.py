def solve_nsga2_cached(pop_size=100, n_gen=200, seed=42):
    if not PYMOO_AVAILABLE:
        return {
            "success": False,
            "message": "Chưa cài pymoo.",
            "X": None,
            "F": None,
            "pareto": pd.DataFrame(),
        }

    problem = VietnamDigitalProblem()

    algorithm = NSGA2(
        pop_size=int(pop_size),
        eliminate_duplicates=True,
    )

    try:
        res = minimize(
            problem,
            algorithm,
            ("n_gen", int(n_gen)),
            seed=int(seed),
            verbose=False,
        )

        X = res.X
        F = res.F

        if X is None or F is None:
            X_pop = res.pop.get("X")
            F_pop = res.pop.get("F")
            G_pop = res.pop.get("G")

            feasible = np.all(G_pop <= 1e-6, axis=1)
            X = X_pop[feasible]
            F = F_pop[feasible]

            if len(F) > 0:
                nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
                X = X[nds]
                F = F[nds]

        if X is None or F is None or len(F) == 0:
            return {
                "success": False,
                "message": "Không tìm thấy nghiệm Pareto khả thi.",
                "X": None,
                "F": None,
                "pareto": pd.DataFrame(),
            }

        X = np.asarray(X, dtype=float)
        F = np.asarray(F, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if F.ndim == 1:
            F = F.reshape(1, -1)

        pareto = pd.DataFrame(
            {
                "Nghiệm": [f"S{i + 1}" for i in range(len(F))],
                "GDP gain": (-F[:, 0]).astype(float),
                "Bao trùm cost": F[:, 1].astype(float),
                "Phát thải": F[:, 2].astype(float),
                "Rủi ro ròng": F[:, 3].astype(float),
                "Tổng ngân sách": X.sum(axis=1).astype(float),
            }
        )

        pareto = pareto.reset_index(drop=True)

        return {
            "success": True,
            "message": "Optimal Pareto population extracted.",
            "X": X,
            "F": F,
            "pareto": pareto,
        }

    except Exception as exc:
        return {
            "success": False,
            "message": str(exc),
            "X": None,
            "F": None,
            "pareto": pd.DataFrame(),
        }
