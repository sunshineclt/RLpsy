

PARAMS = ["alpha", "eta", "tau", "gamma", "forget_MF", "forget_MB", "I", "k"]
PARAM_BOUNDS = {"alpha": (0, 1),
                "eta": (0, 1),
                "tau": (1e-5, 100),
                "gamma": (0, 1),
                "forget_MF": (0, 0.05),
                "forget_MB": (0, 0.05),
                "I": (0, 1),
                "k": (0, 0.1),
                "gate": (0, 1)}
