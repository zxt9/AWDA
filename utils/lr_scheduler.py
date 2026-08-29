"""Polynomial learning-rate schedule."""


class Poly:
    def __init__(self, optimizer, num_epochs, iters_per_epoch):
        self.optimizer = optimizer
        self.total_iters = num_epochs * iters_per_epoch
        self.current_iter = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        self.current_iter += 1
        factor = max(1.0 - self.current_iter / self.total_iters, 0.0) ** 0.9
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def state_dict(self):
        return {"current_iter": self.current_iter, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.current_iter = state["current_iter"]
        self.base_lrs = state["base_lrs"]
