import numpy as np

from cs231n.classifiers.fc_net import FullyConnectedNet
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver


full_data = get_CIFAR10_data()
data = {
    "X_train": full_data["X_train"][:50],
    "y_train": full_data["y_train"][:50],
    "X_val": full_data["X_val"],
    "y_val": full_data["y_val"],
}
configs = [
    (5e-2, 1e-3),
    (7e-2, 5e-4),
    (7e-2, 1e-3),
    (1e-1, 5e-4),
    (1e-1, 1e-3),
    (1.2e-1, 5e-4),
]

for weight_scale, learning_rate in configs:
    final_accuracies = []
    for seed in range(5):
        np.random.seed(seed)
        model = FullyConnectedNet(
            [100, 100, 100, 100], weight_scale=weight_scale, dtype=np.float64
        )
        solver = Solver(
            model,
            data,
            print_every=999,
            num_epochs=20,
            batch_size=25,
            update_rule="sgd",
            optim_config={"learning_rate": learning_rate},
            verbose=False,
        )
        solver.train()
        final_accuracies.append(solver.train_acc_history[-1])
    print(
        f"weight_scale={weight_scale:g} learning_rate={learning_rate:g} "
        f"successes={sum(acc == 1.0 for acc in final_accuracies)}/5 "
        f"final_accs={final_accuracies}"
    )
