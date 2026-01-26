import torch
from cs336_basics.optimizer import SGD


def run_experiment(lr):
    print(f"\n{'=' * 20}")
    print(f"Learning Rate: {lr}")
    print(f"{'=' * 20}")

    # Ensure same initialization for comparison
    torch.manual_seed(42)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

    try:
        opt = SGD([weights], lr=lr)

        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            print(f"Iter {t + 1}: Loss = {loss.item():.4f}")
            loss.backward()
            opt.step()

    except Exception as e:
        print(f"Error: {e}")


# Run for requested learning rates
learning_rates = [10.0, 100.0, 1000.0]
for lr in learning_rates:
    run_experiment(lr)
