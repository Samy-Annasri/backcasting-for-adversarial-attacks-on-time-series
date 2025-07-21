import torch
import numpy as np

def fgsm_surrogate_attack(
    model_surrogate, model_target,
    data_loader, epsilon,
    price_min, price_max
):
    device = next(model_surrogate.parameters()).device
    model_surrogate.eval()
    model_target.eval()
    loss_fn = torch.nn.MSELoss()

    adv_predictions = []
    true_values = []

    for x, y in data_loader:
        x = x.to(device).clone().detach().requires_grad_(True)
        y = y.to(device).clone().detach()

        model_surrogate.train()

        # 1. Forward pass on the surrogate
        output = model_surrogate(x)
        loss = loss_fn(output, y)

        # 2. Backward to get gradients
        model_surrogate.zero_grad()
        loss.backward()

        model_surrogate.eval()

        data_grad = x.grad.data
        x_adv = x + epsilon * data_grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)

        # 3. Evaluate prediction on the true target model
        with torch.no_grad():
            output_adv = model_target(x_adv)

        adv_predictions.extend(output_adv.cpu().squeeze().numpy())
        true_values.extend(y.cpu().squeeze().numpy())

    # Denormalize
    adv_predictions = np.array(adv_predictions) * (price_max - price_min) + price_min
    true_values = np.array(true_values) * (price_max - price_min) + price_min

    return true_values, adv_predictions
