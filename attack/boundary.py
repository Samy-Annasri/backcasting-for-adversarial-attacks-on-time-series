import torch
import numpy as np

def boundary_attack(model, test_loader, epsilon, price_min, price_max, num_iter=50, delta=0.1, eta=0.01):
    """
    Decision-based boundary attack for regression models.
    """
    model.eval()
    true_vals = []
    adv_preds = []

    for data, target in test_loader:
        data, target = data.float(), target.float()

        # Step 1: Initialize adversarial example with random perturbation
        perturbation = torch.randn_like(data) * epsilon
        adv_data = torch.clamp(data + perturbation, 0, 1)

        # Step 2: Compute initial MAE
        best_adv_data = adv_data.clone()
        best_mae = torch.mean(torch.abs(model(best_adv_data) - target)).item()

        # Step 3: Iteratively try to improve adversarial example
        for _ in range(num_iter):
            # Compute direction towards original data
            direction = data - adv_data
            direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8) # 1e-8 to avoid division by zero

            # Create candidate adversarial example
            candidate = adv_data + delta * direction + eta * torch.randn_like(data)
            candidate = torch.clamp(candidate, 0, 1)

            candidate = data + torch.clamp(candidate - data, -epsilon, epsilon)
            candidate = torch.clamp(candidate, 0, 1)

            # Evaluate candidate
            candidate_pred = model(candidate).detach()
            candidate_mae = torch.mean(torch.abs(candidate_pred - target)).item()

            # Keep if better
            if candidate_mae > best_mae:
                best_adv_data = candidate.clone()
                best_mae = candidate_mae

            # save the candidate to iterate
            adv_data = candidate

        # Final prediction using the best adversarial example found
        best_adv_pred = model(best_adv_data).detach()

        true_vals.append(target.detach().cpu().numpy())
        adv_preds.append(best_adv_pred.detach().cpu().numpy())

    true_vals = np.concatenate(true_vals, axis=0).flatten()
    adv_preds = np.concatenate(adv_preds, axis=0).flatten()

    true_vals_denorm = true_vals * (price_max - price_min) + price_min
    adv_preds_denorm = adv_preds * (price_max - price_min) + price_min

    return true_vals_denorm, adv_preds_denorm
