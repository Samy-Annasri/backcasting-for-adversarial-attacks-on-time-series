import torch

def reverse_forecast_attack(
    model_rev, model_normal,
    test_loader_rev, test_loader,
    epsilons,
    price_min, price_max
):
    device = next(model_rev.parameters()).device
    model_rev.eval()
    model_normal.eval()
    loss_fn = torch.nn.MSELoss()
    
    results = {}
    
    for esp in epsilons:
        all_true = []
        all_pred = []
        
        for (x_rev, y_rev), (x_normal, y_normal) in zip(test_loader_rev, test_loader):
            x_rev = x_rev.to(device).clone().detach().requires_grad_(True)
            y_rev = y_rev.to(device).clone().detach()
            x_normal = x_normal.to(device)
            y_normal = y_normal.to(device)

            model_rev.train()

            # Predict with reversed model
            output_rev = model_rev(x_rev)

            # Compute loss and backward
            loss = loss_fn(output_rev, y_rev)
            model_rev.zero_grad()
            loss.backward()

            model_rev.eval()

            # FGSM attack on reversed input
            x_adv_reversed = x_rev + float(esp) * x_rev.grad.sign()
            x_adv_reversed = torch.clamp(x_adv_reversed, 0, 1)

            # Flip back to normal order
            x_adv = x_adv_reversed.flip(dims=[1])

            # Predict with normal model on adversarial data
            with torch.no_grad():
                pred = model_normal(x_adv)

            # Denormalize (pass to CPU first)
            pred_denorm = pred.cpu().numpy() * (price_max - price_min) + price_min
            y_denorm = y_normal.cpu().numpy() * (price_max - price_min) + price_min

            all_true.extend(y_denorm.flatten())
            all_pred.extend(pred_denorm.flatten())

        results[esp] = (all_true, all_pred)
    
    return results
