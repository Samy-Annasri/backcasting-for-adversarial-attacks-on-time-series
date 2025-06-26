import torch

def reverse_forecast_attack_bim(
    model_rev, model_normal,
    test_loader_rev, test_loader,
    epsilons,
    price_min, price_max,
    alpha,
    num_iter
):
    model_rev.eval()
    model_normal.eval()
    loss_fn = torch.nn.MSELoss()

    results = {}

    for eps in epsilons:
        all_true = []
        all_pred = []

        for (x_rev, y_rev), (x_normal, y_normal) in zip(test_loader_rev, test_loader):
            x_adv = x_rev.clone().detach()
            x_adv.requires_grad = True

            for _ in range(num_iter):
                output_rev = model_rev(x_adv)
                loss = loss_fn(output_rev, y_rev)

                model_rev.zero_grad()
                loss.backward()

                grad_sign = x_adv.grad.data.sign()
                x_adv = x_adv + alpha * grad_sign

                # Clip to stay within epsilon-ball of original reversed input
                x_adv = torch.max(torch.min(x_adv, x_rev + eps), x_rev - eps)
                x_adv = torch.clamp(x_adv, 0, 1).detach()
                x_adv.requires_grad = True

            # Flip back to normal order
            x_adv_flipped = x_adv.flip(dims=[1])

            with torch.no_grad():
                pred = model_normal(x_adv_flipped)

            pred_denorm = pred.cpu().numpy() * (price_max - price_min) + price_min
            y_denorm = y_normal.cpu().numpy() * (price_max - price_min) + price_min

            all_true.extend(y_denorm.flatten())
            all_pred.extend(pred_denorm.flatten())

        results[eps] = (all_true, all_pred)

    return results
