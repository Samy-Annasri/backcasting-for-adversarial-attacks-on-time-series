def fgsm_attack(model, x, y_true, loss_fn, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)
    y_pred = model(x_adv)
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    perturbation = epsilon * x_adv.grad.sign()
    x_adv = x_adv + perturbation
    return x_adv.detach()

