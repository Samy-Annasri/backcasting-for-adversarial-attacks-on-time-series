import torch
import numpy as np

def evaluate_model_births(model, test_loader, dates_test, train_size):

    model.eval()
    device = next(model.parameters()).device

    real_values = []
    predicted_values = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            real_values.append(labels.cpu().numpy())
            predicted_values.append(outputs.cpu().numpy())

    real_values = np.concatenate(real_values).squeeze()
    predicted_values = np.concatenate(predicted_values).squeeze()

    return {
        'real_values': real_values,
        'predicted_values': predicted_values,
        'test_dates': dates_test[:len(real_values)]
    }
