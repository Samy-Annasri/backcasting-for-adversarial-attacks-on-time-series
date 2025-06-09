import torch
import torch.nn as nn

class SimpleGRU(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(SimpleGRU, self).__init__()

    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    self.linear = nn.Linear(hidden_size,output_size)

  def forward(self,x):

    h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device) #memorie of the GRU

    out, _ = self.gru(x, h0)

    out = out[:, -1, :] # only take the prediction

    out = self.linear(out)
    return out
