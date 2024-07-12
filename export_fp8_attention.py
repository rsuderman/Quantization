
import torch.nn
import sharktank.ops as ops
from shark_turbine import aot


class AttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return ops.scaled_dot_product_attention(q, k, v, a=None)

q = torch.zeros((20, 4096, 64), dtype=torch.float8_e4m3fnuz, device="cuda:0")
k = torch.zeros((20, 4096, 64), dtype=torch.float8_e4m3fnuz, device="cuda:0")
v = torch.zeros((20, 4096, 64), dtype=torch.float8_e4m3fnuz, device="cuda:0")

inputs = {
    "q" : q,
    "k" : k,
    "v" : v, 
}

if __name__ == "__main__":
    pass

mdl = AttentionModel()

# Temporary: Need a dedicated exporter.
output = aot.export(
    mdl,
    kwargs=inputs,
)

output.print_readable()
