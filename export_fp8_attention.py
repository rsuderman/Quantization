
import torch.nn
import sharktank.ops as ops
from shark_turbine import aot

from sharktank.types import PlanarQuantizedTensor
from sharktank.types import QuantizedTensor
from sharktank.types.layouts import TensorScaledLayout

def make_q_tensor(tensor, scale):
    return PlanarQuantizedTensor(
        name="qq", shape=tensor.shape, layout=TensorScaledLayout(
            shape = tensor.shape,
            qs=tensor,
            d=torch.scalar_tensor(scale, dtype=torch.float16),
            dtype=torch.float16))

class AttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(
        self, s: torch.Tensor,
        q:  torch.Tensor, k: torch.Tensor,  v: torch.Tensor,
        qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor):

        q = make_q_tensor(q, qs)
        k = make_q_tensor(k, ks)
        v = make_q_tensor(v, vs)

        return ops.scaled_dot_product_attention(q, k, v, a=None)

q = torch.zeros((1, 1, 4096, 64), dtype=torch.float8_e4m3fnuz, device="cuda:0")
k = torch.zeros((1, 1, 4096, 64), dtype=torch.float8_e4m3fnuz, device="cuda:0")
v = torch.zeros((1, 1, 4096, 64), dtype=torch.float8_e4m3fnuz, device="cuda:0")
s = torch.zeros((), dtype=torch.float32, device="cuda:0")
qs = torch.zeros((), dtype=torch.float32, device="cuda:0")
ks = torch.zeros((), dtype=torch.float32, device="cuda:0")
vs = torch.zeros((), dtype=torch.float32, device="cuda:0")



inputs = {
    "q" : q,
    "k" : k,
    "v" : v, 
    "s" : s,
    "qs" : qs,
    "ks" : ks,
    "vs" : vs, 
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
