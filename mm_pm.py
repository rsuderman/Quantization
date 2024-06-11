import matplotlib.pyplot as plt
import torch

A_SHAPE = (8, 128)
B_SHAPE = (16, 128)

torch.manual_seed(12345)
A_OFFSET = torch.rand((A_SHAPE[0],1), dtype=torch.float)
B_OFFSET = torch.rand((B_SHAPE[0],1), dtype=torch.float)

A_QUANT = torch.rand((A_SHAPE[0],1), dtype=torch.float)
B_QUANT = torch.rand((B_SHAPE[0],1), dtype=torch.float)

def generate_input(shape):
    M = torch.rand(shape, dtype=torch.float)
    return M

A = generate_input(A_SHAPE)
B = generate_input(B_SHAPE)


A = A + 1.0
B = B + 1.0

A = A * A_QUANT
B = B * B_QUANT

def quant_i8_per_tensor(A, offset=True):
    MIN = 0.0
    MAX = torch.max(A)

    if offset:
        MIN = torch.min(A)

    RANGE = MAX - MIN
    OFFSET = MIN
    SCALE = 255.0 / RANGE

    SCALE = torch.full((A.shape[0],), SCALE)
    OFFSET = torch.full((A.shape[0],), OFFSET)
    OFFSET = OFFSET * SCALE

    A = A * SCALE.unsqueeze(1) - OFFSET.unsqueeze(1)
    A = torch.round(A)
    A = torch.clamp(A, 0, 255.0)
    A = A.to(torch.uint8)

    SCALE = 1.0 / SCALE
    return A, SCALE, -OFFSET


def quant_i8_per_channel(A,axis=1, offset=False):
    MIN = torch.full((A.shape[0],), 0.0)
    MAX = torch.max(A, axis=axis).values

    if offset:
        MIN = torch.min(A, axis=axis).values

    SCALE = 255.0 / (MAX - MIN)
    OFFSET = MIN * SCALE
    A = A * SCALE.unsqueeze(axis) - OFFSET.unsqueeze(axis)
    SCALE = 1.0 / SCALE

    A = torch.round(A)
    A = torch.clamp(A, 0, 255.0)
    A = A.to(torch.uint8)

    return A, SCALE, -OFFSET


def mmt_quant(A, B, per_tensor=False, offset=True):
    if per_tensor:
        A, A_SCALE, A_OFFSET = quant_i8_per_tensor(A, offset=offset)
        B, B_SCALE, B_OFFSET = quant_i8_per_tensor(B, offset=offset)
    else:
        A, A_SCALE, A_OFFSET = quant_i8_per_channel(A, offset=offset)
        B, B_SCALE, B_OFFSET = quant_i8_per_channel(B, offset=offset)

    B = torch.transpose(B, 0, 1)

    A = A.to(torch.float)
    B = B.to(torch.float)
    MM = torch.mm(A, B)

    A_FIX = torch.sum(B, axis=0, keepdim=True) * A_OFFSET.unsqueeze(1)
    B_FIX = torch.sum(A, axis=1, keepdim=True) * B_OFFSET.unsqueeze(0)
    AB_FIX = A_OFFSET.unsqueeze(1) * B_OFFSET.unsqueeze(0) * A.shape[1]

    MM = MM - A_FIX - B_FIX
    MM = MM + AB_FIX

    # Apply the scaling once normalization is fixed:
    MM = MM * A_SCALE.unsqueeze(1)
    MM = MM * B_SCALE.unsqueeze(0)

    return MM


def mmt_float(A, B):
    B = torch.transpose(B, 0, 1)
    return torch.mm(A, B)


OUT_QUANT = mmt_quant(A, B)
OUT_FLOAT = mmt_float(A, B)

OUT_DIFF = OUT_QUANT - OUT_FLOAT

print(torch.mean(torch.abs(OUT_FLOAT)).item())
print(torch.mean(torch.abs(OUT_QUANT)).item())
print(torch.mean(torch.abs(OUT_DIFF)).item())

