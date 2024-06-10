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


A = A + A_OFFSET
B = B + B_OFFSET

A = A * A_QUANT
B = B * B_QUANT

def quant_i8_per_tensor(A):
    MIN = torch.min(A)
    MAX = torch.max(A)

    MIN = 0.0
    RANGE = MAX - MIN
    OFFSET = MIN
    SCALE = 255.0 / RANGE

    OFFSET = torch.full((A.shape[0],), OFFSET)
    SCALE = torch.full((A.shape[0],), SCALE)
    A = (A - OFFSET.unsqueeze(1)) * SCALE.unsqueeze(1)
    A = torch.round(A)
    A = torch.clamp(A, 0, 255.0)
    A = A.to(torch.uint8)
    return A, SCALE


def quant_i8_per_channel(A,axis=1):
    MIN = 0
    MAX = torch.max(A, axis=axis).values
    SCALE = 255.0 / (MAX - MIN)
    A = A * SCALE.unsqueeze(axis)
    A = torch.round(A)
    A = torch.clamp(A, 0, 255.0)
    A = A.to(torch.uint8)

    return A, SCALE


def mmt_quant(A, B, per_tensor=True):
    if per_tensor:
        A, A_SCALE = quant_i8_per_tensor(A)
        B, B_SCALE = quant_i8_per_tensor(B)
    else:
        A, A_SCALE = quant_i8_per_channel(A)
        B, B_SCALE = quant_i8_per_channel(B)

    B = torch.transpose(B, 0, 1)

    A = A.to(torch.float)
    B = B.to(torch.float)
    MM = torch.mm(A, B)

    MM = MM / A_SCALE.unsqueeze(1)
    MM = MM / B_SCALE.unsqueeze(0)
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

