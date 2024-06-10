import matplotlib.pyplot as plt
import torch

A_SHAPE = (4, 8, 16, 16)
B_SHAPE = (8, 8, 4, 4)

torch.manual_seed(12345)
def generate_input(shape):
    M = torch.rand(shape, dtype=torch.float)
    return M

A = generate_input(A_SHAPE)
B = generate_input(B_SHAPE)

A_QUANT = torch.rand((A_SHAPE[0],1,1,1), dtype=torch.float)
B_QUANT = torch.rand((B_SHAPE[0],1,1,1), dtype=torch.float)
A = A * A_QUANT
B = B * B_QUANT

def quant_i8_per_tensor(A):
    SCALE = torch.full((1,), 255.0)
    SCALE = SCALE / torch.max(A)
    A = A * SCALE
    A = torch.round(A)
    A = torch.clamp(A, 0, 255.0)
    A = A.to(torch.uint8)
    return A, SCALE


def quant_i8_per_channel(A,axislist):
    SCALE = A
    for axis in axislist:
        SCALE = torch.max(SCALE, axis=axis, keepdim=True).values
    SCALE = 255.0 / SCALE
    A = A * SCALE
    A = torch.round(A)
    A = torch.clamp(A, 0, 255.0)
    A = A.to(torch.uint8)
    SCALE = torch.flatten(SCALE)
    return A, SCALE


def mmt_quant(A, B, per_channel=True):
    if per_channel:
        A, A_SCALE = quant_i8_per_channel(A, axislist=[1,2,3])
        B, B_SCALE = quant_i8_per_channel(B, axislist=[1,2,3])
    else:
        A, A_SCALE = quant_i8_per_tensor(A)
        B, B_SCALE = quant_i8_per_tensor(B)


    A = A.to(torch.float)
    B = B.to(torch.float)
    CONV = torch.nn.functional.conv2d(A, B)

    A_SCALE = A_SCALE.reshape(A_SCALE.shape[0], 1, 1, 1)
    B_SCALE = B_SCALE.reshape(1, B_SCALE.shape[0], 1, 1)

    CONV = CONV / A_SCALE
    CONV = CONV / B_SCALE
    return CONV


def mmt_float(A, B):
    return torch.nn.functional.conv2d(A, B)


OUT_QUANT = mmt_quant(A, B)
OUT_FLOAT = mmt_float(A, B)

OUT_DIFF = OUT_QUANT - OUT_FLOAT

print(torch.mean(torch.abs(OUT_FLOAT)).item())
print(torch.mean(torch.abs(OUT_QUANT)).item())
print(torch.mean(torch.abs(OUT_DIFF)).item())

