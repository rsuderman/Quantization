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

A = A * A_QUANT + 1.0
B = B * B_QUANT + 1.0

def quant_i8_per_tensor(A, offset=True):
    MIN = 0.0
    MAX = torch.max(A)

    if offset:
        MIN = torch.min(A)

    RANGE = MAX - MIN
    SCALE = torch.full((A.shape[0],1,1,1), 255.0 / RANGE)
    OFFSET = torch.full((A.shape[0],1,1,1), -MIN) * SCALE

    A = A * SCALE + OFFSET
    A = torch.round(A)
    A = torch.clamp(A, 0, 255.0)
    A = A.to(torch.uint8)
    SCALE = 1.0 / SCALE

    SCALE = torch.flatten(SCALE)
    OFFSET = torch.flatten(OFFSET)
    return A, SCALE, OFFSET


def quant_i8_per_channel(A,axislist, offset=True):
    MIN = torch.full((A.shape[0],1,1,1), 0.0)
    if offset:
        MIN = A
        for axis in axislist:
            MIN = torch.min(MIN, axis=axis, keepdim=True).values

    MAX = A
    for axis in axislist:
        MAX = torch.max(MAX, axis=axis, keepdim=True).values


    OFFSET = MIN
    SCALE = MAX - MIN
    SCALE = 255.0 / SCALE
    OFFSET = -OFFSET * SCALE

    print(SCALE.shape)
    print(OFFSET.shape)

    A = A * SCALE + OFFSET
    A = torch.round(A)
    A = torch.clamp(A, 0, 255.0)
    A = A.to(torch.uint8)

    OFFSET = torch.flatten(OFFSET)
    SCALE = 1.0 / torch.flatten(SCALE)
    return A, SCALE, OFFSET


def mmt_quant(A, B, per_channel=True, offset=True):
    if per_channel:
        A, A_SCALE, A_OFFSET = quant_i8_per_channel(A, axislist=[1,2,3], offset=offset)
        B, B_SCALE, B_OFFSET = quant_i8_per_channel(B, axislist=[1,2,3], offset=offset)
    else:
        A, A_SCALE, A_OFFSET = quant_i8_per_tensor(A, offset=offset)
        B, B_SCALE, B_OFFSET = quant_i8_per_tensor(B, offset=offset)


    A = A.to(torch.float)
    B = B.to(torch.float)
    CONV = torch.nn.functional.conv2d(A, B)

    A_FIX = torch.sum(torch.flatten(B, 1), dim=1).unsqueeze(0) * A_OFFSET.unsqueeze(1)
    A_FIX = A_FIX.unsqueeze(2).unsqueeze(3)

    B_FIX = torch.nn.functional.avg_pool2d(A, (B.shape[2], B.shape[3]), stride=1, divisor_override=1).sum(1, keepdim=True)
    B_FIX = B_FIX * B_OFFSET.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    AB_FIX = A_OFFSET.unsqueeze(1) * B_OFFSET.unsqueeze(0) * B.shape[1] * B.shape[2] * B.shape[3]
    AB_FIX = AB_FIX.unsqueeze(2).unsqueeze(3)

    CONV = CONV - A_FIX - B_FIX + AB_FIX

    A_SCALE = A_SCALE.reshape(A_SCALE.shape[0], 1, 1, 1)
    B_SCALE = B_SCALE.reshape(1, B_SCALE.shape[0], 1, 1)
    CONV = CONV * A_SCALE
    CONV = CONV * B_SCALE
    return CONV


def mmt_float(A, B):
    return torch.nn.functional.conv2d(A, B)


OUT_QUANT = mmt_quant(A, B)
OUT_FLOAT = mmt_float(A, B)

OUT_DIFF = OUT_QUANT - OUT_FLOAT

print(torch.mean(torch.abs(OUT_FLOAT)).item())
print(torch.mean(torch.abs(OUT_QUANT)).item())
print(torch.mean(torch.abs(OUT_DIFF)).item())

