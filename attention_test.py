import math
import numpy
import torch

folder = "/media/rsuderman/Disk2/Models/SDXL/test_inputs"

a = None

q = torch.tensor(numpy.load(f"{folder}/q.numpy"))
k = torch.tensor(numpy.load(f"{folder}/k.numpy"))
v = torch.tensor(numpy.load(f"{folder}/v.numpy"))
o = torch.tensor(numpy.load(f"{folder}/o.numpy"))

fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max

def truncate_f8(f):
    scale = torch.max(torch.abs(f)) / fp8_max

    f = f / scale
    f_8 = f.to(fp8_dtype)
    f_o = f_8.to(f.dtype)
    f_o = f_o * scale
    return f_o


q_fp8 = truncate_f8(q)
k_fp8 = truncate_f8(k)
v_fp8 = truncate_f8(v)
o_fp8 = truncate_f8(o)

def builtin_attention(q,k,v, a):
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=a, dropout_p=0.0, is_causal=False)
    return o

def decomposed_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def flash_attention(query, key, value, attention):
    BLOCK_M = 64
    BLOCK_K = 64
    seq_len = query.shape[-2]
    HIDDEN = query.shape[-1]

    init = torch.ones(*query.shape[:-1], value.shape[-1])

    for b in range(query.shape[0]):
        for h in range(query.shape[1]):
            for i in range(seq_len // BLOCK_M):

                acc = torch.zeros(BLOCK_M, value.shape[-1])
                start = i * BLOCK_M
                end = start + BLOCK_M
                q = query[b, h, start:end, :].to(torch.float32)

                max_stat = torch.full((BLOCK_M,), fill_value=-1e9, dtype=torch.float32)
                sum_stat = torch.full((BLOCK_M,), fill_value=0, dtype=torch.float32)

                for j in range(seq_len // BLOCK_K):
                    start = j * BLOCK_K
                    end = start + BLOCK_K

                    k = key[b, h, start:end, :].to(torch.float32)
                    v = value[b, h, start:end, :].to(torch.float32)

                    kT = torch.transpose(k, 0, 1)

                    # Scaling parameter
                    qkT = torch.matmul(q, kT) 
                    qkT = qkT / math.sqrt(float(HIDDEN))

                    old_max = max_stat
                    old_sum = sum_stat

                    new_max = torch.maximum(torch.max(qkT, dim=1).values, old_max)
                    broadcasted_max = new_max.unsqueeze(1)

                    scale_factor = torch.exp(old_max - new_max)
                    scaled_old_sum = scale_factor * old_sum
                    broadcasted_scale_factor = scale_factor.unsqueeze(1)
                    acc = acc * broadcasted_scale_factor

                    partial_softmax = torch.exp(qkT - broadcasted_max)
                    new_sum = torch.sum(partial_softmax, dim=1) + scaled_old_sum

                    acc = torch.matmul(partial_softmax, v) + acc
                    sum_stat = new_sum
                    max_stat = new_max

                acc = acc / sum_stat.unsqueeze(1)

                start = i * BLOCK_K
                end = start + BLOCK_K
                init[b, h, start:end, :] = acc

    return init


def compute_error(lhs, rhs):
    lhs = lhs.to(torch.float32)
    rhs = rhs.to(torch.float32)
    diff = lhs - rhs

    diff2 = diff * diff
    mxerr = torch.max(torch.abs(diff))
    sserr = torch.sqrt(torch.sum(diff2) / torch.numel(diff2))

    return mxerr.item(), sserr.item()


def evaluate(f, q, k, v, a, o):
    res = f(q, k, v, a)
    print(res[0, 0, :4, :4])
    mx, serr = compute_error(res, o)
    range = (torch.min(res).item(), torch.max(res).item())
    print(f.__name__, q.dtype)
    print(" max err ", mx)
    print(" sq err  ", serr)
    print(" range   ", range)

q = q[:1, :1, :, :]
k = k[:1, :1, :, :]
v = v[:1, :1, :, :]
o = o[:1, :1, :, :]

print(o[0, 0, :4, :4])

evaluate(builtin_attention, q, k, v, a, o)
# evaluate(builtin_attention, q_fp8, k_fp8, v_fp8, a, o_fp8)
evaluate(flash_attention, q, k, v, a, o)
