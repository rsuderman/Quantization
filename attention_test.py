import functools
import math
import numpy
import torch

import iree.runtime as ireert

folder = "data"

a = None

q = torch.tensor(numpy.load(f"{folder}/q.numpy"))
k = torch.tensor(numpy.load(f"{folder}/k.numpy"))
v = torch.tensor(numpy.load(f"{folder}/v.numpy"))
o = torch.tensor(numpy.load(f"{folder}/o.numpy"))

q = q[:1, 0, :, :]
k = k[:1, 0, :, :]
v = v[:1, 0, :, :]
o = o[:1, 0, :, :]

fp8_dtype = torch.float8_e4m3fnuz
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

def quantize_fp8(tensor, scale=None):
    if scale is None:
        scale = torch.max(torch.abs(tensor)).item() / fp8_max
    tensor = tensor / scale
    tensor = torch.clamp(tensor, -fp8_max, fp8_max)
    tensor = tensor.to(fp8_dtype)
    return scale, tensor.to(torch.float32)

def save_fp8(tensor, tname, sname):
    scale, tensor = quantize_fp8(tensor)
    scale = numpy.asarray(scale, dtype=numpy.single)
    tensor = numpy.asarray(tensor, dtype=numpy.single)
    numpy.save(tname, tensor)
    numpy.save(sname, scale)

if True:
    save_fp8(q, "data/fp8/q", "data/fp8/qscale")
    save_fp8(k, "data/fp8/k", "data/fp8/kscale")
    save_fp8(v, "data/fp8/v", "data/fp8/vscale")
    scale = numpy.asarray(1.0 / math.sqrt(64), dtype=numpy.single)
    numpy.save("data/fp8/scale", scale)
    numpy.save("data/fp8/o", o)

def builtin_attention(q, k, v, a=None):
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

def iree_flash_attention(query, key, value, fake_fp8=False):
    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)

    vmfb = "attention_fp32.vmfb" if fake_fp8 else "attention_f32.vmfb"

    with open(vmfb, 'rb') as f:
        contents = f.read()
    vm_module = ireert.VmModule.from_buffer(ctx.instance, contents, warn_if_copy=False)
    ctx.add_vm_module(vm_module)
    main = ctx.modules.module["main"]

    batchdims = query.shape[:-2]
    batch = functools.reduce(lambda x, y : x * y, batchdims, 1)

    query = query.reshape((batch, query.shape[-2], query.shape[-1])).to(torch.float32)
    key = key.reshape((batch, key.shape[-2], key.shape[-1])).to(torch.float32)
    value = value.reshape((batch, value.shape[-2], value.shape[-1])).to(torch.float32)

    if fake_fp8:
        qscale, query = quantize_fp8(query)
        vscale, value = quantize_fp8(value)
        kscale, key = quantize_fp8(key)

        qscale = torch.asarray(qscale).to(torch.float32)
        vscale = torch.asarray(vscale).to(torch.float32)
        kscale = torch.asarray(kscale).to(torch.float32)

    scale = 1.0 / math.sqrt(64)
    scale = torch.asarray(scale, dtype=torch.float32)

    if fake_fp8:
        output = main(query, key, value, scale, qscale, kscale, vscale)
    else :
        output = main(query, key, value, scale)

    output = output.reshape((batch, output.shape[-2], output.shape[-1]))
    output = torch.tensor(output)
    return output
    
def flash_attention(query, key, value, fp8=False):
    BLOCK_M = 64
    BLOCK_K = 64
    seq_len = query.shape[-2]
    HIDDEN = query.shape[-1]

    if fp8:
        query_scale, query = quantize_fp8(query)
        key_scale, key = quantize_fp8(key)
        value_scale, value = quantize_fp8(value)

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

                    if fp8:
                        qkT = qkT * query_scale
                        qkT = qkT * key_scale

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

                    if fp8:
                        partial_softmax_scale, partial_softmax = quantize_fp8(partial_softmax)
                        acc = torch.matmul(partial_softmax, v) * partial_softmax_scale + acc
                    else:
                        acc = torch.matmul(partial_softmax, v) + acc

                    sum_stat = new_sum
                    max_stat = new_max

                acc = acc / sum_stat.unsqueeze(1)
                
                if fp8:
                    acc = acc * value_scale

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


def evaluate(f, o, *args, **kwargs):
    res = f(*args, **kwargs)
    mx, serr = compute_error(res, o)
    range = (torch.min(res).item(), torch.max(res).item())
    print(f.__name__, kwargs)
    print(" max err ", mx)
    print(" sq err  ", serr)
    print(" range   ", range)

# evaluate(builtin_attention, o, q, k, v)
# evaluate(flash_attention, o, q, k, v)
# evaluate(flash_attention, o, q, k, v, fp8=True)
# evaluate(iree_flash_attention, o, q, k, v)
# evaluate(iree_flash_attention, o, q, k, v, fake_fp8=True)
