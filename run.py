
import numpy
import subprocess

RUN_MODULE = "/home/rsuderman/Repos/iree/build/tools/iree-run-module"
VMFB = "/home/rsuderman/Work/Quantization/vmfb/attention_fp32.vmfb"
FP8_DIR = "/home/rsuderman/Work/Quantization/data/fp8"

OUTPUT = "/tmp/o.npy"

RUN_COMMAND = [
    RUN_MODULE,
    f"--module={VMFB}",
    f"--input=@{FP8_DIR}/q.npy",
    f"--input=@{FP8_DIR}/k.npy",
    f"--input=@{FP8_DIR}/v.npy",
    f"--input=@{FP8_DIR}/scale.npy",
    f"--input=@{FP8_DIR}/qscale.npy",
    f"--input=@{FP8_DIR}/kscale.npy",
    f"--input=@{FP8_DIR}/vscale.npy",
    f"--output=@{OUTPUT}",
]

subprocess.run(RUN_COMMAND)

expected = numpy.load(f"{FP8_DIR}/o.npy")
actual = numpy.load(OUTPUT)

diff = numpy.abs(actual - expected)
sqerr = diff * diff

print(numpy.max(diff))
print(numpy.sqrt(numpy.sum(sqerr) / sqerr.size))



