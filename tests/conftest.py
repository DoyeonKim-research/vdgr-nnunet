import os


# Some sandboxed Windows shells leave platform.machine() empty. py-cpuinfo,
# imported by nnU-Net dependencies, accepts the standard architecture variable.
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
