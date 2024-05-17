# Inspired by https://gist.github.com/s-mawjee/ad0d8e0c7e07265cae097899fe48c023


import nvidia_smi

from robust_llm import logger


def _check_gpu():
    nvidia_smi.nvmlInit()
    num_gpu = nvidia_smi.nvmlDeviceGetCount()
    gpu = num_gpu > 0

    return gpu, num_gpu


def _bytes_to_megabytes(bytes):
    return round((bytes / 1024) / 1024, 2)


def print_gpu_usage():
    gpu, num_gpu = _check_gpu()
    if not gpu:
        logger.info("No GPU found, so no usage to report.")
        return

    for i in range(num_gpu):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        logger.info(
            "GPU-%s: GPU-Memory: %s/%s MB",
            i,
            _bytes_to_megabytes(info.used),
            _bytes_to_megabytes(info.total),
        )
