"""
Helpers for distributed training.
"""

import io
import os
import socket
import warnings

import blobfile as bf
import torch as th
import torch.distributed as dist

try:
    from mpi4py import MPI
except ImportError:
    from guided_diffusion.fake_mpi import MPI

    warnings.warn("Using fake MPI!")

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
SETUP_RETRY_COUNT = 3
# GPU_ID = "0"


def setup_dist(args):
    """
    Setup a distributed process group.
    """
    # global GPU_ID
    # if args.gpu_id == -1:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #     GPU_ID = ""
    # elif args.gpu_id != -2:
    #     # GPU_ID = f":{args.gpu_id}"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if dist.is_initialized():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
    print("visible devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)
    print("world size:", os.environ["WORLD_SIZE"])

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    # global GPU_ID
    if th.cuda.is_available():
        # return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2**30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    # if GPU_ID!="":
    #     rank = int(GPU_ID[-1:])
    # else:
    #     rank = 0
    # if GPU_ID == "":
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
