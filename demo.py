# fsdp_clean_min.py
import os
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


# -------------------- Model --------------------
class Net(nn.Module):
    """5-layer MLP with H-wide Linear blocks."""
    def __init__(self, H: int) -> None:
        super().__init__()
        layers = [nn.Linear(H, H, bias=False) for _ in range(5)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------- Dist utils --------------------
def setup_dist() -> Tuple[int, int, torch.device]:
    """Init process group and return (rank, local_rank, device)."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, torch.device(f"cuda:{local_rank}")


def cleanup_dist() -> None:
    dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


# -------------------- Build FSDP model --------------------
def build_fsdp_model(H: int, device: torch.device) -> nn.Module:
    model = Net(H).to(device)
    # 只 wrap Linear，保持粒度清晰
    policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda m: isinstance(m, nn.Linear)) #只传入 参数 lambda_fn
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=policy,
        device_id=device,
    )


# -------------------- Train / Eval --------------------
def train_steps(model: nn.Module, steps: int, H: int, B: int, device: torch.device) -> float:
    opt = Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    model.train()
    last_loss = 0.0
    for _ in range(steps):
        x = torch.randn(B, H, device=device)
        y = x.detach()
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        last_loss = float(loss)
    return last_loss


@torch.no_grad()
def evaluate(model: nn.Module, H: int, B: int, device: torch.device) -> float:
    model.eval()
    out = model(torch.randn(B, H, device=device))
    return float(out.norm())


# -------------------- Main --------------------
def main() -> None:
    torch.manual_seed(0)
    rank, local_rank, device = setup_dist()

    H, B, STEPS = 1024, 64, 8  # 易改参数
    model = build_fsdp_model(H, device)

    loss = train_steps(model, STEPS, H, B, device)
    
    if is_rank0(rank):
        print(f"[train] last loss: {loss:.6f}")

    norm = evaluate(model, H, B, device)
    # 汇总一个简单指标到 rank0（平均）
    t = torch.tensor([norm], device=device)
    dist.reduce(t, dst=0, op=dist.ReduceOp.AVG)
    if is_rank0(rank):
        print(f"[eval] output L2 norm (avg over ranks): {t.item():.6f}")

    cleanup_dist()


if __name__ == "__main__":
    # 启动示例：torchrun --nproc_per_node=2 fsdp_clean_min.py
    main()
