"""FreeToken compatibility hooks for WSL hosts with a CUDA host-pin ceiling.

Enabled only when the broker wrapper sets FREETOKEN_BROKER_PAGEABLE_MOE=1.
The model's expert GEMMs remain on CUDA; selected host rows are staged through
ordinary pageable host-to-device copies instead of requiring the whole FTW bank
set to be cudaHostRegister'd.
"""

from __future__ import annotations

import os
import sys


if os.environ.get("FREETOKEN_BROKER_PAGEABLE_MOE") == "1":
    from freetoken.moe.host_banks import HostBank
    from freetoken.moe.offload_cache import OffloadMoeCache

    def _pageable_pin(_self: HostBank) -> None:
        # HostBank still owns the mmap and remains resident; only CUDA registration
        # is skipped. The patched cache copy path below stages rows explicitly.
        return None

    def _pageable_copy_missing(self: OffloadMoeCache) -> None:
        """Stage routed rows through pageable CPU memory, then copy to GPU slots."""
        assert self.banks, "set_bank_sources must register the banks first"
        layer_id = self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"

        count = int(self.num_indices.detach().cpu().reshape(-1)[0].item())
        if count <= 0:
            return
        source_ids = self.src_indices.detach().cpu().reshape(-1)[:count].long()
        slot_ids = self.evict_slots.detach().cpu().reshape(-1)[:count].long()
        slot_ids_gpu = slot_ids.to(device=self.device)

        for per_layer, cache in self.banks:
            rows = per_layer[layer_id].index_select(0, source_ids)
            # non_blocking=False makes PyTorch stage pageable memory safely before
            # the GPU expert GEMM consumes the slot cache.
            cache[slot_ids_gpu] = rows.to(device=self.device, non_blocking=False)

    HostBank.pin = _pageable_pin
    OffloadMoeCache.copy_missing = _pageable_copy_missing
    os.environ.setdefault("FREETOKEN_FUSED_COPY", "0")
    print(
        "freetoken: broker pageable-MoE compatibility enabled; GPU GEMMs retained",
        file=sys.stderr,
        flush=True,
    )
