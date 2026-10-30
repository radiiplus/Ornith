# Stage 1 environment and memory baseline

Observed 2026-10-29 UTC. Baseline machine-readable counters: `artifacts/stage1-baseline.json`. Stage 1 did not download any model or install packages.

## Host and capacity

This is **Amazon Linux 2023.12.20261028**, not the requested Ubuntu image. Kernel: `6.18.56-125.164.amzn2023.x86_64`. CPU: KVM-exposed Intel Xeon Platinum 8488C, x86-64, one physical core / two threads (two vCPUs); visible instructions include AVX2, AVX-512, AVX-512 BF16, and AMX. No GPU device node or `nvidia-smi` was found. Physical RAM is 7,966,544 kB (7.60 GiB), with about 7.0 GiB available at baseline. The 100 GiB XFS root volume has about 77 GiB free after the existing swap file; `/boot/efi` is a small vfat partition. `/tmp` is a 3.8 GiB tmpfs and is not used for the project or weights.

The existing `/swapfile` is root-owned, mode 0600, 20 GiB allocated, and active as 20,971,516 KiB usable swap. Baseline usage is zero. `vm.swappiness` is **200**, persistently specified by `/etc/sysctl.d/99-xing-swap.conf`; target for this stage is 100. The user selected **document only**, so the live and persistent values were left at 200. `/etc/fstab` references `/swapfile` with `defaults`; it lacks `nofail`. A missing swap file at boot has not been simulated, so recovery behavior of the current entry is not verified. The new-file procedure uses `nofail` after successful `swapon`.

No C/C++ compiler, CMake, Ninja, Make, Git, Rust toolchain, or pip was found. Python is 3.9.25. Active services include SSH, SSM agent, systemd network/resolved, journald, auditd, chronyd, and login services. The largest observed RSS processes were Codex sessions (about 293, 197, and 134 MiB); these are not model workloads. Memory PSI `some` and `full` averages were 0.00 at baseline. `/proc/vmstat` showed no swap in/out and zero OOM kills; these are cumulative counters, not a stress test.

## Budget and behavior

`configs/memory-budget.json` reserves 2.0 GiB for the OS and services and 1.0 GiB for runtime/KV/temporary tensors, leaving an initial **4.5 GiB model-resident budget**. This is a conservative planning cap, not a claim that model weights fit. Disk preflight requires 30 GiB free *after* any new 20 GiB swap allocation. Recompute both budgets before weight acquisition and measure peak RSS, swap I/O, PSI and major faults under a representative load.

Anonymous memory includes process heaps, stacks, and many temporary tensors; reclaim generally requires swap or process termination. Memory-mapped weights are file-backed: pages enter RAM as accessed, share the page cache, and can be reclaimed and reread from the model file. Mmap reduces eager copying but does not eliminate RAM use, I/O, or page faults. Clean file-backed pages usually can be dropped; dirty anonymous pages need swap. Swap-out writes pages to SSD, and swap-in reads them on demand. Under repeated MoE expert access, a small RAM budget may thrash between model-file paging and swap I/O. This can produce severe latency, SSD write wear, and ultimately OOM if allocation or reclaim fails. High swappiness changes reclaim preference; even 200 cannot increase physical RAM or guarantee that a model runs.

## Procedure and verification

`scripts/configure-memory.sh` is dry-run by default. It reads active swap first, accepts an existing total of at least 18 GiB as meeting the approximate target, and refuses to overwrite an inactive `/swapfile` or add to a smaller existing swap configuration. If no swap exists, it permits one 20 GiB file only on XFS/ext4 after a 30 GiB free-space reserve check. It activates swap before adding `defaults,nofail` to `/etc/fstab`, so a failed activation does not add a boot dependency. A failed first run may leave an inactive file that requires manual review; a repeat run refuses to overwrite it. `--apply --set-swappiness` is the explicit path to persist and activate 100, but was **not run**. The current `/etc/fstab` entry is left unchanged.

`scripts/measure-memory.py --pid PID --output PATH` records RAM, swap, root disk, process RSS/VMS and major faults, `/proc/vmstat`, and memory PSI. Counters can be compared across runs; a single cumulative fault count is not a rate.

Checks performed: shell syntax check; repeated dry runs; dry run with `--set-swappiness`; two measurements (one saved); JSON parse; active swap and sysctl recheck; `findmnt --verify` (0 errors, 4 warnings including generic swap-file warnings). No reboot, deliberate swap failure, OOM test, or high-pressure test was performed. Recovery for the **existing** fstab entry remains an open verification item. Stage 2 may research metadata, but no model download should occur until this environment mismatch, persistent swappiness target, and capacity budget are reviewed.
