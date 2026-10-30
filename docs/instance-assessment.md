# Xing-Specialist instance assessment

Observed 2026-10-29 UTC. Read-only inspection; no configuration changes or model downloads.

## Hardware

| Item | Observation |
| --- | --- |
| OS | Amazon Linux 2023.12.20261028, kernel 6.18.56-125.164.amzn2023.x86_64; **not Ubuntu** |
| CPU | 2 logical CPUs, 1 core / 2 threads, Intel Xeon Platinum 8488C exposed by KVM; AVX2, AVX-512 and BF16 flags visible |
| GPU | No `nvidia-smi`, `/dev/nvidia*`, or `/dev/dri` visible; treat as CPU-only |
| RAM | 7,966,544 kB = 7.60 GiB physical |

CPU flags are exposed to the guest; sustained frequency and actual inference throughput have not been measured.

## Disk

| Mount | Type | Size | Used | Available | Implication |
| --- | --- | ---: | ---: | ---: | --- |
| `/` | XFS on `/dev/nvme0n1p1` | 100 GiB | 3.6 GiB | 97 GiB | Capacity for a single quantized checkpoint and later swap, subject to headroom checks |
| `/tmp` | tmpfs | 3.8 GiB | 22 MiB | 3.8 GiB | Current writable workspace; unsuitable for weights and not durable across reboot |

The 100 GiB NVMe volume has one root partition plus EFI metadata. Current writable access is limited to `/tmp`; a durable project location on the root filesystem must be arranged before implementation. Avoid putting downloads, caches, or swap in `/tmp`. Even after a 20 GiB swap file and an 11.8 GB model, leave generous free space for the OS, cache, logs, and future artifacts. Recheck capacity before any download.

## Memory and swap

At inspection: `free -h` reported 7.6 GiB total, 442 MiB used, 6.9 GiB available. `/proc/meminfo` reported 7,252,320 kB available at a nearby instant. No swap is active (`swapon --show` empty, `SwapTotal=0`). Current `vm.swappiness=60`; requested target is 100. No 20 GiB swap file was created. These targets require a later reviewed system change. Swap is overflow storage, not RAM-speed capacity. With 2 vCPUs and weights larger than RAM, frequent weight page faults could make generation impractically slow; this remains unmeasured.

## Dependencies and compilers

Available: Python 3.9.25, curl 8.21.0, wget. Not installed or not on PATH: `git`, `gcc`, `g++`, `clang`, `cmake`, `ninja`, `make`, `pip`, `docker`, `nvidia-smi`. RPM checks corroborate missing compiler/build packages and pip. A CPU `llama.cpp` build would need a toolchain, CMake, and Git or an equivalent source acquisition path. Package names and versions should be pinned after the OS and durable location decision.

## Evidence and limits

Commands: `uname -a`, `cat /etc/os-release`, `lscpu`, `free -h`, `cat /proc/meminfo`, `swapon --show --bytes`, `cat /proc/sys/vm/swappiness`, `df -hT`, `lsblk`, `findmnt -T /tmp`, `rpm -q`, `command -v`. A concise raw snapshot is in `artifacts/machine-snapshot.txt`. No stress test, benchmark, installation, or reboot test was run.
