#!/usr/bin/env bash
set -euo pipefail

apply=0
set_swappiness=0
for arg in "$@"; do
    case "$arg" in
        --apply) apply=1 ;;
        --set-swappiness) set_swappiness=1 ;;
        *) echo "Usage: $0 [--apply] [--set-swappiness]" >&2; exit 2 ;;
    esac
done

target_bytes=$((20 * 1024 * 1024 * 1024))
minimum_bytes=$((18 * 1024 * 1024 * 1024))
reserve_bytes=$((30 * 1024 * 1024 * 1024))
swap_path=/swapfile
sysctl_path=/etc/sysctl.d/99-xing-swap.conf

if (( apply )) && (( EUID != 0 )); then
    echo "Run with sudo for --apply" >&2
    exit 1
fi

active_bytes=$(awk 'NR > 1 {sum += $3 * 1024} END {printf "%.0f", sum}' /proc/swaps)
fs_type=$(findmnt -n -o FSTYPE -T /)
available_bytes=$(df -B1 --output=avail / | tail -n 1 | tr -d ' ')
current_swappiness=$(cat /proc/sys/vm/swappiness)
echo "Active swap: $active_bytes bytes; root filesystem: $fs_type; root free: $available_bytes bytes; swappiness: $current_swappiness"

if (( active_bytes >= minimum_bytes )); then
    echo "Existing swap meets the approximately 20 GiB target; no swap file change needed."
elif (( active_bytes > 0 )); then
    echo "Existing swap is below target. Review it manually; refusing to add or replace swap." >&2
    exit 1
elif [[ -e "$swap_path" ]]; then
    echo "$swap_path exists but is inactive. Review it manually; refusing to overwrite it." >&2
    exit 1
else
    if [[ "$fs_type" != xfs && "$fs_type" != ext4 ]]; then
        echo "Unsupported root filesystem for this procedure: $fs_type" >&2
        exit 1
    fi
    if (( available_bytes < target_bytes + reserve_bytes )); then
        echo "Insufficient space: need 20 GiB swap plus 30 GiB free reserve." >&2
        exit 1
    fi
    if grep -Eq '^[[:space:]]*/swapfile[[:space:]]' /etc/fstab; then
        echo "An fstab entry already references /swapfile; review it manually." >&2
        exit 1
    fi
    echo "Would create one 20 GiB $swap_path on $fs_type and add a nofail fstab entry."
    if (( apply )); then
        umask 077
        fallocate -l 20G "$swap_path"
        chmod 600 "$swap_path"
        mkswap "$swap_path"
        swapon "$swap_path"
        printf '%s\n' '/swapfile none swap defaults,nofail 0 0' >> /etc/fstab
    fi
fi

if (( set_swappiness )); then
    echo "Requested persistent vm.swappiness=100."
    if (( apply )); then
        printf '%s\n' 'vm.swappiness = 100' > "$sysctl_path"
        sysctl -w vm.swappiness=100
    fi
fi

if (( apply )); then
    swapon --show --bytes
    sysctl vm.swappiness
    df -h /
fi
