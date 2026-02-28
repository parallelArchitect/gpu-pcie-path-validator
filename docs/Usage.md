# GPU PCIe Validator — Usage Reference

---

## Command-Line Interface

GPU PCIe Validator v4.1

Usage:
  ./gpu_pcie_validator --list-devices

  ./gpu_pcie_validator --device N
      [--memory-mode MODE]
      [--window-ms MS]
      [--interval-ms MS]
      [--size-mib MiB]

  ./gpu_pcie_validator --all-devices
      [--memory-mode MODE]
      [--window-ms MS]
      [--interval-ms MS]
      [--size-mib MiB]

Options:
  --list-devices         List all GPUs with index, BDF, NUMA, PCIe link, name
  --device N             CUDA device index to validate (default: 0)
  --all-devices          Validate all GPUs sequentially
  --memory-mode MODE     pinned (default) or unpinned
  --window-ms MS         NVML sampling window in ms (default: 2000)
  --interval-ms MS       NVML poll interval in ms   (default: 100)
  --size-mib MiB         Transfer buffer size in MiB (default: 1024)

Exit codes:
  0  All GPUs HEALTHY
  1  Runtime error (NVML/CUDA failure, bad arguments)
  2  One or more GPUs DEGRADED or LINK_DEGRADED

---

## 1. Baseline Validation

Establish a known-good reference before any deployment or after hardware changes.

```bash
./gpu_pcie_validator --device 0 --size-mib 1024 --window-ms 2000 --interval-ms 100
```

Tests:

* Sustained payload throughput
* Negotiated Gen/width correctness
* Replay counter integrity
* AER error state

Healthy baseline indicators:

* Replay counter increase: NONE
* AER correctable delta: 0
* Link consistency: TRUE
* Efficiency ratio at or above platform typical for that link

---

## 2. Long-Duration Stability Test

Detects thermal or signal degradation over time.

```bash
./gpu_pcie_validator --device 0 --size-mib 1024 --window-ms 60000 --interval-ms 100
```

Watch for:

* Replay counter increase
* Efficiency drift
* Link speed or width changes
* AER correctable error accumulation

---

## 3. Sustained Load Stress Test

Pushes DMA engines and link stability under heavy transfer load.

```bash
./gpu_pcie_validator --device 0 --size-mib 4096 --window-ms 60000 --interval-ms 50
```

Watch for:

* Speed Change: YES
* Width Change: YES
* Efficiency degradation relative to baseline
* AER non-fatal or fatal errors

---

## 4. Small Transfer Latency Test

Evaluates control-path efficiency and PCIe transaction overhead.

```bash
./gpu_pcie_validator --device 0 --size-mib 64 --window-ms 10000 --interval-ms 50
```

Healthy systems:

* 64 KiB H2D memcpy: single-digit microseconds
* 64 KiB D2H memcpy: single-digit microseconds

Elevated latency relative to a known-good baseline suggests link congestion, ASPM interference, or scheduling overhead.

---

## 5. Memory Mode Comparison

Detects host memory configuration inefficiencies.

```bash
# Pinned — page-locked host memory
./gpu_pcie_validator --device 0 --memory-mode pinned --size-mib 1024

# Unpinned — pageable host memory
./gpu_pcie_validator --device 0 --memory-mode unpinned --size-mib 1024
```

Both `unpinned` and `pageable` are accepted (`pageable` is treated as an alias).

On systems with IOMMU remapping active, pinned memory typically outperforms unpinned.
On systems with IOMMU disabled or in passthrough mode, both modes may perform similarly.

---

## 6. Multi-GPU Sweep

Validate all detected GPUs in one command:

```bash
./gpu_pcie_validator --all-devices --memory-mode pinned --window-ms 2000
echo "Fleet status: $?"
```

Output structure:

```
./logs/runs/<timestamp>_ALL/
    report.txt
    report.json
    gpu0.json
    gpu1.json
    ...
```

Exit code 0 = all HEALTHY
Exit code 2 = one or more DEGRADED or LINK_DEGRADED

Summary across GPUs:

```bash
for j in logs/runs/*_ALL/gpu*.json; do
  jq -r '[.gpu_model, .pcie_negotiated_post, .bulk_avg_gbs,
          .efficiency_ratio, .assessment_state] | @tsv' "$j"
done | column -t
```

---

## 7. High-Resolution Telemetry Capture

Detects burst instability and jitter in NVML throughput counters.

```bash
./gpu_pcie_validator --device 0 --size-mib 1024 --window-ms 30000 --interval-ms 10
```

Useful when intermittent drops are suspected but not reproducible at standard intervals.

---

## 8. AER Output Interpretation

AER (PCIe Advanced Error Reporting) counters are captured as deltas.

Correctable Errors (total delta): 0
Non-Fatal Errors (total delta): 0
Fatal Errors (total delta): 0

Interpretation:

* Correctable = 0 → Normal
* Correctable > 0 → Monitor across runs
* Non-Fatal > 0 → Investigate slot, cable, riser
* Fatal > 0 → Replace hardware

---

## 9. Clock and P-State Interpretation

Clocks are captured before and after the transfer window.

* Pre-load P8, post-load P2 → Normal boost behavior
* Pre-load P0, post-load P0 → Persistence mode or active workload
* Post-load remains P8+ → GPU failed to boost
* Memory clock drops → Thermal throttle

Correlate against power and thermal deltas in the report.

---

## 10. System Signals Reference

### Max Payload Size / Max Read Request Size

Values sourced from PCIe config space.

Suppress lspci sudo prompt:

```bash
echo "$(whoami) ALL=(ALL) NOPASSWD: /usr/bin/lspci" \
  | sudo tee /etc/sudoers.d/pcie-validator
```

Typical MPS:

* 128–256 bytes → consumer/workstation
* 512–4096 bytes → server platforms

### ASPM Policy

`auto` means runtime power management may reduce link power state.

Disable for benchmarking:

```bash
sudo bash -c 'echo on > /sys/bus/pci/devices/<bdf>/power/control'
```

### IOMMU

`no groups` → disabled or passthrough (normal for GPU workloads)

If strict mode is enabled and throughput is low, translation overhead may contribute.

---

## Interpretation Signals

| Signal                  | Healthy          | Investigate                          |
| ----------------------- | ---------------- | ------------------------------------ |
| Link consistency        | TRUE             | FALSE                                |
| Replay counter increase | NONE             | > 0                                  |
| AER correctable delta   | 0                | > 0 across runs                      |
| AER fatal delta         | 0                | > 0                                  |
| Efficiency ratio        | Platform typical | Sustained drop vs baseline           |
| Post-load P-state       | P0–P3            | P8+                                  |
| Thermal delta           | Stable           | Sustained rise with clock throttling |

