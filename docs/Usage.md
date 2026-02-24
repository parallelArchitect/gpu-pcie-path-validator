# GPU PCIe Validator — Usage Reference

---

## 1. Baseline Validation (Mandatory)

Establish a known-good reference before any deployment or after hardware changes.

```bash
./gpu_pcie_validator --device 0 --size-mib 1024 --window-ms 2000 --interval-ms 100
```

**Tests:**

* Sustained payload throughput
* Negotiated Gen/width correctness
* Replay counter integrity
* AER error state

**Healthy baseline indicators:**

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

**Watch for:**

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

**Watch for:**

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

**Healthy systems:**

* 64 KiB H2D memcpy: single-digit microseconds
* 64 KiB D2H memcpy: single-digit microseconds

Elevated latency relative to a known-good baseline suggests link congestion, ASPM interference, or scheduling overhead.

---

## 5. Memory Mode Comparison

Detects host memory configuration inefficiencies.

```bash
# Pinned — page-locked host memory (cudaHostAlloc). Reference / certification mode.
./gpu_pcie_validator --device 0 --memory-mode pinned --size-mib 1024

# Unpinned — pageable host memory (standard malloc).
# CUDA internally stages pageable memory into pinned buffers before DMA.
# Impact depends on IOMMU configuration — see note below.
./gpu_pcie_validator --device 0 --memory-mode unpinned --size-mib 1024
```

Both `unpinned` and `pageable` are accepted (`pageable` is treated as an alias).

On systems with IOMMU remapping active, pinned memory will outperform unpinned.
On systems with IOMMU disabled or in passthrough mode, both modes may perform identically because the driver uses the same DMA path. The tool reports IOMMU state in System Signals.

---

## 6. Multi-GPU Sweep

Validate all detected GPUs in one command:

```bash
./gpu_pcie_validator --all-devices --memory-mode pinned --window-ms 2000
echo "Fleet status: $?"
```

**Output:**

```
./logs/runs/<timestamp>_ALL/
    report.txt       — all GPUs combined
    report.json      — JSON array of all GPU objects
    gpu0.json
    gpu1.json
    ...
```

Exit code 0 = all HEALTHY. Exit code 2 = one or more DEGRADED or LINK_DEGRADED.

**Summary across all GPUs:**

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

AER (PCIe Advanced Error Reporting) counters are captured as deltas (post-load minus pre-load).

```
Correctable Errors (total delta)   : 0
Non-Fatal Errors (total delta)     : 0
Fatal Errors (total delta)         : 0
```

| Counter         | Meaning                                               | Action                         |
| --------------- | ----------------------------------------------------- | ------------------------------ |
| Correctable = 0 | Normal                                                | None                           |
| Correctable > 0 | Receiver errors, bad DLLP/TLP, replayed and recovered | Monitor across runs            |
| Non-Fatal > 0   | Link degraded but functional                          | Investigate slot, cable, riser |
| Fatal > 0       | Link integrity compromised                            | Replace hardware               |

AER availability depends on kernel version and platform support. If `aer_available: false` in JSON, the kernel does not expose AER sysfs for this slot — not a fault.

---

## 9. Clock and P-State Interpretation

Clocks are captured before and after the transfer window.

| Observation                  | Meaning                                                         |
| ---------------------------- | --------------------------------------------------------------- |
| Pre-load P8, post-load P2    | Normal — GPU was idle, boosted correctly                        |
| Pre-load P0, post-load P0    | Persistence mode enabled or GPU was already active              |
| Post-load P8 or higher       | GPU failed to boost — check TDP, power limits, thermal throttle |
| Memory clock drops post-load | Thermal throttle on memory — check cooling                      |

A post-load SM clock below the GPU's rated boost frequency indicates throttling. Correlate against power delta and thermal delta in the same report.

---

## 10. System Signals Reference

### Max Payload Size / Max Read Request Size

```
Max Payload Size   : <value from PCIe config space>
Max Read Request   : <value from PCIe config space>
```

These values come from PCIe config space (sysfs first, lspci fallback). A sudo prompt may appear on first run if sysfs is unavailable — to suppress permanently:

```bash
echo "$(whoami) ALL=(ALL) NOPASSWD: /usr/bin/lspci" \
  | sudo tee /etc/sudoers.d/pcie-validator
```

| MPS            | Context                        |
| -------------- | ------------------------------ |
| 128–256 bytes  | Typical consumer / workstation |
| 512–4096 bytes | Enterprise / server platforms  |

Mismatched MPS between endpoint and root port can silently reduce throughput.

### ASPM Policy

```
ASPM Policy : auto (runtime PM enabled)
```

`auto` means the kernel may allow the link to enter low-power states between transfers. For benchmarking, disable with:

```bash
sudo bash -c 'echo on > /sys/bus/pci/devices/<bdf>/power/control'
```

### IOMMU

```
IOMMU : no groups (passthrough or disabled)
```

IOMMU disabled or in passthrough mode. This is normal for direct GPU workloads. If `strict` mode is active and throughput is unexpectedly low, IOMMU translation overhead may be a factor.

---

## Interpretation Signals

| Signal                  | Healthy                      | Investigate                                    |
| ----------------------- | ---------------------------- | ---------------------------------------------- |
| Link consistency        | TRUE                         | FALSE — link retrained during test             |
| Replay counter increase | NONE                         | Any value > 0                                  |
| AER correctable delta   | 0                            | > 0 across multiple runs                       |
| AER fatal delta         | 0                            | Any value > 0                                  |
| Efficiency ratio        | At or above platform typical | Sustained drop vs baseline                     |
| Post-load P-state       | P0–P3                        | P8+ (failed to boost)                          |
| Thermal delta           | Stable across window         | Sustained rise — correlate with clock throttle |




