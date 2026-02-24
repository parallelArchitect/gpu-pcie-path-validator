# gpu-pcie-path-validator

![License](https://img.shields.io/badge/license-MIT-blue)
![CUDA](https://img.shields.io/badge/CUDA-supported-green)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![NVML](https://img.shields.io/badge/NVML-supported-informational)
![PCIe](https://img.shields.io/badge/PCIe-transport-critical)

Deterministic PCIe transport validation for NVIDIA GPUs.

This tool collects live transport and telemetry signals from the GPU using NVIDIA-supported interfaces (CUDA, NVML, sysfs) and computes all metrics directly from the negotiated PCIe link state.

---

## What It Measures

* CUDA memcpy throughput — H2D and D2H, bulk transfer and link latency
* NVML PCIe bus-level throughput — RX, TX, combined
* Link negotiation state — pre and post load, speed and width
* PCIe replay counter delta
* AER error counters — correctable, non-fatal, fatal deltas
* GPU clock state — SM, memory, graphics clocks, P-state pre and post load
* Max Payload Size and Max Read Request Size
* Persistence mode, ASPM policy, IOMMU state
* NUMA affinity validation
* Power and thermal telemetry — pre, average, peak, end, delta

All values are sourced from live hardware via NVML and sysfs. Nothing is estimated.

---

## Requirements

* Linux
* NVIDIA GPU
* NVIDIA Driver (with NVML support)
* CUDA Toolkit
* `pciutils`

---

## Install

Clone and build:

```bash
git clone https://github.com/parallelArchitect/gpu-pcie-path-validator.git
cd gpu-pcie-path-validator
make
```

Optional system-wide install (if supported in Makefile):

```bash
sudo make install
```

---

## Quick Start

> Note: On some systems, a sudo prompt may appear if `lspci` fallback is required to read PCIe configuration space (MPS/MRRS).

List all detected GPUs:

```bash
./gpu_pcie_validator --list-devices
```

Validate a single device:

```bash
./gpu_pcie_validator --device 0
```

Validate all devices:

```bash
./gpu_pcie_validator --all-devices
```

---

## Output

Single device:

```
./logs/runs/<timestamp>_GPU<N>/
    report.txt
    report.json
```

All devices:

```
./logs/runs/<timestamp>_ALL/
    report.txt
    report.json
    gpu0.json
    gpu1.json
    ...
```

---

## Exit Codes

```
0 — All validated GPUs HEALTHY
1 — Runtime error
2 — One or more GPUs DEGRADED or LINK_DEGRADED
```

---

## Usage

See: [docs/USAGE.md](docs/USAGE.md)

---

## Interpretation

CUDA memcpy throughput measures payload transfer rate per direction for a defined buffer size.

NVML PCIe throughput (`nvmlDeviceGetPcieThroughput`) reports bus-level traffic counters sampled over a measurement window.

These metrics may differ due to sampling window alignment, DMA overlap, or concurrent PCIe activity.

NVML API reference:
[https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)

NVIDIA Developer:
[https://developer.nvidia.com/](https://developer.nvidia.com/)

---

## Scope

Does not:

* Benchmark kernel compute performance
* Diagnose motherboard electrical faults
* Replace full-system hardware diagnostics

---

## Author

Joe McLaren (parallelArchitect)
Human-directed GPU engineering with AI assistance.

---

## License

MIT






