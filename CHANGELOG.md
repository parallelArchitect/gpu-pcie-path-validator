# GPU PCIe Validator — Engineering Changelog

---

## v4.1 — Progress Bar + NVML Smoothing

Date: 2026-02-27
Author: Joe McLaren (parallelArchitect)

Files Modified:

* gpu_pcie_validator.cu
* README.md
* docs/Usage.md

Makefile: Unchanged

---

## Changes

* Added single-line progress bar for NVML sampling window.
* Displays elapsed time, rolling-average RX/TX throughput, power, and temperature.
* Added rolling-average NVML throughput (10-sample window) to live progress display.
* Updated README.md and docs/Usage.md to reflect v4.1 behavior.

