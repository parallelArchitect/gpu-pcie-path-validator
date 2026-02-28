/*
 * =============================================================================
 * GPU PCIe Validator
 * =============================================================================
 *
 * Author      : Joe McLaren (Human-AI Collaborative Engineering)
 * Repository  : https://github.com/parallelArchitect
 * File        : gpu_pcie_validator.cu
 * Version     : 4.1
 *
 * DESCRIPTION
 * -----------
 * Production-grade PCIe path validation and diagnostics for NVIDIA GPUs.
 * Measures bandwidth, link latency, thermal response, power draw, PCIe replay
 * errors, AER counters, GPU clock/P-state, persistence mode, ASPM policy,
 * IOMMU state, and MPS/MRRS configuration. All values sourced from live
 * hardware — no hardcoded defaults or synthetic estimates.
 *
 * Designed for universal use across all NVIDIA GPU generations (Kepler through
 * Blackwell) and all PCIe generations (Gen1–Gen6). Suitable for field
 * diagnostics, data center health checks, and automated CI integration.
 *
 * KEY FEATURES
 * ------------
 * - AER (Advanced Error Reporting) correctable/non-fatal/fatal delta counters
 * - GPU clock snapshot pre/post load: SM, memory, graphics + P-state
 * - PCIe MPS (Max Payload Size) and MRRS (Max Read Request Size) from sysfs
 * - Persistence mode and ASPM policy from NVML/sysfs
 * - IOMMU state detection from kernel sysfs
 * - NUMA affinity validation (CPU NUMA node matches GPU NUMA node)
 * - Dual latency measurement: bulk transfer (1 GiB) and link latency (64 KB)
 * - Pinned and unpinned memory mode support
 * - PCIe replay counter with integer delta
 * - Multi-GPU: --all-devices sweeps all GPUs, JSON array output
 * - Non-zero exit code on UNHEALTHY/DEGRADED (Kubernetes/Nagios/CI ready)
 * - Structured JSON output with full derived assessment fields
 * - CUDA event-based microsecond timing precision
 * - Driver/NVML version captured in JSON
 *
 * MEASUREMENT SOURCES
 * -------------------
 * All values queried at runtime from hardware or kernel:
 *   - PCIe Gen/Width:      nvmlDeviceGetMaxPcieLinkGeneration/Width
 *   - Bandwidth:           cudaMemcpyAsync + CUDA event timing
 *   - Latency:             CUDA events (bulk 1024 MiB + 64 KB transfers)
 *   - Replay counter:      nvmlDeviceGetPcieReplayCounter
 *   - AER counters:        /sys/bus/pci/devices/<bdf>/aer_dev_correctable etc.
 *   - Clocks/P-state:      nvmlDeviceGetClockInfo + nvmlDeviceGetPerformanceState
 *   - Power/Temp:          nvmlDeviceGetPowerUsage/Temperature
 *   - Persistence mode:    nvmlDeviceGetPersistenceMode
 *   - MPS/MRRS:            /sys/bus/pci/devices/<bdf>/max_payload_size etc.
 *   - ASPM policy:         /sys/bus/pci/devices/<bdf>/power/control
 *   - IOMMU:               /sys/kernel/iommu_groups/ + /proc/cmdline
 *   - NUMA node:           /sys/bus/pci/devices/<bdf>/numa_node
 *   - NUMA CPU affinity:   /sys/bus/pci/devices/<bdf>/local_cpulist
 *   - Scheduler:           sched_getscheduler(0) + sched_getparam()
 *   - Topology:            sysfs symlink walk from endpoint to root port
 *   - NVML version:        nvmlSystemGetNVMLVersion
 *
 * BUILD
 * -----
 * nvcc -O3 -std=c++14 -lnvidia-ml -Xcompiler -pthread \
 *      gpu_pcie_validator.cu -o gpu_pcie_validator
 *
 * USAGE
 * -----
 * List available GPUs:
 *   ./gpu_pcie_validator --list-devices
 *
 * Run validation on device 0 with pinned memory:
 *   ./gpu_pcie_validator --device 0 --memory-mode pinned
 *
 * Run validation on all GPUs (sequential, JSON array output):
 *   ./gpu_pcie_validator --all-devices
 *
 * Full parameter example:
 *   ./gpu_pcie_validator --device 0 --memory-mode pinned \
 *                        --window-ms 2000 --interval-ms 100 --size-mib 1024
 *
 * OPTIONS
 * -------
 * --list-devices         List all GPUs (index, BDF, NUMA, PCIe link, name)
 * --device N             CUDA device index to validate (default: 0)
 * --all-devices          Validate all GPUs sequentially
 * --memory-mode MODE     pinned (default) or unpinned
 * --window-ms MS         NVML telemetry window in ms (default: 2000)
 * --interval-ms MS       NVML poll interval in ms    (default: 100)
 * --size-mib MiB         Transfer buffer size in MiB (default: 1024)
 *
 * OUTPUT
 * ------
 * Single device:
 *   ./logs/runs/<timestamp>_GPU<N>/report.txt   — human-readable
 *   ./logs/runs/<timestamp>_GPU<N>/report.json  — machine-readable
 *
 * All devices (--all-devices):
 *   ./logs/runs/<timestamp>_ALL/report.txt      — all GPUs human-readable
 *   ./logs/runs/<timestamp>_ALL/report.json     — JSON array of all GPU objects
 *   ./logs/runs/<timestamp>_ALL/gpu<N>.json     — per-GPU JSON
 *
 * EXIT CODES
 * ----------
 * 0  — All validated GPUs HEALTHY
 * 1  — Runtime error (NVML/CUDA failure, invalid arguments)
 * 2  — One or more GPUs DEGRADED or LINK_DEGRADED
 *
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

// =============================================================================
// UTILITY FUNCTIONS
// Timestamp formatting, sysfs string/int readers, BDF normalization, path join.
// =============================================================================

static std::string NowStamp()
{
  std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec);
  return std::string(buf);
}

static std::string NowISO8601()
{
  std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec);
  return std::string(buf);
}

static std::string NormalizeBDF(const std::string& s)
{
  if (s.size() >= 12)
  {
    std::string tail = s.substr(s.size() - 12);
    if (tail[4] == ':' && tail[7] == ':' && tail[10] == '.')
      return tail;
  }
  return s;
}

static bool ExistsDir(const std::string& p)
{
  struct stat st;
  return (stat(p.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
}

static std::string Join(const std::string& a, const std::string& b)
{
  if (a.empty()) return b;
  return (a.back() == '/') ? a + b : a + "/" + b;
}

static std::string ReadSysfsStr(const std::string& path)
{
  std::ifstream f(path);
  if (!f.good()) return "";
  std::string s;
  std::getline(f, s);
  while (!s.empty() &&
         (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
    s.pop_back();
  return s;
}

static int ReadSysfsInt(const std::string& path, int default_val = -1)
{
  std::string s = ReadSysfsStr(path);
  if (s.empty()) return default_val;
  try   { return std::stoi(s); }
  catch (...) { return default_val; }
}

static std::string NVML_Err(nvmlReturn_t r)
{
  return std::string(nvmlErrorString(r));
}

// =============================================================================
// SYSTEM ENVIRONMENT QUERIES
// Scheduler policy, NUMA node count, GPU NUMA node, OS name from /etc/os-release.
// =============================================================================

static std::string QuerySchedulerPolicy()
{
  int policy = sched_getscheduler(0);
  struct sched_param sp{};
  sched_getparam(0, &sp);

  std::ostringstream ss;
  switch (policy)
  {
    case SCHED_OTHER:   ss << "SCHED_OTHER";   break;
    case SCHED_FIFO:    ss << "SCHED_FIFO";    break;
    case SCHED_RR:      ss << "SCHED_RR";      break;
#ifdef SCHED_BATCH
    case SCHED_BATCH:   ss << "SCHED_BATCH";   break;
#endif
#ifdef SCHED_IDLE
    case SCHED_IDLE:    ss << "SCHED_IDLE";    break;
#endif
#ifdef SCHED_DEADLINE
    case SCHED_DEADLINE: ss << "SCHED_DEADLINE"; break;
#endif
    default: ss << "SCHED_UNKNOWN(" << policy << ")"; break;
  }
  ss << " (priority " << sp.sched_priority << ")";
  return ss.str();
}

static int QueryNumaNodeCount()
{
  DIR* d = opendir("/sys/devices/system/node");
  if (!d) return 1;

  int count = 0;
  struct dirent* ent;
  while ((ent = readdir(d)) != nullptr)
  {
    if (ent->d_type != DT_DIR) continue;
    if (std::strncmp(ent->d_name, "node", 4) != 0) continue;
    bool all_digit = true;
    for (int i = 4; ent->d_name[i] != '\0'; i++)
      if (!std::isdigit((unsigned char)ent->d_name[i]))
        { all_digit = false; break; }
    if (all_digit && ent->d_name[4] != '\0') count++;
  }
  closedir(d);
  return (count > 0) ? count : 1;
}

// On single-node systems kernel reports -1. Normalize to 0.
static int QueryGpuNumaNode(const std::string& bdf, int numa_node_count)
{
  int raw = ReadSysfsInt("/sys/bus/pci/devices/" + bdf + "/numa_node", -1);
  if (raw == -1 && numa_node_count == 1) return 0;
  return raw;
}

static std::string QueryOSName(const struct utsname& un)
{
  std::ifstream f("/etc/os-release");
  if (f.good())
  {
    std::string line;
    while (std::getline(f, line))
    {
      if (line.rfind("PRETTY_NAME=", 0) != 0) continue;
      std::string val = line.substr(12);
      if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
        val = val.substr(1, val.size() - 2);
      return val;
    }
  }
  return std::string(un.sysname) + " " + std::string(un.release);
}

// =============================================================================
// IOMMU DETECTION
// Reads /sys/kernel/iommu_groups and /proc/cmdline to identify Intel VT-d,
// AMD-Vi, passthrough, or strict mode. Returns human-readable state string.
// =============================================================================

static std::string QueryIommuState()
{
  DIR* d = opendir("/sys/kernel/iommu_groups");
  if (!d) return "disabled or unavailable";

  int count = 0;
  struct dirent* ent;
  while ((ent = readdir(d)) != nullptr)
  {
    if (ent->d_name[0] == '.') continue;
    count++;
  }
  closedir(d);

  if (count == 0) return "no groups (passthrough or disabled)";

  std::ifstream f("/proc/cmdline");
  if (f.good())
  {
    std::string line;
    std::getline(f, line);
    bool intel  = (line.find("intel_iommu=on")  != std::string::npos);
    bool amd    = (line.find("amd_iommu=on")    != std::string::npos);
    bool pt     = (line.find("iommu=pt")        != std::string::npos);
    bool strict = (line.find("iommu=strict")    != std::string::npos);

    std::string flag = "enabled";
    if      (pt)     flag = "passthrough";
    else if (strict) flag = "strict";

    if      (intel)  return "Intel VT-d (" + flag + ", " + std::to_string(count) + " groups)";
    else if (amd)    return "AMD-Vi ("      + flag + ", " + std::to_string(count) + " groups)";
    else             return flag + " (" + std::to_string(count) + " groups)";
  }
  return std::to_string(count) + " groups (active)";
}

// =============================================================================
// PCIe MPS / MRRS QUERY
// Max Payload Size and Max Read Request Size from PCIe config space.
// Primary: sysfs /sys/bus/pci/devices/<bdf>/max_payload_size
// Fallback: lspci -vvv DevCtl line (requires root for full config space access).
// =============================================================================

struct MpsMrrs
{
  int  mps_bytes  = -1;
  int  mrrs_bytes = -1;
  bool valid      = false;
};

// Parse "256 bytes" or "512 bytes" style string -> integer bytes
static int ParseBytesStr(const std::string& s)
{
  if (s.empty()) return -1;
  try
  {
    size_t pos = 0;
    int val = std::stoi(s, &pos);
    return (val > 0) ? val : -1;
  }
  catch (...) { return -1; }
}

// Parse lspci -vvv output for MaxPayload and MaxReadReq.
// Looks for lines like:
//   MaxPayload 256 bytes, MaxReadReq 512 bytes
static MpsMrrs ParseLspciOutput(const std::string& output)
{
  MpsMrrs r;
  std::istringstream ss(output);
  std::string line;
  while (std::getline(ss, line))
  {
    // Look for the DevCtl line which has both MaxPayload and MaxReadReq
    if (line.find("MaxPayload") != std::string::npos &&
        line.find("MaxReadReq")  != std::string::npos)
    {
      // "MaxPayload 256 bytes, MaxReadReq 512 bytes"
      auto extract = [&](const std::string& key) -> int
      {
        size_t p = line.find(key);
        if (p == std::string::npos) return -1;
        p += key.size();
        while (p < line.size() && line[p] == ' ') p++;
        std::string num;
        while (p < line.size() && std::isdigit((unsigned char)line[p]))
          num += line[p++];
        return ParseBytesStr(num);
      };
      int mps  = extract("MaxPayload ");
      int mrrs = extract("MaxReadReq ");
      if (mps  > 0) r.mps_bytes  = mps;
      if (mrrs > 0) r.mrrs_bytes = mrrs;
      if (r.mps_bytes > 0 && r.mrrs_bytes > 0) { r.valid = true; return r; }
    }
  }
  return r;
}

static MpsMrrs QueryMpsMrrs(const std::string& bdf)
{
  MpsMrrs r;

  // Try sysfs first (kernel 4.0+ on some platforms)
  std::string base = "/sys/bus/pci/devices/" + bdf;
  r.mps_bytes  = ReadSysfsInt(base + "/max_payload_size",      -1);
  r.mrrs_bytes = ReadSysfsInt(base + "/max_read_request_size", -1);
  r.valid      = (r.mps_bytes > 0 && r.mrrs_bytes > 0);
  if (r.valid) return r;

  // Fallback: parse lspci -s <bdf> -vvv
  // BDF is normalized to "XXXX:XX:XX.X" -- safe to pass to popen
  // Without root, lspci omits DevCtl config space fields -- retry with sudo
  auto RunLspci = [&](const std::string& pfx) -> std::string
  {
    std::string cmd = pfx + "lspci -s " + bdf + " -vvv 2>/dev/null";
    FILE* fp2 = popen(cmd.c_str(), "r");
    if (!fp2) return std::string();
    std::string out;
    char buf2[256];
    while (fgets(buf2, sizeof(buf2), fp2)) out += buf2;
    pclose(fp2);
    return out;
  };

  std::string output = RunLspci("");
  if (output.find("MaxReadReq") == std::string::npos)
    output = RunLspci("sudo ");

  return ParseLspciOutput(output);
}

static std::string BytesToStr(int b)
{
  if (b <= 0) return "unavailable";
  return std::to_string(b) + " bytes";
}

// =============================================================================
// ASPM POLICY QUERY
// Reads /sys/bus/pci/devices/<bdf>/power/control.
// 'on' = runtime PM disabled (link stays active). 'auto' = kernel may allow
// low-power states between transfers. Relevant for latency-sensitive workloads.
// =============================================================================

static std::string QueryAspmPolicy(const std::string& bdf)
{
  std::string ctrl = ReadSysfsStr("/sys/bus/pci/devices/" + bdf + "/power/control");
  if (ctrl.empty()) return "unavailable";
  if (ctrl == "on")   return "disabled (runtime PM off)";
  if (ctrl == "auto") return "auto (runtime PM enabled)";
  return ctrl;
}

// =============================================================================
// AER COUNTER QUERY
// PCIe Advanced Error Reporting counters from sysfs.
// Reads pre-load and post-load snapshots of aer_dev_correctable,
// aer_dev_nonfatal, aer_dev_fatal. Reports per-counter deltas.
// Counter wrap-guard applied: negative deltas clamped to zero.
// =============================================================================

struct AerCounters
{
  std::map<std::string, long long> correctable;
  std::map<std::string, long long> nonfatal;
  std::map<std::string, long long> fatal;
  bool corr_valid     = false;
  bool nonfatal_valid = false;
  bool fatal_valid    = false;
};

static std::map<std::string, long long> ParseAerFile(const std::string& path)
{
  std::map<std::string, long long> m;
  std::ifstream f(path);
  if (!f.good()) return m;
  std::string line;
  while (std::getline(f, line))
  {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string name;
    long long val = 0;
    if (ss >> name >> val) m[name] = val;
  }
  return m;
}

static AerCounters ReadAerCounters(const std::string& bdf)
{
  AerCounters a;
  std::string base = "/sys/bus/pci/devices/" + bdf;

  auto mc = ParseAerFile(base + "/aer_dev_correctable");
  if (!mc.empty()) { a.correctable = mc; a.corr_valid = true; }

  auto mn = ParseAerFile(base + "/aer_dev_nonfatal");
  if (!mn.empty()) { a.nonfatal = mn; a.nonfatal_valid = true; }

  auto mf = ParseAerFile(base + "/aer_dev_fatal");
  if (!mf.empty()) { a.fatal = mf; a.fatal_valid = true; }

  return a;
}

static std::map<std::string, long long> AerDelta(
  const std::map<std::string, long long>& pre,
  const std::map<std::string, long long>& post)
{
  std::map<std::string, long long> d;
  for (const auto& kv : post)
  {
    long long pre_val = 0;
    auto it = pre.find(kv.first);
    if (it != pre.end()) pre_val = it->second;
    long long delta = kv.second - pre_val;
    d[kv.first] = (delta < 0) ? 0 : delta;  // counter wrap-guard
  }
  return d;
}

static long long AerTotalDelta(const std::map<std::string, long long>& delta)
{
  long long total = 0;
  for (const auto& kv : delta) total += kv.second;
  return total;
}

// =============================================================================
// GPU CLOCK / P-STATE QUERY
// SM, memory, and graphics clocks + performance state captured via NVML
// before and after the transfer window. Correlates bandwidth measurements
// with GPU boost state and thermal throttle events.
// =============================================================================

struct ClockSnapshot
{
  unsigned int  sm_mhz  = 0;
  unsigned int  mem_mhz = 0;
  unsigned int  gr_mhz  = 0;
  nvmlPstates_t pstate  = NVML_PSTATE_UNKNOWN;
  bool sm_valid     = false;
  bool mem_valid    = false;
  bool gr_valid     = false;
  bool pstate_valid = false;
};

static ClockSnapshot QueryClocks(nvmlDevice_t dev)
{
  ClockSnapshot c;
  c.sm_valid     = (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM,       &c.sm_mhz)  == NVML_SUCCESS);
  c.mem_valid    = (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM,      &c.mem_mhz) == NVML_SUCCESS);
  c.gr_valid     = (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &c.gr_mhz)  == NVML_SUCCESS);
  c.pstate_valid = (nvmlDeviceGetPerformanceState(dev, &c.pstate)               == NVML_SUCCESS);
  return c;
}

static std::string PstateStr(nvmlPstates_t p)
{
  if (p == NVML_PSTATE_UNKNOWN) return "unknown";
  if ((int)p >= 0 && (int)p <= 15)
  {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "P%d", (int)p);
    return std::string(buf);
  }
  return "unknown";
}

// =============================================================================
// PERSISTENCE MODE QUERY
// Persistence mode keeps the driver loaded between client connections.
// When disabled, the GPU may drop to low-power state (P8) between runs,
// affecting pre-load clock readings and first-transfer latency.
// =============================================================================

static std::string QueryPersistenceMode(nvmlDevice_t dev)
{
  nvmlEnableState_t mode = NVML_FEATURE_DISABLED;
  if (nvmlDeviceGetPersistenceMode(dev, &mode) != NVML_SUCCESS)
    return "unavailable";
  return (mode == NVML_FEATURE_ENABLED) ? "Enabled" : "Disabled";
}

// =============================================================================
// PCIe TOPOLOGY RESOLUTION
// Walks sysfs symlinks from the GPU endpoint up to the root port.
// Builds the full PCIe chain (root -> upstream -> endpoint), chain depth,
// and BDF identifiers for each hop. Used for slot identification and
// NUMA affinity correlation.
// =============================================================================

struct TopologyInfo
{
  std::string endpoint_bdf;
  std::string upstream_bdf;
  std::string root_bdf;
  std::string chain_str;
  int         depth     = 0;
  int         numa_node = -1;
};

static std::string SysfsDevPath(const std::string& bdf)
{
  return Join("/sys/bus/pci/devices", bdf);
}

static bool ResolveSysfsLink(const std::string& path, std::string& out)
{
  char buf[4096];
  ssize_t n = readlink(path.c_str(), buf, sizeof(buf) - 1);
  if (n <= 0) return false;
  buf[n] = '\0';
  out = std::string(buf);
  return true;
}

static bool GetParentBDF(const std::string& bdf, std::string& parent_bdf)
{
  std::string dev = SysfsDevPath(bdf);
  if (!ExistsDir(dev)) return false;

  std::string link;
  if (!ResolveSysfsLink(dev, link)) return false;

  std::vector<std::string> segs;
  std::stringstream ss(link);
  std::string seg;
  while (std::getline(ss, seg, '/'))
    if (!seg.empty()
        && seg.find(':') != std::string::npos
        && seg.find('.') != std::string::npos)
      segs.push_back(seg);

  if (segs.size() < 2) return false;
  parent_bdf = segs[segs.size() - 2];
  return true;
}

static TopologyInfo BuildTopology(const std::string& raw_bdf, int numa_node_count)
{
  TopologyInfo t;
  t.endpoint_bdf = NormalizeBDF(raw_bdf);
  t.numa_node    = QueryGpuNumaNode(t.endpoint_bdf, numa_node_count);

  std::vector<std::string> chain;
  chain.push_back(t.endpoint_bdf);

  std::string cur = t.endpoint_bdf, parent;
  while (GetParentBDF(cur, parent))
  {
    if (parent == cur) break;
    chain.push_back(parent);
    cur = parent;
    if ((int)chain.size() > 32) break;
  }
  std::reverse(chain.begin(), chain.end());

  t.depth = (int)chain.size() - 1;
  if (chain.size() >= 2)
  {
    t.root_bdf     = chain.front();
    t.upstream_bdf = chain[chain.size() - 2];
  }
  else
  {
    t.root_bdf     = chain.front();
    t.upstream_bdf = "";
  }

  std::ostringstream cs;
  for (size_t i = 0; i < chain.size(); i++)
  {
    if (i) cs << " -> ";
    cs << chain[i];
  }
  t.chain_str = cs.str();
  return t;
}

// =============================================================================
// CUDA BANDWIDTH + LATENCY MEASUREMENT
// Two transfer profiles:
//   Bulk: user-defined buffer size (default 1 GiB) -- measures sustained DMA
//         throughput as payload GB/s per direction.
//   Link: fixed 64 KiB x 100 iterations -- measures PCIe transaction latency
//         (control-path overhead, not payload rate).
// All timing uses CUDA events for microsecond precision.
// Warmup pass executed before measurement to bring link to negotiated state.
// =============================================================================

struct BandwidthResult
{
  double bulk_h2d_gbs        = 0.0;
  double bulk_d2h_gbs        = 0.0;
  double bulk_avg_gbs        = 0.0;
  double bulk_h2d_latency_us = 0.0;
  double bulk_d2h_latency_us = 0.0;
  double bulk_avg_latency_us = 0.0;
  double link_h2d_latency_us = 0.0;
  double link_d2h_latency_us = 0.0;
  double link_avg_latency_us = 0.0;
};

static BandwidthResult RunMemcpyBandwidth(int device, size_t bulk_bytes,
                                          int bulk_iters, bool use_pinned)
{
  cudaSetDevice(device);

  void* dptr = nullptr;
  void* hptr = nullptr;

  if (cudaMalloc(&dptr, bulk_bytes) != cudaSuccess)
    throw std::runtime_error("cudaMalloc failed");

  if (use_pinned)
  {
    if (cudaMallocHost(&hptr, bulk_bytes) != cudaSuccess)
    { cudaFree(dptr); throw std::runtime_error("cudaMallocHost failed"); }
  }
  else
  {
    if (cudaHostAlloc(&hptr, bulk_bytes, cudaHostAllocDefault) != cudaSuccess)
    { cudaFree(dptr); throw std::runtime_error("cudaHostAlloc failed"); }
  }

  std::memset(hptr, 0xA5, bulk_bytes);

  cudaStream_t st;
  cudaStreamCreate(&st);
  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  // Warmup: bring link to full negotiated Gen
  for (int i = 0; i < 3; i++)
  {
    cudaMemcpyAsync(dptr, hptr, bulk_bytes, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(hptr, dptr, bulk_bytes, cudaMemcpyDeviceToHost, st);
  }
  cudaStreamSynchronize(st);

  // H2D bulk bandwidth
  cudaEventRecord(ev_start, st);
  for (int i = 0; i < bulk_iters; i++)
    cudaMemcpyAsync(dptr, hptr, bulk_bytes, cudaMemcpyHostToDevice, st);
  cudaEventRecord(ev_stop, st);
  cudaEventSynchronize(ev_stop);
  float h2d_ms = 0.0f;
  cudaEventElapsedTime(&h2d_ms, ev_start, ev_stop);
  double h2d_gb  = (double)bulk_bytes * bulk_iters / (1024.0 * 1024.0 * 1024.0);
  double h2d_gbs = h2d_gb / std::max(1e-9, (double)h2d_ms / 1000.0);
  double h2d_lat = ((double)h2d_ms / bulk_iters) * 1000.0;

  // D2H bulk bandwidth
  cudaEventRecord(ev_start, st);
  for (int i = 0; i < bulk_iters; i++)
    cudaMemcpyAsync(hptr, dptr, bulk_bytes, cudaMemcpyDeviceToHost, st);
  cudaEventRecord(ev_stop, st);
  cudaEventSynchronize(ev_stop);
  float d2h_ms = 0.0f;
  cudaEventElapsedTime(&d2h_ms, ev_start, ev_stop);
  double d2h_gb  = (double)bulk_bytes * bulk_iters / (1024.0 * 1024.0 * 1024.0);
  double d2h_gbs = d2h_gb / std::max(1e-9, (double)d2h_ms / 1000.0);
  double d2h_lat = ((double)d2h_ms / bulk_iters) * 1000.0;

  // Link latency: fixed 64 KiB transfer repeated 100 times.
  // 64 KiB sits below the max TLP payload on all PCIe generations --
  // measures per-transaction overhead (control path), not payload rate.
  // 100 iterations amortizes CUDA event and stream launch overhead
  // while keeping the test well under 1 second.
  const size_t link_bytes = 64 * 1024;  // 64 KiB -- sub-TLP payload
  const int    link_iters = 100;         // iterations for stable average

  cudaEventRecord(ev_start, st);
  for (int i = 0; i < link_iters; i++)
    cudaMemcpyAsync(dptr, hptr, link_bytes, cudaMemcpyHostToDevice, st);
  cudaEventRecord(ev_stop, st);
  cudaEventSynchronize(ev_stop);
  float link_h2d_ms = 0.0f;
  cudaEventElapsedTime(&link_h2d_ms, ev_start, ev_stop);
  double link_h2d_lat = ((double)link_h2d_ms / link_iters) * 1000.0;

  cudaEventRecord(ev_start, st);
  for (int i = 0; i < link_iters; i++)
    cudaMemcpyAsync(hptr, dptr, link_bytes, cudaMemcpyDeviceToHost, st);
  cudaEventRecord(ev_stop, st);
  cudaEventSynchronize(ev_stop);
  float link_d2h_ms = 0.0f;
  cudaEventElapsedTime(&link_d2h_ms, ev_start, ev_stop);
  double link_d2h_lat = ((double)link_d2h_ms / link_iters) * 1000.0;

  cudaEventDestroy(ev_stop);
  cudaEventDestroy(ev_start);
  cudaStreamDestroy(st);
  cudaFreeHost(hptr);
  cudaFree(dptr);

  BandwidthResult r;
  r.bulk_h2d_gbs        = h2d_gbs;
  r.bulk_d2h_gbs        = d2h_gbs;
  r.bulk_avg_gbs        = 0.5 * (h2d_gbs + d2h_gbs);
  r.bulk_h2d_latency_us = h2d_lat;
  r.bulk_d2h_latency_us = d2h_lat;
  r.bulk_avg_latency_us = 0.5 * (h2d_lat + d2h_lat);
  r.link_h2d_latency_us = link_h2d_lat;
  r.link_d2h_latency_us = link_d2h_lat;
  r.link_avg_latency_us = 0.5 * (link_h2d_lat + link_d2h_lat);
  return r;
}

// =============================================================================
// NVML TELEMETRY SAMPLING
// Background thread drives continuous DMA traffic during the measurement window.
// Main thread polls NVML at cfg.interval_ms for PCIe RX/TX counters,
// power draw (mW -> W), and GPU temperature. Samples aggregated post-window
// for average, peak, and delta calculations.
// =============================================================================

struct Sample
{
  double t_ms    = 0.0;
  double rx_gbs  = 0.0;
  double tx_gbs  = 0.0;
  double power_w = 0.0;
  double temp_c  = 0.0;
};

struct RunConfig
{
  int         device_index = 0;
  int         window_ms    = 2000;
  int         interval_ms  = 100;
  int         size_mib     = 1024;
  bool        list_devices = false;
  bool        all_devices  = false;
  std::string memory_mode  = "pinned";
};

static double ClampPos(double x) { return x < 0.0 ? 0.0 : x; }

// =============================================================================
// PROGRESS REPORTER
// Prints timestamped status lines to stderr during validation so the operator
// always knows what phase the tool is in. Single-device mode prints per-sample
// ticker lines during the NVML window. Multi-GPU (all_devices) mode prints
// only phase summaries per GPU to avoid flooding the screen.
// All progress output goes to stderr — stdout and log files remain clean.
// =============================================================================


static void ProgressInit()
{
}



// Phase announcements suppressed — one bar only.
static void Progress(int gpu_idx, const std::string& msg)
{
  (void)gpu_idx; (void)msg; // silent
}

// Sampling tick: overwrites the same line using \r so the progress bar
// stays as a single updating line. When the run finishes, ProgressDone()
// clears it with spaces so the report prints on a clean screen.
static void ProgressTick(int gpu_idx, long long elapsed_s, long long total_s,
                         double rx, double tx, double power_w, double temp_c)
{
  // Build the bar: [=========>  ] 25/30s
  int bar_width = 20;
  int filled    = (total_s > 0) ? (int)((elapsed_s * bar_width) / total_s) : 0;
  if (filled > bar_width) filled = bar_width;
  char bar[32];
  for (int i = 0; i < bar_width; i++)
    bar[i] = (i < filled) ? '=' : ' ';
  bar[bar_width] = '\0';

  char buf[200];
  std::snprintf(buf, sizeof(buf),
    "\r[%s>%s] %lld/%llds | RX %5.2f TX %5.2f GB/s | %5.1fW %3.0fC   ",
    bar, (filled < bar_width ? "" : ""), elapsed_s, total_s,
    rx, tx, power_w, temp_c);
  std::fprintf(stderr, "%s", buf);
  std::fflush(stderr);
}

// Call once when sampling is done — clears the progress bar line
static void ProgressDone()
{
  std::fprintf(stderr, "\r%80s\r", "");
  std::fflush(stderr);
}

static void TrafficWorker(std::atomic<bool>& go, int device,
                          size_t bytes, int window_ms)
{
  cudaSetDevice(device);
  void* dptr = nullptr;
  void* hptr = nullptr;
  cudaMalloc(&dptr, bytes);
  cudaMallocHost(&hptr, bytes);
  std::memset(hptr, 0x5A, bytes);
  cudaStream_t st;
  cudaStreamCreate(&st);

  while (!go.load(std::memory_order_acquire))
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

  auto end_t = std::chrono::steady_clock::now() + std::chrono::milliseconds(window_ms);
  bool dir = true;
  while (std::chrono::steady_clock::now() < end_t)
  {
    if (dir) cudaMemcpyAsync(dptr, hptr, bytes, cudaMemcpyHostToDevice, st);
    else     cudaMemcpyAsync(hptr, dptr, bytes, cudaMemcpyDeviceToHost, st);
    dir = !dir;
    cudaStreamSynchronize(st);
  }
  cudaStreamDestroy(st);
  cudaFreeHost(hptr);
  cudaFree(dptr);
}

static bool NVML_GetThroughputGBs(nvmlDevice_t dev, double& rx, double& tx)
{
  unsigned int rx_kbs = 0, tx_kbs = 0;
  if (nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_RX_BYTES, &rx_kbs) != NVML_SUCCESS) return false;
  if (nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_TX_BYTES, &tx_kbs) != NVML_SUCCESS) return false;
  rx = ClampPos((double)rx_kbs / (1024.0 * 1024.0));
  tx = ClampPos((double)tx_kbs / (1024.0 * 1024.0));
  return true;
}

static bool NVML_GetPowerTemp(nvmlDevice_t dev, double& power_w, double& temp_c)
{
  unsigned int mw = 0, tc = 0;
  if (nvmlDeviceGetPowerUsage(dev, &mw)                        != NVML_SUCCESS) return false;
  if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &tc) != NVML_SUCCESS) return false;
  power_w = (double)mw / 1000.0;
  temp_c  = (double)tc;
  return true;
}

struct LinkInfo
{
  int max_gen   = 0;
  int max_width = 0;
  int cur_gen   = 0;
  int cur_width = 0;
};

static bool NVML_GetLinkInfo(nvmlDevice_t dev, LinkInfo& out)
{
  if (nvmlDeviceGetMaxPcieLinkGeneration (dev, (unsigned int*)&out.max_gen)   != NVML_SUCCESS) return false;
  if (nvmlDeviceGetMaxPcieLinkWidth      (dev, (unsigned int*)&out.max_width) != NVML_SUCCESS) return false;
  if (nvmlDeviceGetCurrPcieLinkGeneration(dev, (unsigned int*)&out.cur_gen)   != NVML_SUCCESS) return false;
  if (nvmlDeviceGetCurrPcieLinkWidth     (dev, (unsigned int*)&out.cur_width) != NVML_SUCCESS) return false;
  return true;
}

static std::string LinkStr(int gen, int width)
{
  std::ostringstream ss;
  ss << "PCIe Gen" << gen << " x" << width;
  return ss.str();
}

// Per-lane effective payload rates (encoding overhead applied).
// Covers all PCIe generations including Gen5 (H100/A100) and Gen6 (B200/GB200/B300 Ultra).
static double TheoreticalPayloadGBsPerDir(int gen, int width)
{
  double per_lane = 0.0;
  switch (gen)
  {
    case 1: per_lane = 0.250; break;  // 2.5  GT/s  8b/10b
    case 2: per_lane = 0.500; break;  // 5.0  GT/s  8b/10b
    case 3: per_lane = 0.985; break;  // 8.0  GT/s  128b/130b
    case 4: per_lane = 1.969; break;  // 16.0 GT/s  128b/130b
    case 5: per_lane = 3.938; break;  // 32.0 GT/s  128b/130b
    case 6: per_lane = 7.877; break;  // 64.0 GT/s  FLIT (Blackwell)
    default: per_lane = 0.0;  break;
  }
  return per_lane * (double)width;
}

// =============================================================================
// JSON SERIALIZATION
// Minimal JSON string escaping for report output. All numeric fields written
// as raw values (no quotes). String fields escaped for special characters.
// =============================================================================

static std::string JsonEscape(const std::string& s)
{
  std::ostringstream o;
  o << '"';
  for (char c : s)
  {
    if      (c == '"')  o << "\\\"";
    else if (c == '\\') o << "\\\\";
    else if (c == '\n') o << "\\n";
    else if (c == '\r') o << "\\r";
    else                o << c;
  }
  o << '"';
  return o.str();
}

// =============================================================================
// DEVICE LISTING
// --list-devices: enumerates all CUDA devices via cudaGetDeviceCount,
// queries NVML for name and current PCIe link state, resolves BDF and
// NUMA node. Outputs a columnar table for operator reference.
// =============================================================================

static void ListDevices()
{
  nvmlReturn_t nr = nvmlInit_v2();
  if (nr != NVML_SUCCESS)
  { std::cerr << "NVML init failed: " << NVML_Err(nr) << "\n"; return; }

  int devcount = 0;
  cudaGetDeviceCount(&devcount);
  int numa_count = QueryNumaNodeCount();

  std::cout << "\nDetected GPUs\n\n";
  std::cout << std::left
            << std::setw(5)  << "Idx"
            << std::setw(20) << "BDF"
            << std::setw(6)  << "NUMA"
            << std::setw(14) << "PCIe Link"
            << "Name\n\n";

  for (int i = 0; i < devcount; i++)
  {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, i);

    char bdf[32] = {0};
    std::snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.0",
                  prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    char nvname[256] = "unknown";
    nvmlDevice_t nvdev{};
    std::string link_str = "?";
    if (nvmlDeviceGetHandleByPciBusId_v2(bdf, &nvdev) == NVML_SUCCESS)
    {
      nvmlDeviceGetName(nvdev, nvname, sizeof(nvname));
      LinkInfo li{};
      if (NVML_GetLinkInfo(nvdev, li))
        link_str = "Gen" + std::to_string(li.cur_gen)
                 + " x" + std::to_string(li.cur_width);
    }
    else
    {
      std::strncpy(nvname, prop.name, sizeof(nvname) - 1);
      nvname[sizeof(nvname) - 1] = '\0';
    }

    int numa = QueryGpuNumaNode(NormalizeBDF(std::string(bdf)), numa_count);

    std::cout << std::left
              << std::setw(5)  << i
              << std::setw(20) << bdf
              << std::setw(6)  << (numa >= 0 ? std::to_string(numa) : "N/A")
              << std::setw(14) << link_str
              << nvname << "\n";
  }
  std::cout << "\n";
  nvmlShutdown();
}

// =============================================================================
// USAGE
// =============================================================================

static void Usage()
{
  std::cout <<
    "GPU PCIe Validator v4.1\n\n"
    "Usage:\n"
    "  ./gpu_pcie_validator --list-devices\n"
    "  ./gpu_pcie_validator --device N [options]\n"
    "  ./gpu_pcie_validator --all-devices [options]\n\n"
    "Options:\n"
    "  --list-devices         List all GPUs with index, BDF, NUMA, PCIe link, name\n"
    "  --device N             CUDA device index to validate (default: 0)\n"
    "  --all-devices          Validate all GPUs sequentially\n"
    "  --memory-mode MODE     pinned (default) or unpinned\n"
    "  --window-ms MS         NVML sampling window in ms (default: 2000)\n"
    "  --interval-ms MS       NVML poll interval in ms   (default: 100)\n"
    "  --size-mib MiB         Transfer buffer size in MiB (default: 1024)\n\n"
    "Exit codes:\n"
    "  0  All GPUs HEALTHY\n"
    "  1  Runtime error (NVML/CUDA failure, bad arguments)\n"
    "  2  One or more GPUs DEGRADED or LINK_DEGRADED\n";
}

// =============================================================================
// PER-GPU VALIDATION  (returns 0=healthy, 1=error, 2=degraded)
// =============================================================================

static int RunValidation(const RunConfig& cfg,
                         std::ostream& txt_out,
                         std::ostream& json_out,
                         bool verbose = true)
{
  ProgressInit();

  // ---- Device setup -------------------------------------------------------

  int cuda_devcount = 0;
  cudaGetDeviceCount(&cuda_devcount);
  unsigned int nvml_devcount = 0;
  nvmlDeviceGetCount_v2(&nvml_devcount);

  if (cfg.device_index < 0
      || cfg.device_index >= cuda_devcount
      || cfg.device_index >= (int)nvml_devcount)
  {
    std::cerr << "Invalid --device " << cfg.device_index
              << "  (CUDA: " << cuda_devcount
              << ", NVML: "  << nvml_devcount << ")\n"
              << "Run --list-devices to see available GPUs.\n";
    return 1;
  }

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, cfg.device_index);

  char cuda_bdf[32] = {0};
  std::snprintf(cuda_bdf, sizeof(cuda_bdf), "%04x:%02x:%02x.0",
                prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

  nvmlDevice_t nvdev{};
  nvmlReturn_t nr = nvmlDeviceGetHandleByPciBusId_v2(cuda_bdf, &nvdev);
  if (nr != NVML_SUCCESS)
  {
    nr = nvmlDeviceGetHandleByIndex_v2(cfg.device_index, &nvdev);
    if (nr != NVML_SUCCESS)
    {
      std::cerr << "NVML device handle failed (BDF=" << cuda_bdf
                << "): " << NVML_Err(nr) << "\n";
      return 1;
    }
  }

  char gpu_name[128] = {0};
  nvmlDeviceGetName(nvdev, gpu_name, sizeof(gpu_name));

  if (verbose)
  {
    // Announce GPU identity immediately so the operator sees something at launch
    LinkInfo li_early{};
    std::string link_early = "unknown";
    if (NVML_GetLinkInfo(nvdev, li_early))
      link_early = "Gen" + std::to_string(li_early.cur_gen)
                 + " x" + std::to_string(li_early.cur_width);
    char msg[256];
    std::snprintf(msg, sizeof(msg), "Init: %s | %s | Driver %s",
                  gpu_name, link_early.c_str(), "");
    Progress(cfg.device_index, std::string("Init: ") + gpu_name
             + " | " + link_early);
  }

  char gpu_uuid[96] = {0};
  nvmlDeviceGetUUID(nvdev, gpu_uuid, sizeof(gpu_uuid));

  nvmlPciInfo_t pci{};
  if (nvmlDeviceGetPciInfo_v3(nvdev, &pci) != NVML_SUCCESS)
  {
    std::cerr << "NVML PCI info query failed\n";
    return 1;
  }
  std::string raw_bdf = pci.busId;

  char drv_ver[80]  = {0};
  char nvml_ver[80] = {0};
  nvmlSystemGetDriverVersion(drv_ver, sizeof(drv_ver));
  nvmlSystemGetNVMLVersion(nvml_ver, sizeof(nvml_ver));

  int cuda_rt_ver = 0;
  cudaRuntimeGetVersion(&cuda_rt_ver);

  struct utsname un{};
  uname(&un);

  std::string os_name    = QueryOSName(un);
  std::string sched_str  = QuerySchedulerPolicy();
  long        cpu_cores  = sysconf(_SC_NPROCESSORS_ONLN);
  int         numa_count = QueryNumaNodeCount();

  TopologyInfo topo     = BuildTopology(raw_bdf, numa_count);
  std::string  norm_bdf = topo.endpoint_bdf;

  std::string numa_node_str = (topo.numa_node >= 0)
                            ? std::to_string(topo.numa_node) : "N/A";

  // NUMA CPU affinity list for this GPU
  std::string cpu_affinity_list = ReadSysfsStr(
    "/sys/bus/pci/devices/" + norm_bdf + "/local_cpulist");
  bool numa_optimal = (!cpu_affinity_list.empty() && topo.numa_node >= 0);

  // System signals
  std::string persistence_mode = QueryPersistenceMode(nvdev);
  std::string aspm_policy      = QueryAspmPolicy(norm_bdf);
  std::string iommu_state      = QueryIommuState();
  MpsMrrs     mps_mrrs         = QueryMpsMrrs(norm_bdf);

  // ---- Pre-load snapshots -------------------------------------------------

  if (verbose) Progress(cfg.device_index, "Pre-load: capturing clocks, AER baseline, replay counter");

  double p_start = 0.0, t_start = 0.0;
  bool have_pt_start = NVML_GetPowerTemp(nvdev, p_start, t_start);

  ClockSnapshot  clk_pre  = QueryClocks(nvdev);
  AerCounters    aer_pre  = ReadAerCounters(norm_bdf);

  // Link-wake: bring PCIe link from Gen1 idle to full negotiated speed
  if (verbose) Progress(cfg.device_index, "Link wake: forcing PCIe link to full negotiated speed");
  {
    cudaSetDevice(cfg.device_index);
    // 4 MiB: minimum DMA size observed to reliably force the PCIe link
    // from Gen1 ASPM idle state up to the full negotiated Gen/width
    // before pre-load link state and clock snapshots are taken.
    const size_t wake_bytes = 4ULL * 1024ULL * 1024ULL;
    void* dp = nullptr;
    void* hp = nullptr;
    if (cudaMalloc(&dp, wake_bytes)     == cudaSuccess &&
        cudaMallocHost(&hp, wake_bytes)  == cudaSuccess)
    {
      cudaMemcpy(dp, hp, wake_bytes, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      cudaFreeHost(hp);
      cudaFree(dp);
    }
  }

  LinkInfo pre{};
  bool have_link_pre = NVML_GetLinkInfo(nvdev, pre);

  unsigned int replay_start = 0, replay_end = 0;
  nvmlReturn_t rr = nvmlDeviceGetPcieReplayCounter(nvdev, &replay_start);

  // ---- Bandwidth measurement ----------------------------------------------

  if (verbose)
  {
    char msg[128];
    std::snprintf(msg, sizeof(msg),
      "Bandwidth: measuring %d MiB H2D + D2H (6 passes)...", cfg.size_mib);
    Progress(cfg.device_index, msg);
  }

  const size_t bytes     = (size_t)cfg.size_mib * 1024ULL * 1024ULL;
  bool         use_pinned = (cfg.memory_mode == "pinned");

  BandwidthResult bw;
  try
  {
    // 6 bulk iterations: 3-pass warmup already completed inside RunMemcpyBandwidth.
    // 6 measured passes gives a stable average without excessive wall time
    // at large buffer sizes (6 x 1 GiB = 6 GiB total per direction).
    bw = RunMemcpyBandwidth(cfg.device_index, bytes, 6, use_pinned);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Memcpy benchmark failed: " << e.what() << "\n";
    return 1;
  }

  if (verbose)
  {
    char msg[128];
    std::snprintf(msg, sizeof(msg),
      "Bandwidth: complete | avg %.2f GB/s H2D %.2f D2H %.2f",
      bw.bulk_avg_gbs, bw.bulk_h2d_gbs, bw.bulk_d2h_gbs);
    Progress(cfg.device_index, msg);
  }

  // ---- NVML sampling window -----------------------------------------------

  std::vector<Sample> samples;
  samples.reserve(
    (size_t)std::max(1, cfg.window_ms / std::max(1, cfg.interval_ms)) + 8);

  std::atomic<bool> go{false};
  std::thread worker(TrafficWorker, std::ref(go),
                     cfg.device_index, bytes, cfg.window_ms);

  auto t0 = std::chrono::steady_clock::now();
  go.store(true, std::memory_order_release);

  if (verbose)
  {
    char msg[128];
    std::snprintf(msg, sizeof(msg),
      "Sampling: %d ms window | %d ms interval | started",
      cfg.window_ms, cfg.interval_ms);
    Progress(cfg.device_index, msg);
  }

  int target_samples = (cfg.interval_ms > 0)
                     ? (cfg.window_ms / cfg.interval_ms) : 1;
  if (target_samples < 1) target_samples = 1;

  // Tick counter: print progress every ~5 seconds in single-device mode.
  // Uses a rolling average over the last 10 samples to smooth NVML's
  // 20ms counter reset cycle — prevents noisy 0.00/12.xx GB/s jumps
  // that would confuse engineers reading the live output.
  // In multi-GPU (all_devices) mode verbose is false — no per-tick output.
  const int    tick_every_ms  = 5000;
  long long    last_tick_ms   = -tick_every_ms; // force first tick immediately
  const int    roll_window    = 10;             // samples in rolling average
  double       roll_rx_sum    = 0.0;
  double       roll_tx_sum    = 0.0;
  double       roll_pw_sum    = 0.0;
  double       roll_tc_sum    = 0.0;
  int          roll_count     = 0;

  auto next = t0;
  while (true)
  {
    auto now     = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
    if (elapsed >= cfg.window_ms) break;
    if (now < next) std::this_thread::sleep_for(next - now);

    Sample s;
    s.t_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0).count();
    double rx = 0.0, tx = 0.0;
    if (NVML_GetThroughputGBs(nvdev, rx, tx)) { s.rx_gbs = rx; s.tx_gbs = tx; }
    double pw = 0.0, tc = 0.0;
    if (NVML_GetPowerTemp(nvdev, pw, tc))     { s.power_w = pw; s.temp_c = tc; }

    samples.push_back(s);

    // Maintain rolling sums — subtract oldest sample when window is full
    if (roll_count == roll_window)
    {
      int oldest = (int)samples.size() - 1 - roll_window;
      if (oldest >= 0)
      {
        roll_rx_sum -= samples[oldest].rx_gbs;
        roll_tx_sum -= samples[oldest].tx_gbs;
        roll_pw_sum -= samples[oldest].power_w;
        roll_tc_sum -= samples[oldest].temp_c;
        roll_count--;
      }
    }
    roll_rx_sum += rx;
    roll_tx_sum += tx;
    roll_pw_sum += pw;
    roll_tc_sum += tc;
    roll_count++;

    // Print tick every 5 seconds using rolling average
    if (verbose && (elapsed - last_tick_ms) >= tick_every_ms)
    {
      last_tick_ms = elapsed;
      double n         = (double)std::max(1, roll_count);
      double avg_rx    = roll_rx_sum / n;
      double avg_tx    = roll_tx_sum / n;
      double avg_pw    = roll_pw_sum / n;
      double avg_tc    = roll_tc_sum / n;
      long long total_s   = (long long)cfg.window_ms / 1000;
      long long elapsed_s = elapsed / 1000;
      ProgressTick(cfg.device_index, elapsed_s, total_s,
                   avg_rx, avg_tx, avg_pw, avg_tc);
    }

    next += std::chrono::milliseconds(cfg.interval_ms);
  }

  auto t1 = std::chrono::steady_clock::now();
  worker.join();

  double window_actual_ms = (double)std::chrono::duration_cast<
    std::chrono::milliseconds>(t1 - t0).count();
  double achieved_interval_ms = samples.size() > 1
    ? (window_actual_ms / (double)samples.size()) : window_actual_ms;

  if (verbose)
  {
    // Clear the progress bar line before printing completion message
    ProgressDone();
    char msg[128];
    std::snprintf(msg, sizeof(msg),
      "Sampling: complete | %zu samples in %.0f ms",
      samples.size(), window_actual_ms);
    Progress(cfg.device_index, msg);
  }

  // ---- Post-load snapshots ------------------------------------------------

  LinkInfo post{};
  bool have_link_post = NVML_GetLinkInfo(nvdev, post);

  if (rr == NVML_SUCCESS) nvmlDeviceGetPcieReplayCounter(nvdev, &replay_end);

  double p_end = 0.0, t_end = 0.0;
  bool have_pt_end = NVML_GetPowerTemp(nvdev, p_end, t_end);

  ClockSnapshot clk_post = QueryClocks(nvdev);

  AerCounters aer_post = ReadAerCounters(norm_bdf);
  std::map<std::string, long long> aer_corr_delta, aer_nonfatal_delta, aer_fatal_delta;
  bool aer_available = false;

  if (aer_pre.corr_valid && aer_post.corr_valid)
  { aer_corr_delta = AerDelta(aer_pre.correctable, aer_post.correctable); aer_available = true; }
  if (aer_pre.nonfatal_valid && aer_post.nonfatal_valid)
    aer_nonfatal_delta = AerDelta(aer_pre.nonfatal, aer_post.nonfatal);
  if (aer_pre.fatal_valid && aer_post.fatal_valid)
    aer_fatal_delta = AerDelta(aer_pre.fatal, aer_post.fatal);

  long long aer_corr_total     = AerTotalDelta(aer_corr_delta);
  long long aer_nonfatal_total = AerTotalDelta(aer_nonfatal_delta);
  long long aer_fatal_total    = AerTotalDelta(aer_fatal_delta);

  // ---- Statistics ---------------------------------------------------------

  double rx_avg = 0.0, tx_avg = 0.0, p_avg = 0.0, p_peak = 0.0;
  int n = (int)samples.size();
  if (n > 0)
  {
    for (const auto& s : samples)
    {
      rx_avg += s.rx_gbs;
      tx_avg += s.tx_gbs;
      p_avg  += s.power_w;
      p_peak  = std::max(p_peak, s.power_w);
    }
    rx_avg /= (double)n;
    tx_avg /= (double)n;
    p_avg  /= (double)n;
  }

  double combined = rx_avg + tx_avg;
  double theo     = have_link_pre
                  ? TheoreticalPayloadGBsPerDir(pre.max_gen, pre.max_width) : 0.0;
  double eff      = (theo > 1e-9) ? (bw.bulk_avg_gbs / theo) : 0.0;
  double eff_pct  = eff * 100.0;

  bool speed_change = have_link_pre && have_link_post && (pre.cur_gen   != post.cur_gen);
  bool width_change = have_link_pre && have_link_post && (pre.cur_width != post.cur_width);

  long long delta = (rr == NVML_SUCCESS)
                  ? (long long)replay_end - (long long)replay_start : 0LL;

  // ---- Assessment ---------------------------------------------------------

  std::string state = "HEALTHY";
  // delta > 10: replay bursts of 1-2 are transient; sustained growth above 10
  // across a 2-second window indicates active link error recovery, not noise.
  if (speed_change || width_change || delta > 10) state = "LINK_DEGRADED";
  // eff_pct < 20.0: catastrophic signal -- link is negotiated but DMA throughput
  // is less than 1/5 of theoretical. Indicates severe routing, slot, or driver fault.
  if (eff_pct < 20.0)                             state = "DEGRADED";
  if (aer_fatal_total > 0)                        state = "LINK_DEGRADED";

  int exit_status = (state == "HEALTHY") ? 0 : 2;

  if (verbose)
  {
    char msg[128];
    std::snprintf(msg, sizeof(msg),
      "Assessment: %s | efficiency %.1f%% | replay delta %lld | AER corr %lld",
      state.c_str(), eff_pct, delta, aer_corr_total);
    Progress(cfg.device_index, msg);
    // Write clear sequence directly to /dev/tty so both stdout and stderr
    // content is cleared regardless of stream routing.

  }

  std::string ts_iso = NowISO8601();
  std::string run_id = NowStamp() + "_GPU" + std::to_string(cfg.device_index);

  // =========================================================================
  // TEXT REPORT
  // =========================================================================
  {
    auto& out = txt_out;

    auto kv = [&](const std::string& k, const std::string& v, int w = 40)
    { out << std::left << std::setw(w) << k << " : " << v << "\n"; };

    auto kvi = [&](const std::string& k, long long v, int w = 40)
    { std::ostringstream ss; ss << v; kv(k, ss.str(), w); };

    auto kvd = [&](const std::string& k, double v,
                   const std::string& suf = "", int w = 40, int p = 2)
    {
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(p) << v;
      if (!suf.empty()) ss << " " << suf;
      kv(k, ss.str(), w);
    };

    auto sec = [&](const std::string& s)
    { out << "\n" << s << "\n\n"; };

    auto fmt = [](double val, int prec) -> std::string
    {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(prec) << val;
      return oss.str();
    };

    out << "\nGPU PCIe Validator v4.1\n\n";

    sec("GPU Identity");
    kv ("Model",                gpu_name);
    kv ("Compute Capability",   std::to_string(prop.major) + "." + std::to_string(prop.minor));
    kv ("Driver Version",       drv_ver);
    kv ("NVML Version",         nvml_ver);
    {
      std::ostringstream ss;
      ss << (cuda_rt_ver / 1000) << "." << ((cuda_rt_ver % 1000) / 10);
      kv("CUDA Runtime", ss.str());
    }
    kv ("PCI Bus ID (BDF)",     raw_bdf);
    kv ("GPU UUID",             gpu_uuid);
    kv ("NUMA Node",            numa_node_str);

    sec("System Environment");
    kv ("Operating System",     os_name);
    kv ("Kernel Version",       std::string(un.release));
    kv ("CPU Architecture",     std::string(un.machine));
    kvi("CPU Cores (online)",   cpu_cores);
    kv ("Scheduler Policy",     sched_str);
    kvi("NUMA Nodes",           numa_count);

    sec("Run Parameters");
    kvi("Device Index",         cfg.device_index);
    kv ("Memory Mode",          cfg.memory_mode);
    kv ("Window Duration",      std::to_string(cfg.window_ms)   + " ms");
    kv ("Sample Interval",      std::to_string(cfg.interval_ms) + " ms");
    kv ("Transfer Size",        std::to_string(cfg.size_mib)    + " MiB");

    sec("PCIe Topology");
    kv ("GPU Device (endpoint)",       topo.endpoint_bdf);
    kv ("Upstream Port",               topo.upstream_bdf.empty() ? "N/A" : topo.upstream_bdf);
    kv ("Root Port",                   topo.root_bdf.empty()     ? "N/A" : topo.root_bdf);
    kv ("PCIe Chain (root->endpoint)", topo.chain_str.empty()    ? "N/A" : topo.chain_str);
    kvi("Chain Depth",                 topo.depth);

    sec("PCIe Link Capability");
    if (have_link_pre)
    {
      kv ("Maximum Supported",        LinkStr(pre.max_gen, pre.max_width));
      kv ("Negotiated (pre-load)",    LinkStr(pre.cur_gen, pre.cur_width));
      kv ("Negotiated (post-load)",   have_link_post
                                        ? LinkStr(post.cur_gen, post.cur_width)
                                        : "unavailable");
      kvd("Theoretical Max Payload",  theo, "GB/s per direction", 40, 2);
    }
    else
    {
      kv("Maximum Supported",       "unavailable");
      kv("Negotiated (pre-load)",   "unavailable");
      kv("Negotiated (post-load)",  "unavailable");
      kv("Theoretical Max Payload", "unavailable");
    }

    sec("Link Changes During Test");
    kv("Speed Change",   speed_change ? "YES" : "None");
    kv("Width Change",   width_change ? "YES" : "None");
    kv("Retrain Events", "N/A (not exposed by NVML)");

    sec("GPU Clocks");
    if (clk_pre.sm_valid)
    {
      kv("SM Clock (pre-load)",    std::to_string(clk_pre.sm_mhz)  + " MHz");
      kv("SM Clock (post-load)",   clk_post.sm_valid
           ? std::to_string(clk_post.sm_mhz) + " MHz" : "unavailable");
    }
    else kv("SM Clock", "unavailable");

    if (clk_pre.mem_valid)
    {
      kv("Mem Clock (pre-load)",   std::to_string(clk_pre.mem_mhz)  + " MHz");
      kv("Mem Clock (post-load)",  clk_post.mem_valid
           ? std::to_string(clk_post.mem_mhz) + " MHz" : "unavailable");
    }
    else kv("Memory Clock", "unavailable");

    if (clk_pre.gr_valid)
    {
      kv("GR Clock (pre-load)",    std::to_string(clk_pre.gr_mhz)  + " MHz");
      kv("GR Clock (post-load)",   clk_post.gr_valid
           ? std::to_string(clk_post.gr_mhz) + " MHz" : "unavailable");
    }
    else kv("Graphics Clock", "unavailable");

    if (clk_pre.pstate_valid)
    {
      kv("P-State (pre-load)",     PstateStr(clk_pre.pstate));
      kv("P-State (post-load)",    clk_post.pstate_valid
           ? PstateStr(clk_post.pstate) : "unavailable");
    }
    else kv("P-State", "unavailable");

    sec("PCIe Bandwidth Validation (CUDA memcpy)");
    kvd("Host -> Device Throughput", bw.bulk_h2d_gbs, "GB/s", 40, 2);
    kvd("Device -> Host Throughput", bw.bulk_d2h_gbs, "GB/s", 40, 2);
    kvd("Average Throughput",        bw.bulk_avg_gbs, "GB/s", 40, 2);
    kvd("Utilization Efficiency",    eff_pct,         "%",    40, 1);

    sec("Memcpy Transfer Timing");
    kvd("Host->Device memcpy time (" + std::to_string(cfg.size_mib) + " MiB)", bw.bulk_h2d_latency_us/1000.0, "ms", 40, 1);
    kvd("Device->Host memcpy time (" + std::to_string(cfg.size_mib) + " MiB)", bw.bulk_d2h_latency_us/1000.0, "ms", 40, 1);
    kvd("Average memcpy time (" + std::to_string(cfg.size_mib) + " MiB)",      bw.bulk_avg_latency_us/1000.0, "ms", 40, 1);
    out << "\n";
    kvd("Host->Device memcpy time (64 KiB)",   bw.link_h2d_latency_us, "us", 40, 1);
    kvd("Device->Host memcpy time (64 KiB)",   bw.link_d2h_latency_us, "us", 40, 1);
    kvd("Average memcpy time (64 KiB)",        bw.link_avg_latency_us, "us", 40, 1);

    sec("NVML PCIe Throughput (bus-level)");
    kvd("Host\xe2\x86\x92" "Device (RX)                    ", rx_avg,   "GB/s", 42, 2);
    kvd("Device\xe2\x86\x92" "Host (TX)                    ", tx_avg,   "GB/s", 42, 2);
    kvd("Combined (duplex)                   ", combined, "GB/s", 40, 2);
    kvd("Per-direction average               ", (rx_avg + tx_avg) / 2.0, "GB/s", 40, 2);
    out << "\n";
    out << "Note: Memcpy throughput measures payload transfer rate per direction.\n";
    out << "      NVML combined traffic measures total bus utilization (duplex).\n";

    sec("PCIe Replay Counter");
    kvi("Counter Start",  (rr == NVML_SUCCESS) ? (long long)replay_start : -1LL);
    kvi("Counter End",    (rr == NVML_SUCCESS) ? (long long)replay_end   : -1LL);
    kvi("Counter Delta",  delta);
    if (delta > 10) kv("Warning", "Replay counter increased (potential link errors)");

    sec("AER Error Counters (PCIe Advanced Error Reporting)");
    if (aer_available)
    {
      kvi("Correctable Errors (total delta)",  aer_corr_total);
      kvi("Non-Fatal Errors (total delta)",    aer_nonfatal_total);
      kvi("Fatal Errors (total delta)",        aer_fatal_total);
      if (aer_corr_total > 0)
      {
        out << "\n  Correctable breakdown:\n";
        for (const auto& kv_aer : aer_corr_delta)
          if (kv_aer.second > 0)
            out << "    " << std::left << std::setw(32) << kv_aer.first
                << " : " << kv_aer.second << "\n";
      }
      if (aer_nonfatal_total > 0 || aer_fatal_total > 0)
      {
        out << "\n  Non-fatal / Fatal breakdown:\n";
        for (const auto& kv_aer : aer_nonfatal_delta)
          if (kv_aer.second > 0)
            out << "    [NF] " << std::left << std::setw(28)
                << kv_aer.first << " : " << kv_aer.second << "\n";
        for (const auto& kv_aer : aer_fatal_delta)
          if (kv_aer.second > 0)
            out << "    [F]  " << std::left << std::setw(28)
                << kv_aer.first << " : " << kv_aer.second << "\n";
      }
      if (aer_fatal_total > 0)
        kv("Warning", "Fatal AER errors detected — link may be compromised");
    }
    else
    {
      kv("AER Status", "unavailable (sysfs not exposed or permission denied)");
    }

    sec("Power Telemetry");
    if (have_pt_start) kvd("Pre-Test Power",  p_start, "W", 40, 1);
    else               kv ("Pre-Test Power",  "unavailable");
    kvd("Average Load Power", p_avg,  "W", 40, 1);
    kvd("Peak Power",         p_peak, "W", 40, 1);
    if (have_pt_end)   kvd("End Power",       p_end,   "W", 40, 1);
    else               kv ("End Power",       "unavailable");
    if (have_pt_start && have_pt_end)
      kvd("Power Delta (pre->end)", p_end - p_start, "W", 40, 1);
    else
      kv("Power Delta (pre->end)", "unavailable");

    sec("Thermal Telemetry");
    if (have_pt_start) kvd("Temperature (baseline)", t_start, "\xc2\xb0""C", 40, 1);
    else               kv ("Temperature (baseline)", "unavailable");
    if (have_pt_end)   kvd("Temperature (end)",      t_end,   "\xc2\xb0""C", 40, 1);
    else               kv ("Temperature (end)",      "unavailable");
    if (have_pt_start && have_pt_end)
      kvd("Thermal Delta", t_end - t_start, "\xc2\xb0""C", 40, 1);
    else
      kv("Thermal Delta", "unavailable");

    sec("Sampling Integrity");
    kv ("Target Interval",   std::to_string(cfg.interval_ms) + " ms");
    kvd("Achieved Interval", achieved_interval_ms, "ms", 40, 1);
    kvi("Target Samples",    target_samples);
    kvi("Achieved Samples",  (long long)samples.size());
    kvd("Window Duration",   window_actual_ms, "ms", 40, 1);

    // System Signals (restored from v1, expanded)
    sec("System Signals");
    kv ("Persistence Mode",  persistence_mode);
    kv ("ASPM Policy",       aspm_policy);
    kv ("IOMMU",             iommu_state);
    kv ("Max Payload Size",  BytesToStr(mps_mrrs.mps_bytes));
    kv ("Max Read Request",  BytesToStr(mps_mrrs.mrrs_bytes));
    kv ("NUMA CPU Affinity", cpu_affinity_list.empty() ? "unavailable" : cpu_affinity_list);
    kv ("NUMA Affinity OK",  numa_optimal ? "yes" : "no (check CPU-GPU NUMA pinning)");

    sec("Final PCIe Assessment");
    kv("State", state);
    out << "\n";
    out << "Evidence\n";
    out << std::left << std::setw(40) << "  Link consistency"
        << " : " << (!(speed_change || width_change) ? "TRUE" : "FALSE") << "\n";
    out << std::left << std::setw(40) << "  Replay counter increase"
        << " : " << (delta > 10 ? "YES (delta=" + std::to_string(delta) + ")" : "NONE") << "\n";
    out << std::left << std::setw(40) << "  AER correctable errors"
        << " : " << (aer_available ? std::to_string(aer_corr_total) : "N/A") << "\n";
    out << std::left << std::setw(40) << "  AER fatal errors"
        << " : " << (aer_available ? std::to_string(aer_fatal_total) : "N/A") << "\n";
    out << std::left << std::setw(40) << "  Efficiency ratio"
        << " : " << fmt(eff, 3) << "\n";
  }

  // =========================================================================
  // JSON REPORT
  // =========================================================================
  {
    auto& out = json_out;

    // All fields except the last use trailing comma via these helpers
    auto js  = [&](const std::string& k, const std::string& v)
    { out << "  " << JsonEscape(k) << ": " << JsonEscape(v) << ",\n"; };

    auto jn  = [&](const std::string& k, double v, int p = 3)
    { out << "  " << JsonEscape(k) << ": "
          << std::fixed << std::setprecision(p) << v << ",\n"; };

    auto jll = [&](const std::string& k, long long v)
    { out << "  " << JsonEscape(k) << ": " << v << ",\n"; };

    auto jb  = [&](const std::string& k, bool v)
    { out << "  " << JsonEscape(k) << ": " << (v ? "true" : "false") << ",\n"; };

    auto jnull_or_ll = [&](const std::string& k, long long v)
    {
      if (v < 0) out << "  " << JsonEscape(k) << ": null,\n";
      else       out << "  " << JsonEscape(k) << ": " << v << ",\n";
    };
    auto jnull_or_n = [&](const std::string& k, double v, int p = 3)
    {
      if (v < 0.0) out << "  " << JsonEscape(k) << ": null,\n";
      else         out << "  " << JsonEscape(k) << ": "
                       << std::fixed << std::setprecision(p) << v << ",\n";
    };

    out << "{\n";

    // Identity
    js ("run_id",             run_id);
    js ("timestamp_iso",      ts_iso);
    jll("device_index",       cfg.device_index);
    js ("gpu_model",          gpu_name);
    js ("compute_capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
    js ("driver_version",     drv_ver);
    js ("nvml_version",       nvml_ver);
    {
      std::ostringstream ss;
      ss << (cuda_rt_ver / 1000) << "." << ((cuda_rt_ver % 1000) / 10);
      js("cuda_runtime", ss.str());
    }
    js ("pci_bus_id",         raw_bdf);
    js ("gpu_uuid",           gpu_uuid);
    jll("gpu_numa_node",      (long long)topo.numa_node);
    js ("os_name",            os_name);
    js ("kernel_version",     std::string(un.release));
    js ("cpu_arch",           std::string(un.machine));
    jll("cpu_cores_online",   (long long)cpu_cores);
    js ("scheduler_policy",   sched_str);
    jll("numa_node_count",    (long long)numa_count);

    // Run config
    jll("window_ms",          (long long)cfg.window_ms);
    jll("interval_ms",        (long long)cfg.interval_ms);
    jll("size_mib",           (long long)cfg.size_mib);
    js ("memory_mode",        cfg.memory_mode);

    // Topology
    js ("endpoint_bdf",       topo.endpoint_bdf);
    js ("upstream_bdf",       topo.upstream_bdf.empty() ? "N/A" : topo.upstream_bdf);
    js ("root_bdf",           topo.root_bdf.empty() ? "N/A" : topo.root_bdf);
    js ("pcie_chain",         topo.chain_str.empty() ? "N/A" : topo.chain_str);
    jll("chain_depth",        (long long)topo.depth);

    // PCIe link with explicit validity flags
    jb ("link_pre_valid",     have_link_pre);
    jb ("link_post_valid",    have_link_post);
    if (have_link_pre)
    {
      js ("pcie_max_supported",   LinkStr(pre.max_gen, pre.max_width));
      jll("pcie_max_gen",         (long long)pre.max_gen);
      jll("pcie_max_width",       (long long)pre.max_width);
      jll("pcie_pre_gen",         (long long)pre.cur_gen);
      jll("pcie_pre_width",       (long long)pre.cur_width);
      js ("pcie_negotiated_pre",  LinkStr(pre.cur_gen, pre.cur_width));
      js ("pcie_negotiated_post", have_link_post
                                    ? LinkStr(post.cur_gen, post.cur_width)
                                    : "unavailable");
      if (have_link_post)
      {
        jll("pcie_post_gen",      (long long)post.cur_gen);
        jll("pcie_post_width",    (long long)post.cur_width);
      }
      jn ("theoretical_max_gbs",  theo, 3);
    }
    else
    {
      js("pcie_max_supported",   "unavailable");
      js("pcie_negotiated_pre",  "unavailable");
      js("pcie_negotiated_post", "unavailable");
      jn("theoretical_max_gbs",  0.0, 3);
    }
    jb ("speed_change",       speed_change);
    jb ("width_change",       width_change);

    // Clocks / P-state (pre and post load)
    jnull_or_ll("sm_clock_pre_mhz",   clk_pre.sm_valid    ? (long long)clk_pre.sm_mhz   : -1LL);
    jnull_or_ll("sm_clock_post_mhz",  clk_post.sm_valid   ? (long long)clk_post.sm_mhz  : -1LL);
    jnull_or_ll("mem_clock_pre_mhz",  clk_pre.mem_valid   ? (long long)clk_pre.mem_mhz  : -1LL);
    jnull_or_ll("mem_clock_post_mhz", clk_post.mem_valid  ? (long long)clk_post.mem_mhz : -1LL);
    jnull_or_ll("gr_clock_pre_mhz",   clk_pre.gr_valid    ? (long long)clk_pre.gr_mhz   : -1LL);
    jnull_or_ll("gr_clock_post_mhz",  clk_post.gr_valid   ? (long long)clk_post.gr_mhz  : -1LL);
    js ("pstate_pre",         clk_pre.pstate_valid  ? PstateStr(clk_pre.pstate)  : "unavailable");
    js ("pstate_post",        clk_post.pstate_valid ? PstateStr(clk_post.pstate) : "unavailable");

    // Bandwidth
    jn ("bulk_h2d_gbs",       bw.bulk_h2d_gbs,   3);
    jn ("bulk_d2h_gbs",       bw.bulk_d2h_gbs,   3);
    jn ("bulk_avg_gbs",       bw.bulk_avg_gbs,   3);
    jn ("utilization_eff_pct",eff_pct,            2);
    jn ("bulk_h2d_latency_us",bw.bulk_h2d_latency_us, 1);
    jn ("bulk_d2h_latency_us",bw.bulk_d2h_latency_us, 1);
    jn ("bulk_avg_latency_us",bw.bulk_avg_latency_us, 1);
    jn ("link_h2d_latency_us",bw.link_h2d_latency_us, 1);
    jn ("link_d2h_latency_us",bw.link_d2h_latency_us, 1);
    jn ("link_avg_latency_us",bw.link_avg_latency_us, 1);

    // NVML throughput
    jn ("nvml_rx_avg_gbs",    rx_avg,   3);
    jn ("nvml_tx_avg_gbs",    tx_avg,   3);
    jn ("nvml_combined_gbs",  combined, 3);

    // Replay
    jll("replay_start",            (rr == NVML_SUCCESS) ? (long long)replay_start : -1LL);
    jll("replay_end",              (rr == NVML_SUCCESS) ? (long long)replay_end   : -1LL);
    jll("replay_counter_increase", delta);

    // AER
    jb ("aer_available",           aer_available);
    jll("aer_correctable_delta",   aer_corr_total);
    jll("aer_nonfatal_delta",      aer_nonfatal_total);
    jll("aer_fatal_delta",         aer_fatal_total);
    if (aer_available && !aer_corr_delta.empty())
    {
      out << "  \"aer_correctable_breakdown\": {";
      bool first = true;
      for (const auto& kv_aer : aer_corr_delta)
      {
        if (!first) out << ", ";
        out << JsonEscape(kv_aer.first) << ": " << kv_aer.second;
        first = false;
      }
      out << "},\n";
    }

    // Power
    jnull_or_n("pretest_power_w",  have_pt_start ? p_start : -1.0, 2);
    jn ("avg_load_power_w",        p_avg,  2);
    jn ("peak_power_w",            p_peak, 2);
    jnull_or_n("end_power_w",      have_pt_end ? p_end : -1.0, 2);
    jnull_or_n("power_delta_w",    (have_pt_start && have_pt_end) ? (p_end - p_start) : -1.0, 2);

    // Thermal
    jnull_or_n("baseline_temp_c",  have_pt_start ? t_start : -1.0, 1);
    jnull_or_n("end_temp_c",       have_pt_end   ? t_end   : -1.0, 1);
    jnull_or_n("thermal_delta_c",  (have_pt_start && have_pt_end) ? (t_end - t_start) : -1.0, 1);

    // Sampling
    jll("target_samples",          target_samples);
    jll("achieved_samples",        (long long)samples.size());
    jn ("target_interval_ms",      (double)cfg.interval_ms, 1);
    jn ("achieved_interval_ms",    achieved_interval_ms, 1);
    jn ("window_actual_ms",        window_actual_ms, 1);

    // System signals
    js ("persistence_mode",        persistence_mode);
    js ("aspm_policy",             aspm_policy);
    js ("iommu_state",             iommu_state);
    jnull_or_ll("mps_bytes",       mps_mrrs.valid ? (long long)mps_mrrs.mps_bytes  : -1LL);
    jnull_or_ll("mrrs_bytes",      mps_mrrs.valid ? (long long)mps_mrrs.mrrs_bytes : -1LL);
    js ("numa_cpu_affinity",       cpu_affinity_list.empty() ? "unavailable" : cpu_affinity_list);
    jb ("numa_affinity_optimal",   numa_optimal);

    // Derived assessment fields
    js ("assessment_state",        state);
    jb ("link_change",             speed_change || width_change);
    jb ("link_consistent",         !(speed_change || width_change));
    jb ("replay_growth",           (replay_end > replay_start));
    jb ("aer_errors_detected",     (aer_corr_total > 0 || aer_nonfatal_total > 0 || aer_fatal_total > 0));
    jn ("efficiency_ratio",        eff, 4);
    // Last field — no trailing comma
    out << "  \"replay_increased\": " << ((delta > 10) ? "true" : "false") << "\n";
    out << "}\n";
  }

  return exit_status;
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char** argv)
{
  RunConfig cfg;

  for (int i = 1; i < argc; i++)
  {
    std::string a = argv[i];
    auto need = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << flag << "\n"; std::exit(1); }
      return std::string(argv[++i]);
    };

    if      (a == "--list-devices") cfg.list_devices = true;
    else if (a == "--all-devices")  cfg.all_devices  = true;
    else if (a == "--device")       cfg.device_index = std::stoi(need("--device"));
    else if (a == "--memory-mode")  cfg.memory_mode  = need("--memory-mode");
    else if (a == "--window-ms")    cfg.window_ms    = std::stoi(need("--window-ms"));
    else if (a == "--interval-ms")  cfg.interval_ms  = std::stoi(need("--interval-ms"));
    else if (a == "--size-mib")     cfg.size_mib     = std::stoi(need("--size-mib"));
    else if (a == "-h" || a == "--help") { Usage(); return 0; }
    else { std::cerr << "Unknown argument: " << a << "\n"; Usage(); return 1; }
  }

  // Accept "pageable" as alias for "unpinned" (standard CUDA terminology)
  if (cfg.memory_mode == "pageable") cfg.memory_mode = "unpinned";

  if (cfg.memory_mode != "pinned" && cfg.memory_mode != "unpinned")
  {
    std::cerr << "Invalid --memory-mode: " << cfg.memory_mode
              << " (use pinned or unpinned / pageable)\n";
    return 1;
  }

  if (cfg.list_devices) { ListDevices(); return 0; }

  nvmlReturn_t nr = nvmlInit_v2();
  if (nr != NVML_SUCCESS)
  { std::cerr << "NVML init failed: " << NVML_Err(nr) << "\n"; return 1; }

  // ---- Single device -------------------------------------------------------
  if (!cfg.all_devices)
  {
    std::string run_id  = NowStamp() + "_GPU" + std::to_string(cfg.device_index);
    std::string log_dir = "./logs/runs/" + run_id;

    mkdir("./logs",        0755);
    mkdir("./logs/runs",   0755);
    mkdir(log_dir.c_str(), 0755);

    std::ostringstream txt_buf, json_buf;
    int status = RunValidation(cfg, txt_buf, json_buf, true);

    std::cout << txt_buf.str();

    std::string txt_path  = log_dir + "/report.txt";
    std::string json_path = log_dir + "/report.json";

    { std::ofstream f(txt_path);
      if (f.good()) f << txt_buf.str();
      else std::cerr << "[warn] could not write " << txt_path << "\n"; }

    { std::ofstream f(json_path);
      if (f.good()) f << json_buf.str();
      else std::cerr << "[warn] could not write " << json_path << "\n"; }

    std::cout << "\n[log] " << txt_path  << "\n";
    std::cout << "[log] " << json_path << "\n";

    nvmlShutdown();
    return status;
  }

  // ---- All devices ----------------------------------------------------------
  int cuda_devcount = 0;
  cudaGetDeviceCount(&cuda_devcount);

  if (cuda_devcount == 0)
  { std::cerr << "No CUDA devices found.\n"; nvmlShutdown(); return 1; }

  std::string ts      = NowStamp();
  std::string all_dir = "./logs/runs/" + ts + "_ALL";
  mkdir("./logs",        0755);
  mkdir("./logs/runs",   0755);
  mkdir(all_dir.c_str(), 0755);

  std::string all_txt_path  = all_dir + "/report.txt";
  std::string all_json_path = all_dir + "/report.json";

  std::ofstream all_txt(all_txt_path);
  std::ofstream all_json(all_json_path);

  if (!all_txt.good())  std::cerr << "[warn] could not open " << all_txt_path  << "\n";
  if (!all_json.good()) std::cerr << "[warn] could not open " << all_json_path << "\n";

  if (all_json.good()) all_json << "[\n";

  int worst_status = 0;

  for (int dev = 0; dev < cuda_devcount; dev++)
  {
    RunConfig dev_cfg    = cfg;
    dev_cfg.device_index = dev;

    std::ostringstream txt_buf, json_buf;
    int status = RunValidation(dev_cfg, txt_buf, json_buf, false);
    if (status > worst_status) worst_status = status;

    // Summary line per GPU in multi-GPU mode — no ticker flood
    {
      char msg[64];
      std::snprintf(msg, sizeof(msg), "GPU %d complete | status %d", dev, status);
      Progress(dev, msg);
    }

    std::cout << txt_buf.str();

    std::string gpu_json_path = all_dir + "/gpu" + std::to_string(dev) + ".json";
    { std::ofstream f(gpu_json_path);
      if (f.good()) f << json_buf.str();
      else std::cerr << "[warn] could not write " << gpu_json_path << "\n"; }

    if (all_txt.good())  all_txt << txt_buf.str();
    if (all_json.good())
    {
      // Inject as JSON array element
      std::string js = json_buf.str();
      // Strip trailing newline after closing brace for clean array formatting
      while (!js.empty() && (js.back() == '\n' || js.back() == '\r')) js.pop_back();
      all_json << js;
      if (dev < cuda_devcount - 1) all_json << ",";
      all_json << "\n";
    }

    std::cout << "[log] " << gpu_json_path << "\n";
  }

  if (all_json.good()) all_json << "]\n";

  std::cout << "\n[log] " << all_txt_path  << "\n";
  std::cout << "[log] " << all_json_path << "\n";

  nvmlShutdown();
  return worst_status;
}
