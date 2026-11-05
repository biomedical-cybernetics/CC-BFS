//
// Created by Junda on 2026/3/19.
//

#ifndef CPP_PROCESS_MEMORY_UTILS_H
#define CPP_PROCESS_MEMORY_UTILS_H

#include <cstdint>
#include <iostream>
#include <limits>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include <windows.h>
#include <psapi.h>
#endif

#ifdef _WIN32
struct ProcessMemorySnapshot {
    uint64_t working_set_bytes = 0;
    uint64_t private_bytes = 0;
    bool valid = false;
};

inline ProcessMemorySnapshot GetCurrentProcessMemorySnapshot() {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    ProcessMemorySnapshot snapshot{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(),
                                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                                sizeof(pmc))) {
        snapshot.working_set_bytes = static_cast<uint64_t>(pmc.WorkingSetSize);
        snapshot.private_bytes = static_cast<uint64_t>(pmc.PrivateUsage);
        snapshot.valid = true;
    }
    return snapshot;
}
#else
struct ProcessMemorySnapshot {
    uint64_t working_set_bytes = 0;
    uint64_t private_bytes = 0;
    bool valid = false;
};

inline ProcessMemorySnapshot GetCurrentProcessMemorySnapshot() {
    return ProcessMemorySnapshot{};
}
#endif

struct ProcessMemoryPeakTracker {
    ProcessMemorySnapshot baseline{};
    ProcessMemorySnapshot peak{};
    bool started = false;

    void CaptureBaseline() {
        baseline = GetCurrentProcessMemorySnapshot();
        peak = baseline;
        started = baseline.valid;
    }

    void UpdatePeak() {
        const ProcessMemorySnapshot current = GetCurrentProcessMemorySnapshot();
        if (!current.valid) {
            return;
        }
        if (!started) {
            baseline = current;
            peak = current;
            started = true;
            return;
        }
        if (current.working_set_bytes > peak.working_set_bytes) {
            peak.working_set_bytes = current.working_set_bytes;
        }
        if (current.private_bytes > peak.private_bytes) {
            peak.private_bytes = current.private_bytes;
        }
        peak.valid = true;
    }

    uint64_t DeltaPeakWorkingSetBytes() const {
        return (peak.working_set_bytes >= baseline.working_set_bytes)
            ? (peak.working_set_bytes - baseline.working_set_bytes)
            : 0;
    }

    uint64_t DeltaPeakPrivateBytes() const {
        return (peak.private_bytes >= baseline.private_bytes)
            ? (peak.private_bytes - baseline.private_bytes)
            : 0;
    }
};

inline int64_t ClampUint64ToInt64(uint64_t value) {
    const uint64_t int64_max = static_cast<uint64_t>((std::numeric_limits<int64_t>::max)());
    return static_cast<int64_t>(value > int64_max ? int64_max : value);
}

inline double BytesToGiB(uint64_t bytes) {
    constexpr double bytes_per_gib = 1024.0 * 1024.0 * 1024.0;
    return static_cast<double>(bytes) / bytes_per_gib;
}

template <typename CountersType>
inline void FillProcessMemoryCounters(CountersType& counters, const ProcessMemoryPeakTracker& tracker) {
    if (!tracker.started) {
        return;
    }
    counters.process_baseline_working_set_bytes = ClampUint64ToInt64(tracker.baseline.working_set_bytes);
    counters.process_peak_working_set_bytes = ClampUint64ToInt64(tracker.peak.working_set_bytes);
    counters.process_delta_peak_working_set_bytes = ClampUint64ToInt64(tracker.DeltaPeakWorkingSetBytes());
    counters.process_baseline_private_bytes = ClampUint64ToInt64(tracker.baseline.private_bytes);
    counters.process_peak_private_bytes = ClampUint64ToInt64(tracker.peak.private_bytes);
    counters.process_delta_peak_private_bytes = ClampUint64ToInt64(tracker.DeltaPeakPrivateBytes());
}

inline ProcessMemoryPeakTracker*& ActiveProcessMemoryPeakTracker() {
    static ProcessMemoryPeakTracker* tracker = nullptr;
    return tracker;
}

inline void UpdateProcessMemoryPeakCheckpoint() {
    ProcessMemoryPeakTracker* tracker = ActiveProcessMemoryPeakTracker();
    if (tracker != nullptr) {
        tracker->UpdatePeak();
    }
}

inline void ResetProcessMemoryPeakBaseline() {
    ProcessMemoryPeakTracker* tracker = ActiveProcessMemoryPeakTracker();
    if (tracker != nullptr) {
        tracker->CaptureBaseline();
    }
}

struct ProcessMemoryTrackerScope {
    explicit ProcessMemoryTrackerScope(ProcessMemoryPeakTracker* tracker) {
        ActiveProcessMemoryPeakTracker() = tracker;
    }

    ~ProcessMemoryTrackerScope() {
        ActiveProcessMemoryPeakTracker() = nullptr;
    }
};

inline void PrintProcessMemorySummary() {
    ProcessMemoryPeakTracker* tracker = ActiveProcessMemoryPeakTracker();
    if (tracker == nullptr || !tracker->started) {
        return;
    }
    std::cout << "@@ Process memory (working set GiB): baseline="
              << BytesToGiB(tracker->baseline.working_set_bytes)
              << " peak=" << BytesToGiB(tracker->peak.working_set_bytes)
              << " delta_peak=" << BytesToGiB(tracker->DeltaPeakWorkingSetBytes())
              << std::endl;
    std::cout << "@@ Process memory (private GiB): baseline="
              << BytesToGiB(tracker->baseline.private_bytes)
              << " peak=" << BytesToGiB(tracker->peak.private_bytes)
              << " delta_peak=" << BytesToGiB(tracker->DeltaPeakPrivateBytes())
              << std::endl;
}

#endif // CPP_PROCESS_MEMORY_UTILS_H
