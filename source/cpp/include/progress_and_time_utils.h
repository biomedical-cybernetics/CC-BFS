// Utility helpers for console progress and local timestamps.
// Keep this header small and self-contained (header-only).

#ifndef PROGRESS_AND_TIME_UTILS_H
#define PROGRESS_AND_TIME_UTILS_H

#include <cstdint>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

inline void UpdateProgressPercent(const int64_t processed,
                                  const int64_t total,
                                  const int update_interval_pct,
                                  int &next_report_pct) {
    if (total <= 0) {
        return;
    }
    if (update_interval_pct <= 0) {
        return;
    }
    int current_pct = static_cast<int>(100 * processed / total);
    while (current_pct >= next_report_pct && next_report_pct <= 100) {
        // Overwrite the previous "Progress: xxx%" (fixed width, 13 chars)
        std::cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b"
                  << "Progress:" << std::setw(3) << next_report_pct << "%"
                  << std::flush;
        if (next_report_pct == 100) {
            std::cout << std::endl;
        }
        next_report_pct += update_interval_pct;
    }
}

inline std::string NowLocalDateTimeString() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_local{};
#ifdef _WIN32
    localtime_s(&tm_local, &t);
#else
    tm_local = *std::localtime(&t);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_local, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

#endif // PROGRESS_AND_TIME_UTILS_H

