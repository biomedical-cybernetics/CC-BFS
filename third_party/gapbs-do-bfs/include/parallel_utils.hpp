#pragma once

#include <cstddef>
#include <thread>
#include <vector>
#include <set>
#include <iostream>
#include "omp.h"

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#include <fstream>
#include <string>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

size_t GetLogicalProcessorCount() noexcept {
    return std::thread::hardware_concurrency();
}

size_t GetPhysicalCoreCount() noexcept {
#ifdef _WIN32
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);

    if (length == 0)
        return GetLogicalProcessorCount();

    std::vector<char> buffer(length);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length))
        return GetLogicalProcessorCount();

    size_t physicalCoreCount = 0;
    auto ptr = buffer.data();
    while (ptr < buffer.data() + length) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);
        if (info->Relationship == RelationProcessorCore) {
            physicalCoreCount++;
        }
        ptr += info->Size;
    }
    return physicalCoreCount;
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo)
        return GetLogicalProcessorCount();

    std::set<std::pair<int,int>> cores;   // (physical id , core id)
    int phys = -1, core = -1;
    std::string line;

    while (std::getline(cpuinfo, line)) {
        if (line.rfind("physical id", 0) == 0) {
            phys = std::strtol(line.substr(line.find(':') + 1).c_str(), nullptr, 10);
        } else if (line.rfind("core id", 0) == 0) {
            core = std::strtol(line.substr(line.find(':') + 1).c_str(), nullptr, 10);
        } else if (line.empty()) {                // end of one logical-CPU stanza
            if (phys >= 0 && core >= 0)
                cores.emplace(phys, core);
            phys = core = -1;
        }
    }
    // catch the last stanza (file may not end with a blank line)
    if (phys >= 0 && core >= 0)
        cores.emplace(phys, core);

    /* Fallbacks:
       – some embedded / VM kernels omit physical/core ids;
       – older single-socket systems have no 'physical id'.
    */
    return cores.empty() ? GetLogicalProcessorCount() : cores.size();
#elif defined(__APPLE__)
    // macOS implementation
    int physicalCores;
    size_t len = sizeof(physicalCores);
    if (sysctlbyname("hw.physicalcpu", &physicalCores, &len, nullptr, 0) == 0)
        return physicalCores;
    return GetLogicalProcessorCount();

#else
    // Fallback to logical processors if platform-specific code isn't available
    return GetLogicalProcessorCount();
#endif
}



inline void setupOpenMP(bool enable_parallel, int64_t numThreads) {
    if (enable_parallel) {
        /* setup OpenMP */
        int64_t maxThreads = GetPhysicalCoreCount();
        if (numThreads > maxThreads) {
            std::string errorMsg = "Requested number of threads " + std::to_string(numThreads) +
                                   " exceeds maximum available threads " + std::to_string(maxThreads);
            throw std::runtime_error(errorMsg);
        }
        omp_set_dynamic(0);
        omp_set_num_threads(numThreads);
        std::cout << "Specified thread num " << numThreads << "/" << maxThreads << std::endl;
    }
}