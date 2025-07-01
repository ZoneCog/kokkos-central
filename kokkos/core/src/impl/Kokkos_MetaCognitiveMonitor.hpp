//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_METACOGNITIVE_MONITOR_HPP
#define KOKKOS_IMPL_METACOGNITIVE_MONITOR_HPP

#include <Kokkos_Core_fwd.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Kokkos {
namespace Tools {
namespace Experimental {

// Forward declarations
class MetaCognitiveAgent;
class GlobalIntrospector;

/**
 * @brief Resource metrics tracked by monitoring agents
 */
struct ResourceMetrics {
  std::atomic<uint64_t> memory_usage{0};
  std::atomic<uint64_t> kernel_executions{0};
  std::atomic<uint64_t> parallel_regions{0};
  std::atomic<double> execution_time{0.0};
  std::atomic<uint32_t> active_threads{0};
  std::chrono::high_resolution_clock::time_point last_update;
  
  ResourceMetrics() : last_update(std::chrono::high_resolution_clock::now()) {}
  
  // Delete copy constructor and assignment to prevent issues with atomics
  ResourceMetrics(const ResourceMetrics&) = delete;
  ResourceMetrics& operator=(const ResourceMetrics&) = delete;
  
  // Allow move constructor and assignment
  ResourceMetrics(ResourceMetrics&& other) noexcept 
    : memory_usage{other.memory_usage.load()},
      kernel_executions{other.kernel_executions.load()},
      parallel_regions{other.parallel_regions.load()},
      execution_time{other.execution_time.load()},
      active_threads{other.active_threads.load()},
      last_update{other.last_update} {}
      
  ResourceMetrics& operator=(ResourceMetrics&& other) noexcept {
    if (this != &other) {
      memory_usage.store(other.memory_usage.load());
      kernel_executions.store(other.kernel_executions.load());
      parallel_regions.store(other.parallel_regions.load());
      execution_time.store(other.execution_time.load());
      active_threads.store(other.active_threads.load());
      last_update = other.last_update;
    }
    return *this;
  }
};

/**
 * @brief Agent state for meta-cognitive monitoring
 */
struct AgentState {
  uint32_t agent_id;
  std::string agent_name;
  ResourceMetrics metrics;
  std::atomic<double> performance_score{0.0};
  std::atomic<bool> is_active{true};
  std::chrono::high_resolution_clock::time_point creation_time;
  
  AgentState(uint32_t id, const std::string& name) 
    : agent_id(id), agent_name(name), 
      creation_time(std::chrono::high_resolution_clock::now()) {}
};

/**
 * @brief Adaptive feedback for system optimization
 */
struct AdaptiveFeedback {
  enum class FeedbackType {
    PERFORMANCE_OPTIMIZATION,
    RESOURCE_REBALANCING,
    EXECUTION_TUNING,
    MEMORY_OPTIMIZATION
  };
  
  FeedbackType type;
  double confidence_score;
  std::string recommendation;
  std::unordered_map<std::string, double> parameters;
  std::chrono::high_resolution_clock::time_point timestamp;
  
  AdaptiveFeedback(FeedbackType t, double confidence, const std::string& rec)
    : type(t), confidence_score(confidence), recommendation(rec),
      timestamp(std::chrono::high_resolution_clock::now()) {}
};

/**
 * @brief Meta-cognitive monitoring agent for individual execution contexts
 */
class MetaCognitiveAgent {
private:
  std::unique_ptr<AgentState> state_;
  std::vector<AdaptiveFeedback> feedback_history_;
  std::mutex state_mutex_;
  
  // Self-assessment parameters
  double learning_rate_ = 0.1;
  double adaptation_threshold_ = 0.8;
  
public:
  explicit MetaCognitiveAgent(uint32_t id, const std::string& name);
  ~MetaCognitiveAgent() = default;
  
  // Core monitoring functions
  void updateResourceMetrics(const ResourceMetrics& metrics);
  void recordKernelExecution(uint64_t kernel_id, double execution_time);
  void recordMemoryUsage(uint64_t bytes);
  void recordParallelRegion(uint32_t threads);
  
  // Self-assessment capabilities
  double performSelfAssessment();
  void adaptBehavior(const std::vector<AdaptiveFeedback>& global_feedback);
  
  // Feedback generation
  std::vector<AdaptiveFeedback> generateAdaptiveFeedback();
  
  // State access
  const AgentState& getState() const { return *state_; }
  ResourceMetrics getCurrentMetrics() const;
  
  // Recursive monitoring
  void recursivelyAssessSubsystems();
};

/**
 * @brief Global introspector for system-wide monitoring and optimization
 */
class GlobalIntrospector {
private:
  std::unordered_map<uint32_t, std::unique_ptr<MetaCognitiveAgent>> agents_;
  std::vector<AdaptiveFeedback> global_feedback_;
  std::mutex global_mutex_;
  
  // System-wide metrics
  ResourceMetrics aggregated_metrics_;
  std::atomic<uint32_t> next_agent_id_{0};
  
  // Optimization parameters
  double system_efficiency_target_ = 0.85;
  std::chrono::milliseconds feedback_interval_{1000};
  std::chrono::high_resolution_clock::time_point last_optimization_;
  
public:
  GlobalIntrospector();
  ~GlobalIntrospector() = default;
  
  // Agent management
  uint32_t registerAgent(const std::string& name);
  void unregisterAgent(uint32_t agent_id);
  MetaCognitiveAgent* getAgent(uint32_t agent_id);
  
  // Global aggregation
  ResourceMetrics aggregateAgentStates();
  void updateSystemMetrics();
  
  // System-wide optimization
  std::vector<AdaptiveFeedback> generateSystemOptimization();
  void applyAdaptiveFeedback();
  
  // Introspection capabilities
  double calculateSystemEfficiency();
  void performGlobalIntrospection();
  
  // Recursive self-assessment
  void recursiveSystemAssessment();
  
  // Access to aggregated data
  const std::vector<AdaptiveFeedback>& getGlobalFeedback() const { return global_feedback_; }
  size_t getActiveAgentCount() const;
};

/**
 * @brief Singleton access to global introspector
 */
GlobalIntrospector& getGlobalIntrospector();

/**
 * @brief Callback registration for meta-cognitive monitoring
 */
void initializeMetaCognitiveMonitoring();
void finalizeMetaCognitiveMonitoring();

// Callback functions for integration with existing Kokkos tools
void metaCognitiveBeginCallback(const char* name, uint32_t devid, uint64_t* kID);
void metaCognitiveEndCallback(uint64_t kID);
void metaCognitiveRegionCallback(const char* name);

} // namespace Experimental
} // namespace Tools
} // namespace Kokkos

#endif // KOKKOS_IMPL_METACOGNITIVE_MONITOR_HPP