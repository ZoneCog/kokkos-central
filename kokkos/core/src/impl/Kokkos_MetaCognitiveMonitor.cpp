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

#include <impl/Kokkos_MetaCognitiveMonitor.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace Kokkos {
namespace Tools {
namespace Experimental {

// MetaCognitiveAgent Implementation

MetaCognitiveAgent::MetaCognitiveAgent(uint32_t id, const std::string& name)
  : state_(std::make_unique<AgentState>(id, name)) {}

void MetaCognitiveAgent::updateResourceMetrics(const ResourceMetrics& metrics) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  // Update atomic metrics
  state_->metrics.memory_usage.store(metrics.memory_usage.load());
  state_->metrics.kernel_executions.store(metrics.kernel_executions.load());
  state_->metrics.parallel_regions.store(metrics.parallel_regions.load());
  state_->metrics.execution_time.store(metrics.execution_time.load());
  state_->metrics.active_threads.store(metrics.active_threads.load());
  state_->metrics.last_update = std::chrono::high_resolution_clock::now();
}

void MetaCognitiveAgent::recordKernelExecution(uint64_t kernel_id, double execution_time) {
  state_->metrics.kernel_executions.fetch_add(1);
  
  // Update cumulative execution time
  double current_time = state_->metrics.execution_time.load();
  while (!state_->metrics.execution_time.compare_exchange_weak(current_time, current_time + execution_time)) {
    // Retry until successful
  }
  
  // Trigger self-assessment if needed
  if (state_->metrics.kernel_executions.load() % 100 == 0) {
    performSelfAssessment();
  }
}

void MetaCognitiveAgent::recordMemoryUsage(uint64_t bytes) {
  state_->metrics.memory_usage.store(bytes);
}

void MetaCognitiveAgent::recordParallelRegion(uint32_t threads) {
  state_->metrics.parallel_regions.fetch_add(1);
  state_->metrics.active_threads.store(threads);
}

double MetaCognitiveAgent::performSelfAssessment() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  auto now = std::chrono::high_resolution_clock::now();
  auto time_since_creation = std::chrono::duration_cast<std::chrono::seconds>(
    now - state_->creation_time).count();
  
  if (time_since_creation == 0) time_since_creation = 1; // Avoid division by zero
  
  // Calculate performance metrics
  double kernels_per_second = static_cast<double>(state_->metrics.kernel_executions.load()) / time_since_creation;
  double avg_execution_time = state_->metrics.execution_time.load() / 
                              std::max(1UL, state_->metrics.kernel_executions.load());
  double memory_efficiency = std::min(1.0, 1.0 / (state_->metrics.memory_usage.load() / 1024.0 / 1024.0 + 1.0));
  
  // Compute composite performance score
  double performance_score = (kernels_per_second * 0.4 + 
                             (1.0 / (avg_execution_time + 0.001)) * 0.3 + 
                             memory_efficiency * 0.3);
  
  // Apply learning rate for adaptive adjustment
  double current_score = state_->performance_score.load();
  double new_score = current_score + learning_rate_ * (performance_score - current_score);
  state_->performance_score.store(new_score);
  
  return new_score;
}

void MetaCognitiveAgent::adaptBehavior(const std::vector<AdaptiveFeedback>& global_feedback) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  for (const auto& feedback : global_feedback) {
    if (feedback.confidence_score > adaptation_threshold_) {
      // Apply high-confidence feedback
      switch (feedback.type) {
        case AdaptiveFeedback::FeedbackType::PERFORMANCE_OPTIMIZATION:
          learning_rate_ = std::min(0.5, learning_rate_ * 1.1);
          break;
        case AdaptiveFeedback::FeedbackType::RESOURCE_REBALANCING:
          // Adjust resource monitoring sensitivity
          break;
        case AdaptiveFeedback::FeedbackType::EXECUTION_TUNING:
          adaptation_threshold_ = std::max(0.5, adaptation_threshold_ * 0.95);
          break;
        case AdaptiveFeedback::FeedbackType::MEMORY_OPTIMIZATION:
          // Implement memory-specific adaptations
          break;
      }
      
      feedback_history_.push_back(feedback);
      if (feedback_history_.size() > 100) {
        feedback_history_.erase(feedback_history_.begin());
      }
    }
  }
}

std::vector<AdaptiveFeedback> MetaCognitiveAgent::generateAdaptiveFeedback() {
  std::vector<AdaptiveFeedback> feedback;
  
  double current_score = state_->performance_score.load();
  
  // Generate performance-based feedback
  if (current_score < 0.5) {
    feedback.emplace_back(AdaptiveFeedback::FeedbackType::PERFORMANCE_OPTIMIZATION,
                         0.8,
                         "Low performance detected - recommend kernel optimization");
  }
  
  // Memory usage feedback
  uint64_t memory_mb = state_->metrics.memory_usage.load() / (1024 * 1024);
  if (memory_mb > 1000) { // > 1GB
    feedback.emplace_back(AdaptiveFeedback::FeedbackType::MEMORY_OPTIMIZATION,
                         0.9,
                         "High memory usage detected - recommend memory optimization");
  }
  
  return feedback;
}

ResourceMetrics MetaCognitiveAgent::getCurrentMetrics() const {
  ResourceMetrics metrics;
  metrics.memory_usage.store(state_->metrics.memory_usage.load());
  metrics.kernel_executions.store(state_->metrics.kernel_executions.load());
  metrics.parallel_regions.store(state_->metrics.parallel_regions.load());
  metrics.execution_time.store(state_->metrics.execution_time.load());
  metrics.active_threads.store(state_->metrics.active_threads.load());
  metrics.last_update = state_->metrics.last_update;
  return metrics;
}

void MetaCognitiveAgent::recursivelyAssessSubsystems() {
  // Perform recursive assessment of subsystems
  double base_score = performSelfAssessment();
  
  // Assess sub-components recursively
  std::vector<double> subsystem_scores;
  
  // Memory subsystem assessment
  double memory_score = 1.0 - (state_->metrics.memory_usage.load() / (1024.0 * 1024.0 * 1024.0)); // Normalize to GB
  subsystem_scores.push_back(std::max(0.0, std::min(1.0, memory_score)));
  
  // Execution subsystem assessment
  double execution_score = 1.0 / (state_->metrics.execution_time.load() / 
                                 std::max(1UL, state_->metrics.kernel_executions.load()) + 0.001);
  subsystem_scores.push_back(std::max(0.0, std::min(1.0, execution_score)));
  
  // Compute recursive score
  double recursive_score = base_score;
  for (double score : subsystem_scores) {
    recursive_score *= (0.5 + 0.5 * score); // Weighted combination
  }
  
  state_->performance_score.store(recursive_score);
}

// GlobalIntrospector Implementation

GlobalIntrospector::GlobalIntrospector() 
  : last_optimization_(std::chrono::high_resolution_clock::now()) {}

uint32_t GlobalIntrospector::registerAgent(const std::string& name) {
  std::lock_guard<std::mutex> lock(global_mutex_);
  
  uint32_t agent_id = next_agent_id_.fetch_add(1);
  agents_[agent_id] = std::make_unique<MetaCognitiveAgent>(agent_id, name);
  
  return agent_id;
}

void GlobalIntrospector::unregisterAgent(uint32_t agent_id) {
  std::lock_guard<std::mutex> lock(global_mutex_);
  agents_.erase(agent_id);
}

MetaCognitiveAgent* GlobalIntrospector::getAgent(uint32_t agent_id) {
  std::lock_guard<std::mutex> lock(global_mutex_);
  auto it = agents_.find(agent_id);
  return (it != agents_.end()) ? it->second.get() : nullptr;
}

ResourceMetrics GlobalIntrospector::aggregateAgentStates() {
  std::lock_guard<std::mutex> lock(global_mutex_);
  
  ResourceMetrics aggregated;
  
  for (const auto& [id, agent] : agents_) {
    auto metrics = agent->getCurrentMetrics();
    aggregated.memory_usage.fetch_add(metrics.memory_usage.load());
    aggregated.kernel_executions.fetch_add(metrics.kernel_executions.load());
    aggregated.parallel_regions.fetch_add(metrics.parallel_regions.load());
    
    // For execution_time (double), we need to manually handle atomic addition
    double current_time = aggregated.execution_time.load();
    double new_time = current_time + metrics.execution_time.load();
    while (!aggregated.execution_time.compare_exchange_weak(current_time, new_time)) {
      new_time = current_time + metrics.execution_time.load();
    }
    
    // For active threads, take the maximum across agents
    uint32_t current_max = aggregated.active_threads.load();
    uint32_t agent_threads = metrics.active_threads.load();
    while (agent_threads > current_max && 
           !aggregated.active_threads.compare_exchange_weak(current_max, agent_threads)) {
      // Retry until successful
    }
  }
  
  aggregated_metrics_ = std::move(aggregated);
  return aggregated;
}

void GlobalIntrospector::updateSystemMetrics() {
  aggregateAgentStates();
}

std::vector<AdaptiveFeedback> GlobalIntrospector::generateSystemOptimization() {
  std::vector<AdaptiveFeedback> system_feedback;
  
  updateSystemMetrics();
  double system_efficiency = calculateSystemEfficiency();
  
  if (system_efficiency < system_efficiency_target_) {
    system_feedback.emplace_back(
      AdaptiveFeedback::FeedbackType::PERFORMANCE_OPTIMIZATION,
      0.9,
      "System efficiency below target - recommend global optimization"
    );
  }
  
  // Check for resource imbalances
  uint64_t total_memory = aggregated_metrics_.memory_usage.load();
  size_t agent_count = agents_.size();
  
  if (agent_count > 0) {
    uint64_t avg_memory_per_agent = total_memory / agent_count;
    
    for (const auto& [id, agent] : agents_) {
      auto metrics = agent->getCurrentMetrics();
      if (metrics.memory_usage.load() > avg_memory_per_agent * 2) {
        system_feedback.emplace_back(
          AdaptiveFeedback::FeedbackType::RESOURCE_REBALANCING,
          0.8,
          "Agent " + std::to_string(id) + " has high memory usage - recommend rebalancing"
        );
      }
    }
  }
  
  return system_feedback;
}

void GlobalIntrospector::applyAdaptiveFeedback() {
  auto system_feedback = generateSystemOptimization();
  
  std::lock_guard<std::mutex> lock(global_mutex_);
  
  // Apply feedback to all agents
  for (const auto& [id, agent] : agents_) {
    agent->adaptBehavior(system_feedback);
  }
  
  // Store global feedback
  global_feedback_.insert(global_feedback_.end(), system_feedback.begin(), system_feedback.end());
  
  // Limit feedback history
  if (global_feedback_.size() > 1000) {
    global_feedback_.erase(global_feedback_.begin(), global_feedback_.begin() + 500);
  }
}

double GlobalIntrospector::calculateSystemEfficiency() {
  if (agents_.empty()) return 1.0;
  
  double total_score = 0.0;
  for (const auto& [id, agent] : agents_) {
    total_score += agent->getState().performance_score.load();
  }
  
  return total_score / agents_.size();
}

void GlobalIntrospector::performGlobalIntrospection() {
  updateSystemMetrics();
  applyAdaptiveFeedback();
  
  auto now = std::chrono::high_resolution_clock::now();
  last_optimization_ = now;
}

void GlobalIntrospector::recursiveSystemAssessment() {
  // Perform recursive assessment of the entire system
  std::lock_guard<std::mutex> lock(global_mutex_);
  
  // First level: assess individual agents recursively
  for (const auto& [id, agent] : agents_) {
    agent->recursivelyAssessSubsystems();
  }
  
  // Second level: assess system-wide patterns
  double system_coherence = 1.0;
  if (agents_.size() > 1) {
    std::vector<double> agent_scores;
    for (const auto& [id, agent] : agents_) {
      agent_scores.push_back(agent->getState().performance_score.load());
    }
    
    // Calculate coefficient of variation as a measure of system coherence
    double mean = std::accumulate(agent_scores.begin(), agent_scores.end(), 0.0) / agent_scores.size();
    double variance = 0.0;
    for (double score : agent_scores) {
      variance += (score - mean) * (score - mean);
    }
    variance /= agent_scores.size();
    double std_dev = std::sqrt(variance);
    
    if (mean > 0) {
      double cv = std_dev / mean;
      system_coherence = std::max(0.0, 1.0 - cv); // Higher coherence = lower variation
    }
  }
  
  // Third level: generate recursive feedback based on system coherence
  if (system_coherence < 0.8) {
    global_feedback_.emplace_back(
      AdaptiveFeedback::FeedbackType::RESOURCE_REBALANCING,
      0.9,
      "Low system coherence detected - recommend agent synchronization"
    );
  }
}

size_t GlobalIntrospector::getActiveAgentCount() const {
  std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(global_mutex_));
  return std::count_if(agents_.begin(), agents_.end(),
                      [](const auto& pair) { return pair.second->getState().is_active.load(); });
}

// Singleton implementation
GlobalIntrospector& getGlobalIntrospector() {
  static GlobalIntrospector instance;
  return instance;
}

// Initialization and callback functions
void initializeMetaCognitiveMonitoring() {
  // Initialize the global introspector
  auto& introspector = getGlobalIntrospector();
  
  // Register with existing Kokkos tools infrastructure if available
  // This would integrate with the existing profiling callbacks
}

void finalizeMetaCognitiveMonitoring() {
  auto& introspector = getGlobalIntrospector();
  introspector.performGlobalIntrospection();
}

// Callback implementations for integration
static std::unordered_map<uint64_t, uint32_t> kernel_to_agent;
static std::mutex callback_mutex;

void metaCognitiveBeginCallback(const char* name, uint32_t devid, uint64_t* kID) {
  std::lock_guard<std::mutex> lock(callback_mutex);
  
  auto& introspector = getGlobalIntrospector();
  uint32_t agent_id = introspector.registerAgent(std::string(name));
  
  if (kID) {
    kernel_to_agent[*kID] = agent_id;
  }
}

void metaCognitiveEndCallback(uint64_t kID) {
  std::lock_guard<std::mutex> lock(callback_mutex);
  
  auto it = kernel_to_agent.find(kID);
  if (it != kernel_to_agent.end()) {
    auto& introspector = getGlobalIntrospector();
    auto* agent = introspector.getAgent(it->second);
    if (agent) {
      agent->recordKernelExecution(kID, 0.0); // Time would need to be tracked separately
    }
    kernel_to_agent.erase(it);
  }
}

void metaCognitiveRegionCallback(const char* name) {
  auto& introspector = getGlobalIntrospector();
  introspector.performGlobalIntrospection();
}

} // namespace Experimental
} // namespace Tools
} // namespace Kokkos