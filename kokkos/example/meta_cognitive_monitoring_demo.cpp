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

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_MetaCognitiveMonitor.hpp>
#include <iostream>

/**
 * @brief Example demonstrating meta-cognitive monitoring agents for global introspection
 * 
 * This example shows how to use the meta-cognitive monitoring system to:
 * - Track resource usage across multiple execution contexts
 * - Perform adaptive feedback for system optimization
 * - Implement recursive self-assessment capabilities
 * - Aggregate agent states for global introspection
 */

void demonstrate_basic_monitoring() {
  std::cout << "\n=== Basic Meta-Cognitive Monitoring ===\n";
  
  // Get the global introspector
  auto& introspector = Kokkos::Tools::Experimental::getGlobalIntrospector();
  
  // Register monitoring agents for different execution contexts
  uint32_t compute_agent_id = introspector.registerAgent("ComputeIntensiveAgent");
  uint32_t memory_agent_id = introspector.registerAgent("MemoryIntensiveAgent");
  uint32_t balanced_agent_id = introspector.registerAgent("BalancedAgent");
  
  auto* compute_agent = introspector.getAgent(compute_agent_id);
  auto* memory_agent = introspector.getAgent(memory_agent_id);
  auto* balanced_agent = introspector.getAgent(balanced_agent_id);
  
  // Simulate different workload patterns
  std::cout << "Simulating compute-intensive workload...\n";
  for (int i = 0; i < 1000; ++i) {
    compute_agent->recordKernelExecution(i, 0.001); // Fast, many kernels
  }
  compute_agent->recordMemoryUsage(100 * 1024 * 1024); // 100MB
  
  std::cout << "Simulating memory-intensive workload...\n";
  for (int i = 0; i < 50; ++i) {
    memory_agent->recordKernelExecution(i, 0.1); // Slower, fewer kernels
  }
  memory_agent->recordMemoryUsage(2UL * 1024 * 1024 * 1024); // 2GB
  
  std::cout << "Simulating balanced workload...\n";
  for (int i = 0; i < 200; ++i) {
    balanced_agent->recordKernelExecution(i, 0.01); // Moderate
  }
  balanced_agent->recordMemoryUsage(500 * 1024 * 1024); // 500MB
  
  // Perform self-assessments
  double compute_score = compute_agent->performSelfAssessment();
  double memory_score = memory_agent->performSelfAssessment();
  double balanced_score = balanced_agent->performSelfAssessment();
  
  std::cout << "Agent Performance Scores:\n";
  std::cout << "  Compute Agent: " << compute_score << "\n";
  std::cout << "  Memory Agent: " << memory_score << "\n";
  std::cout << "  Balanced Agent: " << balanced_score << "\n";
  
  // Clean up
  introspector.unregisterAgent(compute_agent_id);
  introspector.unregisterAgent(memory_agent_id);
  introspector.unregisterAgent(balanced_agent_id);
}

void demonstrate_adaptive_feedback() {
  std::cout << "\n=== Adaptive Feedback System ===\n";
  
  auto& introspector = Kokkos::Tools::Experimental::getGlobalIntrospector();
  
  // Create agents with different performance characteristics
  uint32_t high_perf_id = introspector.registerAgent("HighPerformanceAgent");
  uint32_t low_perf_id = introspector.registerAgent("LowPerformanceAgent");
  
  auto* high_perf_agent = introspector.getAgent(high_perf_id);
  auto* low_perf_agent = introspector.getAgent(low_perf_id);
  
  // Configure high-performance agent
  for (int i = 0; i < 500; ++i) {
    high_perf_agent->recordKernelExecution(i, 0.002); // Very fast
  }
  high_perf_agent->recordMemoryUsage(200 * 1024 * 1024); // Moderate memory
  
  // Configure low-performance agent  
  for (int i = 0; i < 10; ++i) {
    low_perf_agent->recordKernelExecution(i, 0.5); // Slow execution
  }
  low_perf_agent->recordMemoryUsage(3UL * 1024 * 1024 * 1024); // High memory usage
  
  // Generate agent-specific feedback
  auto high_perf_feedback = high_perf_agent->generateAdaptiveFeedback();
  auto low_perf_feedback = low_perf_agent->generateAdaptiveFeedback();
  
  std::cout << "High-Performance Agent Feedback (" << high_perf_feedback.size() << " items):\n";
  for (const auto& feedback : high_perf_feedback) {
    std::cout << "  - " << feedback.recommendation << " (confidence: " << feedback.confidence_score << ")\n";
  }
  
  std::cout << "Low-Performance Agent Feedback (" << low_perf_feedback.size() << " items):\n";
  for (const auto& feedback : low_perf_feedback) {
    std::cout << "  - " << feedback.recommendation << " (confidence: " << feedback.confidence_score << ")\n";
  }
  
  // Generate system-wide optimization recommendations
  auto system_feedback = introspector.generateSystemOptimization();
  std::cout << "System-wide Optimization Recommendations (" << system_feedback.size() << " items):\n";
  for (const auto& feedback : system_feedback) {
    std::cout << "  - " << feedback.recommendation << " (confidence: " << feedback.confidence_score << ")\n";
  }
  
  // Apply adaptive feedback
  introspector.applyAdaptiveFeedback();
  std::cout << "Adaptive feedback applied to all agents.\n";
  
  // Clean up
  introspector.unregisterAgent(high_perf_id);
  introspector.unregisterAgent(low_perf_id);
}

void demonstrate_recursive_assessment() {
  std::cout << "\n=== Recursive Self-Assessment ===\n";
  
  auto& introspector = Kokkos::Tools::Experimental::getGlobalIntrospector();
  
  // Create a collection of agents for system-wide recursive assessment
  std::vector<uint32_t> agent_ids;
  for (int i = 0; i < 5; ++i) {
    agent_ids.push_back(introspector.registerAgent("RecursiveAgent" + std::to_string(i)));
  }
  
  // Configure agents with varying performance patterns
  for (size_t i = 0; i < agent_ids.size(); ++i) {
    auto* agent = introspector.getAgent(agent_ids[i]);
    
    // Create performance variation across agents
    int kernel_count = 100 + i * 50;
    double execution_time = 0.01 * (i + 1);
    uint64_t memory_usage = (i + 1) * 200 * 1024 * 1024; // 200MB * (i+1)
    
    for (int j = 0; j < kernel_count; ++j) {
      agent->recordKernelExecution(j, execution_time);
    }
    agent->recordMemoryUsage(memory_usage);
    agent->recordParallelRegion(4 + i * 2); // Varying thread counts
    
    // Initial self-assessment
    double initial_score = agent->performSelfAssessment();
    std::cout << "Agent " << i << " initial score: " << initial_score << "\n";
  }
  
  std::cout << "\nPerforming recursive system assessment...\n";
  
  // Perform recursive assessment at system level
  introspector.recursiveSystemAssessment();
  
  // Show updated scores after recursive assessment
  std::cout << "Post-recursive assessment scores:\n";
  for (size_t i = 0; i < agent_ids.size(); ++i) {
    auto* agent = introspector.getAgent(agent_ids[i]);
    double recursive_score = agent->getState().performance_score.load();
    std::cout << "Agent " << i << " recursive score: " << recursive_score << "\n";
  }
  
  // Calculate system efficiency
  double system_efficiency = introspector.calculateSystemEfficiency();
  std::cout << "Overall system efficiency: " << system_efficiency << "\n";
  
  // Clean up
  for (auto agent_id : agent_ids) {
    introspector.unregisterAgent(agent_id);
  }
}

void demonstrate_global_introspection() {
  std::cout << "\n=== Global Introspection ===\n";
  
  auto& introspector = Kokkos::Tools::Experimental::getGlobalIntrospector();
  
  // Create multiple agents to demonstrate global aggregation
  std::vector<uint32_t> agent_ids;
  const int num_agents = 10;
  
  for (int i = 0; i < num_agents; ++i) {
    agent_ids.push_back(introspector.registerAgent("GlobalAgent" + std::to_string(i)));
  }
  
  std::cout << "Created " << introspector.getActiveAgentCount() << " active agents.\n";
  
  // Simulate varied workloads across agents
  uint64_t total_kernels = 0;
  uint64_t total_memory = 0;
  
  for (size_t i = 0; i < agent_ids.size(); ++i) {
    auto* agent = introspector.getAgent(agent_ids[i]);
    
    int kernels = (i + 1) * 100;
    uint64_t memory = (i + 1) * 100 * 1024 * 1024; // 100MB * (i+1)
    
    for (int j = 0; j < kernels; ++j) {
      agent->recordKernelExecution(j, 0.01);
    }
    agent->recordMemoryUsage(memory);
    
    total_kernels += kernels;
    total_memory += memory;
  }
  
  // Perform global aggregation
  auto aggregated = introspector.aggregateAgentStates();
  
  std::cout << "Global Resource Aggregation:\n";
  std::cout << "  Total kernel executions: " << aggregated.kernel_executions.load() 
            << " (expected: " << total_kernels << ")\n";
  std::cout << "  Total memory usage: " << (aggregated.memory_usage.load() / (1024 * 1024)) 
            << " MB (expected: " << (total_memory / (1024 * 1024)) << " MB)\n";
  std::cout << "  Total execution time: " << aggregated.execution_time.load() << " seconds\n";
  
  // Perform comprehensive global introspection
  std::cout << "\nPerforming global introspection...\n";
  introspector.performGlobalIntrospection();
  
  // Display global feedback
  const auto& global_feedback = introspector.getGlobalFeedback();
  std::cout << "Global feedback items: " << global_feedback.size() << "\n";
  
  for (size_t i = 0; i < std::min(size_t(5), global_feedback.size()); ++i) {
    const auto& feedback = global_feedback[global_feedback.size() - 1 - i]; // Latest feedback
    std::cout << "  - " << feedback.recommendation << " (confidence: " << feedback.confidence_score << ")\n";
  }
  
  // Clean up
  for (auto agent_id : agent_ids) {
    introspector.unregisterAgent(agent_id);
  }
}

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  
  std::cout << "Meta-Cognitive Monitoring Agents for Global Introspection\n";
  std::cout << "========================================================\n";
  
  // Initialize meta-cognitive monitoring system
  Kokkos::Tools::Experimental::initializeMetaCognitiveMonitoring();
  
  try {
    // Demonstrate various aspects of the meta-cognitive monitoring system
    demonstrate_basic_monitoring();
    demonstrate_adaptive_feedback();
    demonstrate_recursive_assessment();
    demonstrate_global_introspection();
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Meta-cognitive monitoring system demonstration completed successfully!\n";
    std::cout << "Key capabilities demonstrated:\n";
    std::cout << "✓ Agent state aggregation\n";
    std::cout << "✓ Resource usage tracking\n";
    std::cout << "✓ Adaptive feedback for system-wide optimization\n";
    std::cout << "✓ Recursive self-assessment\n";
    std::cout << "✓ Global introspection and monitoring\n";
    
  } catch (const std::exception& e) {
    std::cerr << "Error during demonstration: " << e.what() << std::endl;
    return 1;
  }
  
  // Finalize meta-cognitive monitoring system
  Kokkos::Tools::Experimental::finalizeMetaCognitiveMonitoring();
  
  return 0;
}