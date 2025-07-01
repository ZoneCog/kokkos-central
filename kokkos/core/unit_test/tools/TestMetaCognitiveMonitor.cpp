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

#include <gtest/gtest.h>
#include <impl/Kokkos_MetaCognitiveMonitor.hpp>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <thread>

namespace Test {

using namespace Kokkos::Tools::Experimental;

class MetaCognitiveMonitorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Kokkos if not already initialized
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
    }
    initializeMetaCognitiveMonitoring();
  }
  
  void TearDown() override {
    finalizeMetaCognitiveMonitoring();
  }
};

// Test MetaCognitiveAgent basic functionality
TEST_F(MetaCognitiveMonitorTest, AgentBasicFunctionality) {
  MetaCognitiveAgent agent(1, "TestAgent");
  
  // Test initial state
  const auto& state = agent.getState();
  EXPECT_EQ(state.agent_id, 1);
  EXPECT_EQ(state.agent_name, "TestAgent");
  EXPECT_TRUE(state.is_active.load());
  
  // Test resource recording
  agent.recordKernelExecution(100, 1.5);
  agent.recordMemoryUsage(1024 * 1024); // 1MB
  agent.recordParallelRegion(8);
  
  auto metrics = agent.getCurrentMetrics();
  EXPECT_EQ(metrics.kernel_executions.load(), 1);
  EXPECT_EQ(metrics.memory_usage.load(), 1024 * 1024);
  EXPECT_EQ(metrics.active_threads.load(), 8);
  EXPECT_GT(metrics.execution_time.load(), 0);
}

TEST_F(MetaCognitiveMonitorTest, AgentSelfAssessment) {
  MetaCognitiveAgent agent(2, "SelfAssessmentAgent");
  
  // Record some activity
  for (int i = 0; i < 10; ++i) {
    agent.recordKernelExecution(100 + i, 0.1 * (i + 1));
    agent.recordMemoryUsage(1024 * 1024 * (i + 1));
  }
  
  // Perform self-assessment
  double score1 = agent.performSelfAssessment();
  EXPECT_GE(score1, 0.0);
  
  // Add more activity and assess again
  for (int i = 0; i < 20; ++i) {
    agent.recordKernelExecution(200 + i, 0.05); // Faster execution
  }
  
  double score2 = agent.performSelfAssessment();
  EXPECT_GE(score2, 0.0);
}

TEST_F(MetaCognitiveMonitorTest, AgentAdaptiveFeedback) {
  MetaCognitiveAgent agent(3, "AdaptiveAgent");
  
  // Generate some metrics to trigger feedback
  agent.recordMemoryUsage(2UL * 1024 * 1024 * 1024); // 2GB - high usage
  
  auto feedback = agent.generateAdaptiveFeedback();
  
  // Should generate memory optimization feedback
  bool found_memory_feedback = false;
  for (const auto& fb : feedback) {
    if (fb.type == AdaptiveFeedback::FeedbackType::MEMORY_OPTIMIZATION) {
      found_memory_feedback = true;
      EXPECT_GT(fb.confidence_score, 0.5);
      break;
    }
  }
  EXPECT_TRUE(found_memory_feedback);
}

TEST_F(MetaCognitiveMonitorTest, AgentRecursiveAssessment) {
  MetaCognitiveAgent agent(4, "RecursiveAgent");
  
  // Set up some baseline metrics
  agent.recordKernelExecution(1, 1.0);
  agent.recordMemoryUsage(512 * 1024 * 1024); // 512MB
  agent.recordParallelRegion(4);
  
  double initial_score = agent.performSelfAssessment();
  
  // Perform recursive assessment
  agent.recursivelyAssessSubsystems();
  
  double recursive_score = agent.getState().performance_score.load();
  
  // Recursive score should be computed and valid
  EXPECT_GE(recursive_score, 0.0);
  EXPECT_LE(recursive_score, 1.0);
}

// Test GlobalIntrospector functionality
TEST_F(MetaCognitiveMonitorTest, GlobalIntrospectorBasics) {
  auto& introspector = getGlobalIntrospector();
  
  // Register agents
  uint32_t agent1_id = introspector.registerAgent("Agent1");
  uint32_t agent2_id = introspector.registerAgent("Agent2");
  
  EXPECT_NE(agent1_id, agent2_id);
  EXPECT_EQ(introspector.getActiveAgentCount(), 2);
  
  // Get agents and verify they exist
  auto* agent1 = introspector.getAgent(agent1_id);
  auto* agent2 = introspector.getAgent(agent2_id);
  
  ASSERT_NE(agent1, nullptr);
  ASSERT_NE(agent2, nullptr);
  
  EXPECT_EQ(agent1->getState().agent_name, "Agent1");
  EXPECT_EQ(agent2->getState().agent_name, "Agent2");
  
  // Unregister one agent
  introspector.unregisterAgent(agent1_id);
  EXPECT_EQ(introspector.getActiveAgentCount(), 1);
  EXPECT_EQ(introspector.getAgent(agent1_id), nullptr);
}

TEST_F(MetaCognitiveMonitorTest, GlobalIntrospectorAggregation) {
  auto& introspector = getGlobalIntrospector();
  
  // Create and configure agents
  uint32_t agent1_id = introspector.registerAgent("AggregationAgent1");
  uint32_t agent2_id = introspector.registerAgent("AggregationAgent2");
  
  auto* agent1 = introspector.getAgent(agent1_id);
  auto* agent2 = introspector.getAgent(agent2_id);
  
  ASSERT_NE(agent1, nullptr);
  ASSERT_NE(agent2, nullptr);
  
  // Add some metrics to agents
  agent1->recordKernelExecution(1, 1.0);
  agent1->recordMemoryUsage(100 * 1024 * 1024); // 100MB
  
  agent2->recordKernelExecution(2, 2.0);
  agent2->recordMemoryUsage(200 * 1024 * 1024); // 200MB
  
  // Test aggregation
  auto aggregated = introspector.aggregateAgentStates();
  
  EXPECT_EQ(aggregated.kernel_executions.load(), 2);
  EXPECT_EQ(aggregated.memory_usage.load(), 300 * 1024 * 1024); // 300MB total
  EXPECT_GT(aggregated.execution_time.load(), 0);
  
  // Clean up
  introspector.unregisterAgent(agent1_id);
  introspector.unregisterAgent(agent2_id);
}

TEST_F(MetaCognitiveMonitorTest, GlobalIntrospectorSystemOptimization) {
  auto& introspector = getGlobalIntrospector();
  
  // Create agents with different performance characteristics
  uint32_t fast_agent_id = introspector.registerAgent("FastAgent");
  uint32_t slow_agent_id = introspector.registerAgent("SlowAgent");
  
  auto* fast_agent = introspector.getAgent(fast_agent_id);
  auto* slow_agent = introspector.getAgent(slow_agent_id);
  
  ASSERT_NE(fast_agent, nullptr);
  ASSERT_NE(slow_agent, nullptr);
  
  // Configure fast agent
  for (int i = 0; i < 100; ++i) {
    fast_agent->recordKernelExecution(i, 0.01); // Fast execution
  }
  fast_agent->recordMemoryUsage(50 * 1024 * 1024); // 50MB
  
  // Configure slow agent
  for (int i = 0; i < 10; ++i) {
    slow_agent->recordKernelExecution(i + 1000, 1.0); // Slow execution
  }
  slow_agent->recordMemoryUsage(2UL * 1024 * 1024 * 1024); // 2GB - high memory
  
  // Perform self-assessments
  fast_agent->performSelfAssessment();
  slow_agent->performSelfAssessment();
  
  // Generate system optimization
  auto system_feedback = introspector.generateSystemOptimization();
  
  // Should generate optimization feedback
  EXPECT_GT(system_feedback.size(), 0);
  
  // Apply feedback
  introspector.applyAdaptiveFeedback();
  
  // Verify feedback was stored
  const auto& global_feedback = introspector.getGlobalFeedback();
  EXPECT_GT(global_feedback.size(), 0);
  
  // Clean up
  introspector.unregisterAgent(fast_agent_id);
  introspector.unregisterAgent(slow_agent_id);
}

TEST_F(MetaCognitiveMonitorTest, GlobalIntrospectorRecursiveAssessment) {
  auto& introspector = getGlobalIntrospector();
  
  // Create multiple agents for system-wide assessment
  std::vector<uint32_t> agent_ids;
  for (int i = 0; i < 5; ++i) {
    agent_ids.push_back(introspector.registerAgent("RecursiveTestAgent" + std::to_string(i)));
  }
  
  // Add varying performance levels to create system heterogeneity
  for (size_t i = 0; i < agent_ids.size(); ++i) {
    auto* agent = introspector.getAgent(agent_ids[i]);
    ASSERT_NE(agent, nullptr);
    
    // Different performance characteristics
    agent->recordKernelExecution(i, 0.1 * (i + 1));
    agent->recordMemoryUsage((i + 1) * 100 * 1024 * 1024); // Varying memory usage
    agent->performSelfAssessment();
  }
  
  // Perform recursive system assessment
  introspector.recursiveSystemAssessment();
  
  // Verify that recursive assessment affects the global feedback
  const auto& feedback_before = introspector.getGlobalFeedback().size();
  
  introspector.performGlobalIntrospection();
  
  const auto& feedback_after = introspector.getGlobalFeedback().size();
  EXPECT_GE(feedback_after, feedback_before);
  
  // Clean up
  for (auto agent_id : agent_ids) {
    introspector.unregisterAgent(agent_id);
  }
}

TEST_F(MetaCognitiveMonitorTest, SystemEfficiencyCalculation) {
  auto& introspector = getGlobalIntrospector();
  
  // Test efficiency calculation with no agents
  double efficiency_empty = introspector.calculateSystemEfficiency();
  EXPECT_EQ(efficiency_empty, 1.0);
  
  // Add agents with known performance scores
  uint32_t agent1_id = introspector.registerAgent("EfficiencyAgent1");
  uint32_t agent2_id = introspector.registerAgent("EfficiencyAgent2");
  
  auto* agent1 = introspector.getAgent(agent1_id);
  auto* agent2 = introspector.getAgent(agent2_id);
  
  ASSERT_NE(agent1, nullptr);
  ASSERT_NE(agent2, nullptr);
  
  // Set up different performance levels
  agent1->recordKernelExecution(1, 0.1); // Fast
  agent2->recordKernelExecution(2, 1.0); // Slow
  
  agent1->performSelfAssessment();
  agent2->performSelfAssessment();
  
  double efficiency = introspector.calculateSystemEfficiency();
  EXPECT_GE(efficiency, 0.0);
  EXPECT_LE(efficiency, 1.0);
  
  // Clean up
  introspector.unregisterAgent(agent1_id);
  introspector.unregisterAgent(agent2_id);
}

// Integration test with callback functions
TEST_F(MetaCognitiveMonitorTest, CallbackIntegration) {
  // Test the callback integration
  uint64_t kernel_id = 12345;
  
  // Simulate begin callback
  metaCognitiveBeginCallback("TestKernel", 0, &kernel_id);
  
  // Verify agent was created
  auto& introspector = getGlobalIntrospector();
  EXPECT_GT(introspector.getActiveAgentCount(), 0);
  
  // Simulate end callback
  metaCognitiveEndCallback(kernel_id);
  
  // Simulate region callback
  metaCognitiveRegionCallback("TestRegion");
  
  // Verify system still functions
  EXPECT_GE(introspector.getActiveAgentCount(), 0);
}

// Performance and stress test
TEST_F(MetaCognitiveMonitorTest, PerformanceStressTest) {
  auto& introspector = getGlobalIntrospector();
  
  const int num_agents = 50;
  const int operations_per_agent = 1000;
  
  std::vector<uint32_t> agent_ids;
  
  // Create many agents
  for (int i = 0; i < num_agents; ++i) {
    agent_ids.push_back(introspector.registerAgent("StressAgent" + std::to_string(i)));
  }
  
  // Perform many operations on each agent
  auto start_time = std::chrono::high_resolution_clock::now();
  
  for (auto agent_id : agent_ids) {
    auto* agent = introspector.getAgent(agent_id);
    ASSERT_NE(agent, nullptr);
    
    for (int op = 0; op < operations_per_agent; ++op) {
      agent->recordKernelExecution(op, 0.001);
      if (op % 100 == 0) {
        agent->performSelfAssessment();
      }
    }
  }
  
  // Perform global introspection
  introspector.performGlobalIntrospection();
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  // Verify reasonable performance (should complete within 10 seconds)
  EXPECT_LT(duration.count(), 10000);
  
  // Verify system state is still coherent
  EXPECT_EQ(introspector.getActiveAgentCount(), num_agents);
  
  auto aggregated = introspector.aggregateAgentStates();
  EXPECT_EQ(aggregated.kernel_executions.load(), num_agents * operations_per_agent);
  
  // Clean up
  for (auto agent_id : agent_ids) {
    introspector.unregisterAgent(agent_id);
  }
}

// Thread safety test
TEST_F(MetaCognitiveMonitorTest, ThreadSafetyTest) {
  auto& introspector = getGlobalIntrospector();
  
  const int num_threads = 8;
  const int operations_per_thread = 100;
  
  std::vector<std::thread> threads;
  std::vector<uint32_t> agent_ids(num_threads);
  
  // Create agents from multiple threads
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      agent_ids[t] = introspector.registerAgent("ThreadAgent" + std::to_string(t));
      auto* agent = introspector.getAgent(agent_ids[t]);
      
      if (agent) {
        for (int op = 0; op < operations_per_thread; ++op) {
          agent->recordKernelExecution(t * 1000 + op, 0.01);
          agent->recordMemoryUsage((t + 1) * 1024 * 1024);
          
          if (op % 20 == 0) {
            agent->performSelfAssessment();
          }
        }
      }
    });
  }
  
  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }
  
  // Verify all agents were created and operated correctly
  EXPECT_EQ(introspector.getActiveAgentCount(), num_threads);
  
  auto aggregated = introspector.aggregateAgentStates();
  EXPECT_EQ(aggregated.kernel_executions.load(), num_threads * operations_per_thread);
  
  // Clean up
  for (auto agent_id : agent_ids) {
    if (agent_id != 0) {  // 0 indicates failure to register
      introspector.unregisterAgent(agent_id);
    }
  }
}

} // namespace Test