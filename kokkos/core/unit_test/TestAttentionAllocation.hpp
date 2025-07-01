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
#include <Kokkos_AttentionAllocation.hpp>
#include <Kokkos_AgenticKernel.hpp>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <iostream>
#include <cassert>

namespace Test {

using namespace Kokkos::Experimental;

// Helper function for assertions
void assert_test(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "Test failed: " << message << std::endl;
    std::abort();
  }
}

// Test basic cognitive agent functionality
void test_basic_cognitive_agent() {
  std::cout << "Running test_basic_cognitive_agent..." << std::endl;
  
  auto agent = std::make_shared<AgenticKernel>(1.0);
  
  // Test initial state
  assert_test(agent->getId() > 0, "Agent ID should be positive");
  assert_test(std::abs(agent->getActivationLevel() - 1.0) < 1e-9, "Initial activation should be 1.0");
  assert_test(std::abs(agent->getCurrencyBalance() - 0.0) < 1e-9, "Initial currency should be 0.0");
  
  // Test currency operations
  agent->receiveCurrency(100.0);
  assert_test(std::abs(agent->getCurrencyBalance() - 100.0) < 1e-9, "Currency should be 100.0 after receiving");
  
  assert_test(agent->spendCurrency(50.0), "Should be able to spend 50.0");
  assert_test(std::abs(agent->getCurrencyBalance() - 50.0) < 1e-9, "Currency should be 50.0 after spending");
  
  assert_test(!agent->spendCurrency(100.0), "Should not be able to spend 100.0 with 50.0 balance");
  assert_test(std::abs(agent->getCurrencyBalance() - 50.0) < 1e-9, "Currency should remain 50.0 after failed spend");
  
  std::cout << "test_basic_cognitive_agent passed!" << std::endl;
}

// Test attention allocator registration and basic allocation
void test_basic_allocation() {
  std::cout << "Running test_basic_allocation..." << std::endl;
  
  auto allocator = std::make_shared<AttentionAllocator<>>(1000.0, 10);
  auto agent1 = std::make_shared<AgenticKernel>(1.0);
  auto agent2 = std::make_shared<AgenticKernel>(2.0);
  
  allocator->registerAgent(agent1);
  allocator->registerAgent(agent2);
  
  assert_test(allocator->getAgentCount() == 2, "Should have 2 agents registered");
  
  // Initial allocations should be zero
  assert_test(std::abs(allocator->getAllocation(agent1->getId()) - 0.0) < 1e-9, "Initial allocation should be 0");
  assert_test(std::abs(allocator->getAllocation(agent2->getId()) - 0.0) < 1e-9, "Initial allocation should be 0");
  
  // Update allocations
  allocator->updateAllocations();
  
  // Agent2 should get more allocation due to higher activation
  double alloc1 = allocator->getAllocation(agent1->getId());
  double alloc2 = allocator->getAllocation(agent2->getId());
  
  assert_test(alloc1 > 0.0, "Agent1 allocation should be positive");
  assert_test(alloc2 > 0.0, "Agent2 allocation should be positive");
  assert_test(alloc2 > alloc1, "Agent2 should get more allocation due to higher activation");
  
  std::cout << "test_basic_allocation passed!" << std::endl;
}

// Test demand-based allocation with varying activation levels
void test_demand_based_allocation() {
  std::cout << "Running test_demand_based_allocation..." << std::endl;
  
  auto allocator = std::make_shared<AttentionAllocator<>>(1000.0, 10);
  std::vector<std::shared_ptr<AgenticKernel>> agents;
  std::vector<double> activations = {0.1, 0.5, 1.0, 2.0, 5.0};
  
  // Create agents with different activation levels
  for (double activation : activations) {
    auto agent = std::make_shared<AgenticKernel>(activation);
    agents.push_back(agent);
    allocator->registerAgent(agent);
    
    // Give agents initial currency
    agent->receiveCurrency(100.0);
  }
  
  // Update allocations
  allocator->updateAllocations();
  
  // Verify allocations increase with activation level
  for (size_t i = 1; i < agents.size(); ++i) {
    double prevAlloc = allocator->getAllocation(agents[i-1]->getId());
    double currAlloc = allocator->getAllocation(agents[i]->getId());
    
    assert_test(currAlloc > prevAlloc, "Higher activation agent should get more allocation");
  }
  
  // Verify total allocation doesn't exceed total resources
  double totalAllocation = 0.0;
  for (const auto& agent : agents) {
    totalAllocation += allocator->getAllocation(agent->getId());
  }
  
  assert_test(totalAllocation <= allocator->getTotalResources() * 1.1, "Total allocation should not exceed resources");
  
  std::cout << "test_demand_based_allocation passed!" << std::endl;
}

// Test factory pattern for agentic kernels
void test_agentic_kernel_factory() {
  std::cout << "Running test_agentic_kernel_factory..." << std::endl;
  
  auto allocator = std::make_shared<AttentionAllocator<>>(1000.0, 10);
  AgenticKernelFactory<> factory(allocator);
  
  auto kernel1 = factory.createKernel(1.0);
  auto kernel2 = factory.createKernel(2.0);
  
  // Kernels should be automatically registered
  assert_test(allocator->getAgentCount() == 2, "Should have 2 agents registered");
  
  // Kernels should have initial currency
  assert_test(kernel1->getCurrencyBalance() > 0.0, "Kernel1 should have initial currency");
  assert_test(kernel2->getCurrencyBalance() > 0.0, "Kernel2 should have initial currency");
  
  // Test callback creation
  bool callbackCalled = false;
  auto kernel3 = factory.createKernel(1.5, [&callbackCalled](double) {
    callbackCalled = true;
  });
  
  // Trigger allocation update to invoke callback
  allocator->updateAllocations();
  assert_test(callbackCalled, "Callback should have been called");
  
  std::cout << "test_agentic_kernel_factory passed!" << std::endl;
}

// Main test runner function
int run_attention_allocation_tests() {
  std::cout << "Starting ECAN Attention Allocation Tests..." << std::endl;
  
  try {
    test_basic_cognitive_agent();
    test_basic_allocation();
    test_demand_based_allocation();
    test_agentic_kernel_factory();
    
    std::cout << "All attention allocation tests passed!" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test failed with unknown exception" << std::endl;
    return 1;
  }
}

}  // namespace Test