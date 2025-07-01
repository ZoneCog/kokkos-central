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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_AGENTIC_KERNEL_HPP
#define KOKKOS_AGENTIC_KERNEL_HPP

#include <Kokkos_AttentionAllocation.hpp>
#include <atomic>
#include <chrono>
#include <functional>
#include <cmath>
#include <type_traits>

namespace Kokkos {
namespace Experimental {

/// \brief Default implementation of CognitiveAgent for agentic kernels
class AgenticKernel : public CognitiveAgent {
 private:
  AgentId id_;
  std::atomic<double> currencyBalance_;
  std::atomic<double> activationLevel_;
  std::atomic<double> currentAllocation_;
  std::atomic<size_t> executionCount_;
  std::atomic<size_t> lastActivityTime_;
  
  // Callback for allocation changes
  std::function<void(double)> allocationCallback_;
  
  /// \brief Generate unique ID based on address and timestamp
  static AgentId generateId() {
    static std::atomic<AgentId> counter{1};
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = now.time_since_epoch().count();
    return counter.fetch_add(1) ^ static_cast<AgentId>(timestamp);
  }
  
 public:
  /// \brief Create agentic kernel with initial activation level
  explicit AgenticKernel(double initialActivation = 1.0)
      : id_(generateId()), currencyBalance_(0.0), 
        activationLevel_(initialActivation), currentAllocation_(0.0),
        executionCount_(0), lastActivityTime_(0) {
    updateActivity();
  }
  
  /// \brief Create agentic kernel with allocation change callback
  AgenticKernel(double initialActivation, 
                std::function<void(double)> callback)
      : id_(generateId()), currencyBalance_(0.0), 
        activationLevel_(initialActivation), currentAllocation_(0.0),
        executionCount_(0), lastActivityTime_(0),
        allocationCallback_(callback) {
    updateActivity();
  }
  
  // CognitiveAgent interface implementation
  AgentId getId() const override { return id_; }
  
  ActivationLevel getActivationLevel() const override { 
    // Decay activation based on time since last activity
    auto currentTime = getCurrentTime();
    auto timeSinceActivity = currentTime - lastActivityTime_.load();
    
    // Apply exponential decay (half-life of ~1000 time units)
    double decayFactor = std::exp(-0.0007 * timeSinceActivity);
    return activationLevel_.load() * decayFactor;
  }
  
  void receiveCurrency(double amount) override {
    if (amount > 0.0) {
      double current = currencyBalance_.load();
      while (!currencyBalance_.compare_exchange_weak(current, current + amount)) {
        // Spin until successful
      }
    }
  }
  
  bool spendCurrency(double amount) override {
    if (amount <= 0.0) return true;
    
    double current = currencyBalance_.load();
    while (current >= amount) {
      if (currencyBalance_.compare_exchange_weak(current, current - amount)) {
        return true;
      }
    }
    return false; // Insufficient funds
  }
  
  double getCurrencyBalance() const override {
    return currencyBalance_.load();
  }
  
  void onAllocationChanged(double newAllocation) override {
    currentAllocation_ = newAllocation;
    if (allocationCallback_) {
      allocationCallback_(newAllocation);
    }
  }
  
  // Additional kernel-specific methods
  
  /// \brief Set activation level (affects resource demand)
  void setActivationLevel(double level) {
    activationLevel_ = std::max(0.0, level);
    updateActivity();
  }
  
  /// \brief Boost activation level temporarily
  void boostActivation(double boost) {
    double current = activationLevel_.load();
    activationLevel_ = current + boost;
    updateActivity();
  }
  
  /// \brief Get current resource allocation
  double getCurrentAllocation() const {
    return currentAllocation_.load();
  }
  
  /// \brief Record kernel execution (affects activation)
  void recordExecution() {
    executionCount_.fetch_add(1);
    updateActivity();
    
    // Boost activation based on execution frequency
    double boost = 0.1 * std::log(1.0 + executionCount_.load());
    boostActivation(boost);
  }
  
  /// \brief Get execution count
  size_t getExecutionCount() const {
    return executionCount_.load();
  }
  
  /// \brief Set allocation change callback
  void setAllocationCallback(std::function<void(double)> callback) {
    allocationCallback_ = callback;
  }
  
 private:
  /// \brief Update last activity timestamp
  void updateActivity() {
    lastActivityTime_ = getCurrentTime();
  }
  
  /// \brief Get current time in arbitrary units
  static size_t getCurrentTime() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<size_t>(now.time_since_epoch().count() / 1000000); // microseconds
  }
};

/// \brief Factory for creating agentic kernels with attention allocation
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class AgenticKernelFactory {
 private:
  std::shared_ptr<AttentionAllocator<ExecutionSpace>> allocator_;
  
 public:
  explicit AgenticKernelFactory(
      std::shared_ptr<AttentionAllocator<ExecutionSpace>> allocator)
      : allocator_(allocator) {}
  
  /// \brief Create an agentic kernel with automatic registration
  std::shared_ptr<AgenticKernel> createKernel(double initialActivation = 1.0) {
    auto kernel = std::make_shared<AgenticKernel>(initialActivation);
    allocator_->registerAgent(kernel);
    return kernel;
  }
  
  /// \brief Create an agentic kernel with callback
  std::shared_ptr<AgenticKernel> createKernel(
      double initialActivation,
      std::function<void(double)> allocationCallback) {
    auto kernel = std::make_shared<AgenticKernel>(initialActivation, allocationCallback);
    allocator_->registerAgent(kernel);
    return kernel;
  }
  
  /// \brief Get the underlying attention allocator
  std::shared_ptr<AttentionAllocator<ExecutionSpace>> getAllocator() const {
    return allocator_;
  }
};

/// \brief Wrapper for Kokkos parallel constructs with attention allocation
/// 
/// Note: Implementation is forward-declared to avoid circular dependencies.
/// Full implementation is available when including the full Kokkos_Core.hpp
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class AttentionAwareParallel {
 private:
  std::shared_ptr<AgenticKernel> agent_;
  std::shared_ptr<AttentionAllocator<ExecutionSpace>> allocator_;
  
 public:
  AttentionAwareParallel(std::shared_ptr<AgenticKernel> agent,
                        std::shared_ptr<AttentionAllocator<ExecutionSpace>> allocator);
  
  /// \brief Execute parallel_for with attention-based resource management
  template <typename Policy, typename Functor>
  void parallel_for(const std::string& label, const Policy& policy, 
                   const Functor& functor);
  
  /// \brief Execute parallel_reduce with attention-based resource management
  template <typename Policy, typename Functor, typename Value>
  void parallel_reduce(const std::string& label, const Policy& policy,
                      const Functor& functor, Value& result);
  
 private:
  /// \brief Scale execution policy based on allocation
  template <typename Policy>
  Policy scalePolicy(const Policy& policy, double allocation);
  
  /// \brief Calculate work cost for currency spending
  template <typename Policy>
  double calculateWorkCost(const Policy& policy, double allocation);
};

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_AGENTIC_KERNEL_HPP