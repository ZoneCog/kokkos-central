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
#ifndef KOKKOS_ATTENTION_ALLOCATION_HPP
#define KOKKOS_ATTENTION_ALLOCATION_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cmath>

namespace Kokkos {
namespace Experimental {

/// \brief ECAN-style Economic Attention Network for resource allocation
///
/// This implements an economic model where agentic kernels participate in
/// cognitive resource management through earning, spending, and reallocating
/// cognitive currency based on demand and activation levels.

/// \brief Cognitive currency value type
using CognitiveCurrency = std::atomic<double>;

/// \brief Activation level for demand-based allocation
using ActivationLevel = double;

/// \brief Agent identifier type
using AgentId = size_t;

/// \brief Interface for cognitive agents that participate in attention allocation
class CognitiveAgent {
 public:
  virtual ~CognitiveAgent() = default;
  
  /// \brief Get the agent's unique identifier
  virtual AgentId getId() const = 0;
  
  /// \brief Get current activation level for demand calculation
  virtual ActivationLevel getActivationLevel() const = 0;
  
  /// \brief Called when agent receives cognitive currency
  virtual void receiveCurrency(double amount) = 0;
  
  /// \brief Called when agent spends cognitive currency
  /// \return true if spending was successful, false if insufficient funds
  virtual bool spendCurrency(double amount) = 0;
  
  /// \brief Get current cognitive currency balance
  virtual double getCurrencyBalance() const = 0;
  
  /// \brief Called when agent's resource allocation changes
  virtual void onAllocationChanged(double newAllocation) = 0;
};

/// \brief ECAN-style attention allocator for distributed resource management
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class AttentionAllocator {
 public:
  using execution_space = ExecutionSpace;
  using size_type = typename execution_space::size_type;
  
 private:
  /// \brief Registry of all participating cognitive agents
  std::unordered_map<AgentId, std::shared_ptr<CognitiveAgent>> agents_;
  
  /// \brief Current resource allocations per agent
  std::unordered_map<AgentId, std::atomic<double>> allocations_;
  
  /// \brief Total available cognitive resources
  std::atomic<double> totalResources_;
  
  /// \brief Economy update frequency (for reallocation cycles)
  size_type updateFrequency_;
  
  /// \brief Counter for economy updates
  std::atomic<size_type> updateCounter_;
  
  /// \brief Mutex for thread-safe agent registration
  mutable std::mutex agentMutex_;
  
  /// \brief Base currency amount for new agents
  static constexpr double BASE_CURRENCY = 100.0;
  
  /// \brief Minimum allocation threshold
  static constexpr double MIN_ALLOCATION = 0.01;
  
 public:
  /// \brief Construct attention allocator with total resources
  explicit AttentionAllocator(double totalResources = 1000.0, 
                            size_type updateFreq = 100)
      : totalResources_(totalResources), updateFrequency_(updateFreq), 
        updateCounter_(0) {}
  
  /// \brief Register a cognitive agent in the attention economy
  void registerAgent(std::shared_ptr<CognitiveAgent> agent) {
    std::lock_guard<std::mutex> lock(agentMutex_);
    AgentId id = agent->getId();
    agents_[id] = agent;
    allocations_[id] = 0.0;
    
    // Give new agent base currency
    agent->receiveCurrency(BASE_CURRENCY);
  }
  
  /// \brief Unregister an agent from the attention economy
  void unregisterAgent(AgentId id) {
    std::lock_guard<std::mutex> lock(agentMutex_);
    agents_.erase(id);
    allocations_.erase(id);
  }
  
  /// \brief Get current allocation for an agent
  double getAllocation(AgentId id) const {
    auto it = allocations_.find(id);
    return (it != allocations_.end()) ? it->second.load() : 0.0;
  }
  
  /// \brief Update allocations based on current demand and activation levels
  void updateAllocations() {
    std::lock_guard<std::mutex> lock(agentMutex_);
    
    if (agents_.empty()) return;
    
    // Calculate total demand weighted by activation and currency
    double totalDemand = 0.0;
    std::vector<std::pair<AgentId, double>> demands;
    
    for (const auto& [id, agent] : agents_) {
      double activation = agent->getActivationLevel();
      double currency = agent->getCurrencyBalance();
      
      // Demand is proportional to activation level and available currency
      double demand = activation * std::sqrt(currency + 1.0);
      demands.emplace_back(id, demand);
      totalDemand += demand;
    }
    
    // Distribute resources proportionally to demand
    double availableResources = totalResources_.load();
    for (const auto& [id, demand] : demands) {
      double allocation = (totalDemand > 0.0) ? 
                         (demand / totalDemand) * availableResources : 
                         availableResources / agents_.size();
      
      // Apply minimum allocation threshold
      allocation = std::max(allocation, MIN_ALLOCATION);
      
      // Update allocation and notify agent
      allocations_[id] = allocation;
      agents_[id]->onAllocationChanged(allocation);
    }
  }
  
  /// \brief Periodic economy update (call this regularly for dynamic allocation)
  void economyTick() {
    size_type currentCount = updateCounter_.fetch_add(1);
    if (currentCount % updateFrequency_ == 0) {
      updateAllocations();
      redistributeResources();
    }
  }
  
  /// \brief Get total number of registered agents
  size_type getAgentCount() const {
    std::lock_guard<std::mutex> lock(agentMutex_);
    return agents_.size();
  }
  
  /// \brief Get total available resources
  double getTotalResources() const {
    return totalResources_.load();
  }
  
  /// \brief Set total available resources
  void setTotalResources(double resources) {
    totalResources_ = resources;
  }
  
 private:
  /// \brief Redistribute resources based on economic activity
  void redistributeResources() {
    std::lock_guard<std::mutex> lock(agentMutex_);
    
    // Implement wealth redistribution mechanism
    // Agents with high currency can "invest" in the economy
    // Agents with low currency receive "stimulus"
    
    double totalCurrency = 0.0;
    for (const auto& [id, agent] : agents_) {
      totalCurrency += agent->getCurrencyBalance();
    }
    
    if (totalCurrency > 0.0) {
      double avgCurrency = totalCurrency / agents_.size();
      
      for (const auto& [id, agent] : agents_) {
        double balance = agent->getCurrencyBalance();
        
        if (balance > avgCurrency * 1.5) {
          // High-balance agents contribute to economy
          double contribution = (balance - avgCurrency) * 0.1;
          if (agent->spendCurrency(contribution)) {
            // Redistribute to low-balance agents
            for (const auto& [otherId, otherAgent] : agents_) {
              if (otherAgent->getCurrencyBalance() < avgCurrency * 0.5) {
                otherAgent->receiveCurrency(contribution / agents_.size());
              }
            }
          }
        }
      }
    }
  }
};

/// \brief RAII helper for attention-based resource acquisition
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class AttentionBasedResource {
 public:
  using execution_space = ExecutionSpace;
  using allocator_type = AttentionAllocator<execution_space>;
  
 private:
  std::shared_ptr<allocator_type> allocator_;
  std::shared_ptr<CognitiveAgent> agent_;
  double allocatedAmount_;
  
 public:
  AttentionBasedResource(std::shared_ptr<allocator_type> allocator,
                        std::shared_ptr<CognitiveAgent> agent)
      : allocator_(allocator), agent_(agent), allocatedAmount_(0.0) {
    
    // Register agent if not already registered
    allocator_->registerAgent(agent_);
    
    // Trigger allocation update
    allocator_->economyTick();
    
    // Get current allocation
    allocatedAmount_ = allocator_->getAllocation(agent_->getId());
  }
  
  ~AttentionBasedResource() {
    // Agent remains registered for continued participation
    // Unregistration should be explicit via allocator
  }
  
  /// \brief Get allocated resource amount
  double getAllocatedAmount() const { return allocatedAmount_; }
  
  /// \brief Update allocation (triggers economy tick)
  void updateAllocation() {
    allocator_->economyTick();
    allocatedAmount_ = allocator_->getAllocation(agent_->getId());
  }
  
  /// \brief Spend currency for additional resources
  bool spendForResources(double amount) {
    return agent_->spendCurrency(amount);
  }
};

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_ATTENTION_ALLOCATION_HPP