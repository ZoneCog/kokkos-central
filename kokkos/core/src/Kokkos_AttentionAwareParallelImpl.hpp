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
#ifndef KOKKOS_ATTENTION_AWARE_PARALLEL_IMPL_HPP
#define KOKKOS_ATTENTION_AWARE_PARALLEL_IMPL_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_AgenticKernel.hpp>

namespace Kokkos {
namespace Experimental {

// Implementation of AttentionAwareParallel methods

template <typename ExecutionSpace>
AttentionAwareParallel<ExecutionSpace>::AttentionAwareParallel(
    std::shared_ptr<AgenticKernel> agent,
    std::shared_ptr<AttentionAllocator<ExecutionSpace>> allocator)
    : agent_(agent), allocator_(allocator) {}

template <typename ExecutionSpace>
template <typename Policy, typename Functor>
void AttentionAwareParallel<ExecutionSpace>::parallel_for(
    const std::string& label, const Policy& policy, const Functor& functor) {
  // Record execution for activation boost
  agent_->recordExecution();
  
  // Trigger attention allocation update
  allocator_->economyTick();
  
  // Get current allocation for potential throttling
  double allocation = agent_->getCurrentAllocation();
  
  // Execute with potential resource scaling based on allocation
  if (allocation > 0.5) {
    // High allocation - full execution
    Kokkos::parallel_for(label, policy, functor);
  } else if (allocation > 0.1) {
    // Medium allocation - potentially reduced execution
    auto scaledPolicy = scalePolicy(policy, allocation);
    Kokkos::parallel_for(label, scaledPolicy, functor);
  } else {
    // Low allocation - minimal execution or defer
    auto minimalPolicy = scalePolicy(policy, 0.1);
    Kokkos::parallel_for(label, minimalPolicy, functor);
  }
  
  // Spend currency based on work performed
  double workCost = calculateWorkCost(policy, allocation);
  agent_->spendCurrency(workCost);
}

template <typename ExecutionSpace>
template <typename Policy, typename Functor, typename Value>
void AttentionAwareParallel<ExecutionSpace>::parallel_reduce(
    const std::string& label, const Policy& policy,
    const Functor& functor, Value& result) {
  agent_->recordExecution();
  allocator_->economyTick();
  
  double allocation = agent_->getCurrentAllocation();
  
  if (allocation > 0.1) {
    auto scaledPolicy = scalePolicy(policy, allocation);
    Kokkos::parallel_reduce(label, scaledPolicy, functor, result);
    
    double workCost = calculateWorkCost(policy, allocation);
    agent_->spendCurrency(workCost);
  } else {
    // Insufficient allocation - return default
    result = Value{};
  }
}

template <typename ExecutionSpace>
template <typename Policy>
Policy AttentionAwareParallel<ExecutionSpace>::scalePolicy(
    const Policy& policy, double allocation) {
  // For RangePolicy, scale the range
  if constexpr (std::is_base_of_v<Kokkos::RangePolicy<ExecutionSpace>, Policy>) {
    auto begin = policy.begin();
    auto end = policy.end();
    auto scaledEnd = begin + static_cast<typename Policy::index_type>(
        (end - begin) * std::min(1.0, allocation * 2.0));
    return Policy(begin, scaledEnd);
  } else {
    // For other policies, return as-is for now
    return policy;
  }
}

template <typename ExecutionSpace>
template <typename Policy>
double AttentionAwareParallel<ExecutionSpace>::calculateWorkCost(
    const Policy& policy, double allocation) {
  // Base cost proportional to work size and inversely to allocation
  if constexpr (std::is_base_of_v<Kokkos::RangePolicy<ExecutionSpace>, Policy>) {
    auto workSize = policy.end() - policy.begin();
    return static_cast<double>(workSize) * 0.001 / (allocation + 0.1);
  } else {
    return 1.0 / (allocation + 0.1);
  }
}

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_ATTENTION_AWARE_PARALLEL_IMPL_HPP