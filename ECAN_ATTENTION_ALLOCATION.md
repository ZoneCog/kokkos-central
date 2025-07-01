# ECAN-Style Attention Allocation for Agentic Kernels

This document describes the ECAN (Economic Attention Network) style attention allocation system integrated into Kokkos for managing computational resources among agentic kernels.

## Overview

The ECAN system implements an economic model where agentic kernels participate in cognitive resource management through earning, spending, and reallocating cognitive currency based on demand and activation levels. This enables true distributed allocation across all participating agents.

## Core Components

### 1. CognitiveAgent Interface

The `CognitiveAgent` interface defines the contract for agents that can participate in the attention economy:

```cpp
class CognitiveAgent {
public:
    virtual AgentId getId() const = 0;
    virtual ActivationLevel getActivationLevel() const = 0;
    virtual void receiveCurrency(double amount) = 0;
    virtual bool spendCurrency(double amount) = 0;
    virtual double getCurrencyBalance() const = 0;
    virtual void onAllocationChanged(double newAllocation) = 0;
};
```

### 2. AttentionAllocator

The `AttentionAllocator` manages the economic distribution of cognitive resources:

```cpp
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class AttentionAllocator {
public:
    explicit AttentionAllocator(double totalResources = 1000.0, 
                              size_type updateFreq = 100);
    
    void registerAgent(std::shared_ptr<CognitiveAgent> agent);
    void unregisterAgent(AgentId id);
    double getAllocation(AgentId id) const;
    void updateAllocations();
    void economyTick();
    // ...
};
```

### 3. AgenticKernel

A concrete implementation of `CognitiveAgent` designed for computational kernels:

```cpp
class AgenticKernel : public CognitiveAgent {
public:
    explicit AgenticKernel(double initialActivation = 1.0);
    
    void setActivationLevel(double level);
    void boostActivation(double boost);
    void recordExecution();
    double getCurrentAllocation() const;
    size_t getExecutionCount() const;
    // ...
};
```

## Key Features

### Economic Resource Management

- **Currency System**: Agents earn and spend cognitive currency
- **Initial Endowment**: New agents receive base currency when registered
- **Redistribution**: Wealth redistribution mechanism prevents excessive inequality
- **Dynamic Economy**: Regular economy ticks update allocations and redistribute resources

### Demand-Based Allocation

Resource allocation is determined by:
- **Activation Level**: Higher activation = higher demand
- **Currency Balance**: Available currency affects resource access
- **Activity History**: Recent execution activity boosts activation
- **Time Decay**: Activation naturally decays over time without activity

### Distributed Architecture

- **Thread-Safe**: All operations are thread-safe using atomic operations
- **Concurrent Registration**: Agents can be registered/unregistered concurrently
- **Scalable**: Supports large numbers of concurrent agents
- **Lock-Free Operations**: Core allocation calculations avoid locks where possible

## Usage Examples

### Basic Usage

```cpp
#include <Kokkos_Core.hpp>

using namespace Kokkos::Experimental;

// Create attention allocator
auto allocator = std::make_shared<AttentionAllocator<>>(1000.0, 10);

// Create agentic kernel factory
AgenticKernelFactory<> factory(allocator);

// Create kernels with different priorities
auto highPriorityKernel = factory.createKernel(5.0);  // High activation
auto lowPriorityKernel = factory.createKernel(1.0);   // Normal activation

// Execute with attention awareness
AttentionAwareParallel<> attentionExec(highPriorityKernel, allocator);

attentionExec.parallel_for(
    "my_kernel",
    Kokkos::RangePolicy<>(0, N),
    KOKKOS_LAMBDA(int i) {
        // Kernel implementation
    });
```

### Advanced Usage with Callbacks

```cpp
// Create kernel with allocation change callback
auto kernel = factory.createKernel(2.0, [](double newAllocation) {
    std::cout << "Allocation changed to: " << newAllocation << std::endl;
});

// Monitor economic activity
for (int round = 0; round < 10; ++round) {
    // Simulate work
    kernel->recordExecution();
    
    // Trigger economy update
    allocator->economyTick();
    
    // Check current status
    std::cout << "Round " << round 
              << ": Allocation=" << allocator->getAllocation(kernel->getId())
              << ", Currency=" << kernel->getCurrencyBalance()
              << ", Activation=" << kernel->getActivationLevel() << std::endl;
}
```

## Integration with Kokkos

The ECAN system integrates seamlessly with existing Kokkos constructs:

### Parallel Execution

```cpp
AttentionAwareParallel<> wrapper(agent, allocator);

// Standard parallel_for with attention awareness
wrapper.parallel_for(label, policy, functor);

// Standard parallel_reduce with attention awareness  
wrapper.parallel_reduce(label, policy, functor, result);
```

### Resource Scaling

The system automatically scales execution based on allocation:
- **High Allocation** (>50%): Full execution
- **Medium Allocation** (10-50%): Scaled execution
- **Low Allocation** (<10%): Minimal or deferred execution

### Cost Management

Currency is spent based on:
- Work size (larger ranges cost more)
- Current allocation (higher allocation reduces cost per unit)
- Resource utilization patterns

## Performance Considerations

### Memory Overhead

- Minimal per-agent overhead (~100 bytes)
- Shared allocator state across all agents
- Atomic operations for thread safety

### Computational Overhead

- O(n) allocation updates where n = number of agents
- Configurable update frequency to balance responsiveness vs. overhead
- Lock-free fast paths for common operations

### Scalability

- Tested with hundreds of concurrent agents
- Linear scaling with agent count
- Efficient memory usage patterns

## Configuration Parameters

### AttentionAllocator Parameters

- `totalResources`: Total cognitive resources available (default: 1000.0)
- `updateFrequency`: Economy update frequency in ticks (default: 100)

### Economic Constants

- `BASE_CURRENCY`: Initial currency for new agents (100.0)
- `MIN_ALLOCATION`: Minimum allocation threshold (0.01)

### Activation Decay

- Time-based exponential decay with configurable half-life
- Execution-based activation boosts
- Configurable boost factors

## Testing and Validation

The system includes comprehensive tests covering:

- Basic agent functionality
- Economic resource management
- Thread safety and concurrent access
- Demand-based allocation algorithms
- Integration with Kokkos parallel constructs
- Resource scaling and cost calculations

Run tests with:
```bash
make test_attention && ./test_attention
```

Run demonstration with:
```bash
make demo_attention && ./demo_attention
```

## Future Extensions

Potential enhancements include:

- **Machine Learning Integration**: Use RL to optimize allocation strategies
- **Hierarchical Agents**: Multi-level agent hierarchies
- **Network Effects**: Agent-to-agent resource trading
- **Predictive Allocation**: Allocation based on predicted future needs
- **Quality of Service**: SLA-based allocation guarantees

## References

This implementation is inspired by:
- Economic Attention Networks (ECAN) from AGI research
- Market-based resource allocation systems
- Attention mechanisms in neural networks
- Economic models in distributed systems