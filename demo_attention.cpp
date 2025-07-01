#define KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>

using namespace Kokkos::Experimental;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    
    std::cout << "=== ECAN-Style Attention Allocation Demonstration ===" << std::endl;
    std::cout << std::endl;
    
    // Create attention allocator with 1000 units of cognitive resources
    auto allocator = std::make_shared<AttentionAllocator<>>(1000.0, 5);
    
    // Create agentic kernel factory
    AgenticKernelFactory<> factory(allocator);
    
    // Create several agentic kernels with different initial activations
    std::vector<std::shared_ptr<AgenticKernel>> agents;
    std::vector<std::string> agentNames = {"HighPriority", "MediumPriority", "LowPriority", "VeryLowPriority"};
    std::vector<double> initialActivations = {5.0, 2.0, 1.0, 0.2};
    
    std::cout << "Creating " << agentNames.size() << " agentic kernels:" << std::endl;
    for (size_t i = 0; i < agentNames.size(); ++i) {
        auto agent = factory.createKernel(initialActivations[i]);
        agents.push_back(agent);
        
        std::cout << "  " << agentNames[i] << " (ID: " << agent->getId() 
                  << ", Activation: " << initialActivations[i] 
                  << ", Currency: " << agent->getCurrencyBalance() << ")" << std::endl;
    }
    std::cout << std::endl;
    
    // Show initial allocation after first economy update
    std::cout << "Initial resource allocations:" << std::endl;
    allocator->updateAllocations();
    for (size_t i = 0; i < agents.size(); ++i) {
        double allocation = allocator->getAllocation(agents[i]->getId());
        std::cout << "  " << agentNames[i] << ": " << std::fixed << std::setprecision(2) 
                  << allocation << " units" << std::endl;
    }
    std::cout << std::endl;
    
    // Simulate activity and show dynamic allocation changes
    std::cout << "Simulating kernel activity and economic dynamics..." << std::endl;
    
    for (int round = 1; round <= 5; ++round) {
        std::cout << "\n--- Round " << round << " ---" << std::endl;
        
        // Simulate different levels of activity
        agents[0]->recordExecution(); // High priority gets very active
        agents[0]->recordExecution();
        agents[0]->recordExecution();
        
        agents[1]->recordExecution(); // Medium priority gets moderately active
        agents[1]->recordExecution();
        
        agents[2]->recordExecution(); // Low priority gets slightly active
        
        // Very low priority remains inactive
        
        // Trigger economy updates
        for (int tick = 0; tick < 10; ++tick) {
            allocator->economyTick();
        }
        
        // Show updated allocations and economic status
        std::cout << "Updated allocations and economic status:" << std::endl;
        for (size_t i = 0; i < agents.size(); ++i) {
            double allocation = allocator->getAllocation(agents[i]->getId());
            double activation = agents[i]->getActivationLevel();
            double currency = agents[i]->getCurrencyBalance();
            size_t executions = agents[i]->getExecutionCount();
            
            std::cout << "  " << agentNames[i] << ": " 
                      << "Alloc=" << std::fixed << std::setprecision(1) << allocation
                      << ", Act=" << std::setprecision(2) << activation
                      << ", Curr=" << std::setprecision(1) << currency
                      << ", Exec=" << executions << std::endl;
        }
        
        // Simulate some currency spending based on allocation
        for (size_t i = 0; i < agents.size(); ++i) {
            double allocation = allocator->getAllocation(agents[i]->getId());
            if (allocation > 0.1) {
                // Spend currency proportional to allocation
                double spendAmount = allocation * 0.1;
                agents[i]->spendCurrency(spendAmount);
            }
        }
    }
    
    std::cout << "\n=== Final Economic State ===" << std::endl;
    double totalAllocation = 0.0;
    for (size_t i = 0; i < agents.size(); ++i) {
        double allocation = allocator->getAllocation(agents[i]->getId());
        totalAllocation += allocation;
        std::cout << agentNames[i] << " final allocation: " 
                  << std::fixed << std::setprecision(2) << allocation 
                  << " (" << (allocation / allocator->getTotalResources() * 100.0) << "%)" << std::endl;
    }
    
    std::cout << "\nTotal allocated: " << totalAllocation << " / " 
              << allocator->getTotalResources() << " available resources" << std::endl;
    
    std::cout << "\n=== Demonstration of Attention-Aware Parallel Execution ===" << std::endl;
    
    // Create attention-aware parallel wrapper for the high-priority agent
    AttentionAwareParallel<> attentionParallel(agents[0], allocator);
    
    std::cout << "Executing parallel kernel with attention allocation..." << std::endl;
    
    constexpr int N = 1000;
    
    double currencyBefore = agents[0]->getCurrencyBalance();
    double allocationBefore = allocator->getAllocation(agents[0]->getId());
    
    // Execute a simple parallel kernel
    {
        Kokkos::View<double*> data("demonstration_data", N);
        attentionParallel.parallel_for(
            "ECAN_Demo_Kernel",
            Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(int i) {
                data(i) = i * 0.5 + 1.0;
            });
        // View will be automatically deallocated here, before finalize
    }
    
    double currencyAfter = agents[0]->getCurrencyBalance();
    double allocationAfter = allocator->getAllocation(agents[0]->getId());
    
    std::cout << "Execution results:" << std::endl;
    std::cout << "  Currency spent: " << (currencyBefore - currencyAfter) << std::endl;
    std::cout << "  Allocation before: " << allocationBefore << std::endl;
    std::cout << "  Allocation after: " << allocationAfter << std::endl;
    std::cout << "  Activation level: " << agents[0]->getActivationLevel() << std::endl;
    std::cout << "  Total executions: " << agents[0]->getExecutionCount() << std::endl;
    
    std::cout << "\n=== ECAN System Features Demonstrated ===" << std::endl;
    std::cout << "✓ Economic currency management (receive, spend, redistribute)" << std::endl;
    std::cout << "✓ Demand-based allocation based on activation levels" << std::endl;
    std::cout << "✓ Distributed resource allocation among multiple agents" << std::endl;
    std::cout << "✓ Dynamic reallocation based on activity and execution patterns" << std::endl;
    std::cout << "✓ Integration with Kokkos parallel execution constructs" << std::endl;
    std::cout << "✓ Attention-aware resource scaling and cost management" << std::endl;
    
    Kokkos::finalize();
    return 0;
}