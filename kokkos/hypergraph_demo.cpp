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

/// @file HyperGraphExample.cpp
/// @brief Comprehensive example demonstrating AtomSpace-style hypergraph mapping
/// for inter-module relations in the Kokkos ecosystem

#include <Kokkos_HyperGraph.hpp>
#include <Kokkos_HyperGraph_ModuleMapper.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace Kokkos::Experimental;

/// @brief Create a demonstration project structure
void createDemoProject() {
    std::filesystem::create_directories("/tmp/demo_project/src");
    std::filesystem::create_directories("/tmp/demo_project/include");
    
    // Create header file
    std::ofstream header("/tmp/demo_project/include/math_utils.hpp");
    header << "#pragma once\n";
    header << "#include <vector>\n";
    header << "\n";
    header << "namespace MathUtils {\n";
    header << "    class Calculator {\n";
    header << "    public:\n";
    header << "        double add(double a, double b);\n";
    header << "        double multiply(double a, double b);\n";
    header << "        std::vector<double> processArray(const std::vector<double>& data);\n";
    header << "    };\n";
    header << "    \n";
    header << "    double factorial(int n);\n";
    header << "    double power(double base, int exp);\n";
    header << "}\n";
    header.close();
    
    // Create implementation file
    std::ofstream impl("/tmp/demo_project/src/math_utils.cpp");
    impl << "#include \"math_utils.hpp\"\n";
    impl << "#include <algorithm>\n";
    impl << "#include <numeric>\n";
    impl << "\n";
    impl << "namespace MathUtils {\n";
    impl << "    double Calculator::add(double a, double b) {\n";
    impl << "        return a + b;\n";
    impl << "    }\n";
    impl << "    \n";
    impl << "    double Calculator::multiply(double a, double b) {\n";
    impl << "        return a * b;\n";
    impl << "    }\n";
    impl << "    \n";
    impl << "    std::vector<double> Calculator::processArray(const std::vector<double>& data) {\n";
    impl << "        std::vector<double> result = data;\n";
    impl << "        std::transform(result.begin(), result.end(), result.begin(),\n";
    impl << "                      [this](double x) { return multiply(x, 2.0); });\n";
    impl << "        return result;\n";
    impl << "    }\n";
    impl << "    \n";
    impl << "    double factorial(int n) {\n";
    impl << "        if (n <= 1) return 1.0;\n";
    impl << "        return n * factorial(n - 1);\n";
    impl << "    }\n";
    impl << "    \n";
    impl << "    double power(double base, int exp) {\n";
    impl << "        if (exp == 0) return 1.0;\n";
    impl << "        if (exp < 0) return 1.0 / power(base, -exp);\n";
    impl << "        double result = 1.0;\n";
    impl << "        for (int i = 0; i < exp; ++i) {\n";
    impl << "            result = Calculator().multiply(result, base);\n";
    impl << "        }\n";
    impl << "        return result;\n";
    impl << "    }\n";
    impl << "}\n";
    impl.close();
    
    // Create main application
    std::ofstream main("/tmp/demo_project/src/main.cpp");
    main << "#include \"math_utils.hpp\"\n";
    main << "#include <iostream>\n";
    main << "#include <vector>\n";
    main << "\n";
    main << "int main() {\n";
    main << "    MathUtils::Calculator calc;\n";
    main << "    \n";
    main << "    double sum = calc.add(3.14, 2.86);\n";
    main << "    double product = calc.multiply(sum, 2.0);\n";
    main << "    \n";
    main << "    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};\n";
    main << "    auto processed = calc.processArray(data);\n";
    main << "    \n";
    main << "    double fact = MathUtils::factorial(5);\n";
    main << "    double pow_result = MathUtils::power(2.0, 8);\n";
    main << "    \n";
    main << "    std::cout << \"Results: sum=\" << sum\n";
    main << "              << \", product=\" << product\n";
    main << "              << \", factorial=\" << fact\n";
    main << "              << \", power=\" << pow_result << std::endl;\n";
    main << "    \n";
    main << "    return 0;\n";
    main << "}\n";
    main.close();
}

/// @brief Demonstrate comprehensive inter-module relation mapping
void demonstrateInterModuleMapping() {
    std::cout << "=== AtomSpace-style HyperGraph: Inter-Module Relation Mapping ===" << std::endl;
    
    // Create demo project
    createDemoProject();
    
    HyperGraph graph;
    ModuleMapper::MapperConfig config;
    config.sourceDirectories = {"/tmp/demo_project"};
    config.fileExtensions = {".cpp", ".hpp"};
    config.analyzeIncludes = true;
    config.analyzeFunctionCalls = true;
    config.analyzeDataFlow = true;
    
    ModuleMapper mapper(config);
    
    std::cout << "\n1. Mapping source files to hypergraph..." << std::endl;
    std::size_t relationCount = mapper.mapToHyperGraph(graph);
    
    std::cout << "   Mapped " << relationCount << " relations" << std::endl;
    std::cout << "   Graph contains:" << std::endl;
    std::cout << "     - " << graph.getNodeCount() << " nodes" << std::endl;
    std::cout << "     - " << graph.getLinkCount() << " links" << std::endl;
    
    // Analyze node types
    std::cout << "\n2. Analyzing node types..." << std::endl;
    auto fileNodes = graph.getNodesByType(HyperNode::NodeType::FILE);
    auto functionNodes = graph.getNodesByType(HyperNode::NodeType::FUNCTION);
    
    std::cout << "   File nodes (" << fileNodes.size() << "):" << std::endl;
    for (const auto& node : fileNodes) {
        std::cout << "     - " << node->getName() << std::endl;
    }
    
    std::cout << "   Function nodes (" << functionNodes.size() << "):" << std::endl;
    for (const auto& node : functionNodes) {
        std::cout << "     - " << node->getName() << std::endl;
    }
    
    // Demonstrate custom queries for agentic adaptation
    std::cout << "\n3. Custom queries for dynamic adaptation..." << std::endl;
    
    // Find functions that might be hot paths
    auto complexFunctions = graph.queryNodes([&graph](const HyperNode& node) {
        if (node.getType() != HyperNode::NodeType::FUNCTION) return false;
        auto connectedNodes = graph.getConnectedNodes(node.getId());
        return connectedNodes.size() > 2; // Functions with many connections
    });
    
    std::cout << "   Complex functions (hot paths): " << complexFunctions.size() << std::endl;
    for (const auto& func : complexFunctions) {
        auto connections = graph.getConnectedNodes(func->getId());
        std::cout << "     - " << func->getName() << " (connected to " 
                  << connections.size() << " nodes)" << std::endl;
    }
    
    // Add dynamic optimization nodes
    std::cout << "\n4. Adding dynamic optimization layer..." << std::endl;
    
    auto performanceAnalyzer = graph.addNode(HyperNode::NodeType::CUSTOM, "PerformanceAnalyzer", {
        {"type", "dynamic_analyzer"},
        {"strategy", "call_frequency_analysis"},
        {"optimization_target", "latency"}
    });
    
    auto memoryOptimizer = graph.addNode(HyperNode::NodeType::CUSTOM, "MemoryOptimizer", {
        {"type", "dynamic_optimizer"}, 
        {"strategy", "cache_optimization"},
        {"optimization_target", "memory_usage"}
    });
    
    auto agenticController = graph.addNode(HyperNode::NodeType::CUSTOM, "AgenticController", {
        {"type", "agentic_controller"},
        {"strategy", "adaptive_optimization"},
        {"learning_algorithm", "reinforcement_learning"}
    });
    
    // Connect optimizer components
    graph.addLink(HyperLink::LinkType::CUSTOM, {performanceAnalyzer}, {agenticController}, {
        {"relationship", "feeds_data_to"},
        {"data_type", "performance_metrics"}
    });
    
    graph.addLink(HyperLink::LinkType::CUSTOM, {memoryOptimizer}, {agenticController}, {
        {"relationship", "feeds_data_to"},
        {"data_type", "memory_metrics"}
    });
    
    // Connect agentic controller to complex functions for monitoring
    for (const auto& func : complexFunctions) {
        graph.addLink(HyperLink::LinkType::CUSTOM, {agenticController}, {func->getId()}, {
            {"relationship", "monitors"},
            {"optimization_type", "dynamic_adaptation"}
        });
    }
    
    std::cout << "   Added optimization layer with agentic adaptation" << std::endl;
    std::cout << "     - Performance analyzer: monitors call frequencies" << std::endl;
    std::cout << "     - Memory optimizer: tracks memory usage patterns" << std::endl;
    std::cout << "     - Agentic controller: learns and adapts optimization strategies" << std::endl;
    
    // Show final statistics
    auto stats = graph.getStats();
    std::cout << "\n5. Final hypergraph statistics..." << std::endl;
    std::cout << "   Total nodes: " << stats.nodeCount << std::endl;
    std::cout << "   Total links: " << stats.linkCount << std::endl;
    std::cout << "   Maximum degree: " << stats.maxDegree << std::endl;
    std::cout << "   Average degree: " << stats.avgDegree << std::endl;
    
    // Demonstrate query capabilities for different relationship types
    std::cout << "\n6. Querying specific relationship types..." << std::endl;
    
    auto includeLinks = graph.queryLinks([](const HyperLink& link) {
        return link.getType() == HyperLink::LinkType::INCLUDE;
    });
    
    auto functionCallLinks = graph.queryLinks([](const HyperLink& link) {
        return link.getType() == HyperLink::LinkType::FUNCTION_CALL;
    });
    
    auto optimizationLinks = graph.queryLinks([](const HyperLink& link) {
        return link.getType() == HyperLink::LinkType::CUSTOM;
    });
    
    std::cout << "   Include relationships: " << includeLinks.size() << std::endl;
    std::cout << "   Function call relationships: " << functionCallLinks.size() << std::endl;
    std::cout << "   Optimization relationships: " << optimizationLinks.size() << std::endl;
    
    std::cout << "\n=== Demonstration Complete ===" << std::endl;
    std::cout << "✓ Inter-module relations successfully mapped to hypergraph" << std::endl;
    std::cout << "✓ Files and functions represented as nodes with metadata" << std::endl;
    std::cout << "✓ Calls, includes, and data flow represented as typed links" << std::endl;
    std::cout << "✓ Dynamic agentic adaptation layer implemented" << std::endl;
    std::cout << "✓ Complex queries and optimization strategies demonstrated" << std::endl;
    
    // Cleanup
    std::filesystem::remove_all("/tmp/demo_project");
}

int main() {
    try {
        demonstrateInterModuleMapping();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}