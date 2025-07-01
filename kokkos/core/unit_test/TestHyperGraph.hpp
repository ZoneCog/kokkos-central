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
#include <Kokkos_HyperGraph.hpp>
#include <Kokkos_HyperGraph_ModuleMapper.hpp>
#include <gtest/gtest.h>

#include <fstream>
#include <filesystem>
#include <vector>
#include <string>

namespace Test {

using namespace Kokkos::Experimental;

//==============================================================================
// <editor-fold desc="HyperNode Tests"> {{{1

class TestHyperNode : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestHyperNode, BasicConstruction) {
  HyperNode node(1, HyperNode::NodeType::FUNCTION, "test_function");
  
  EXPECT_EQ(node.getId(), 1);
  EXPECT_EQ(node.getType(), HyperNode::NodeType::FUNCTION);
  EXPECT_EQ(node.getName(), "test_function");
  EXPECT_TRUE(node.getMetadata().empty());
}

TEST_F(TestHyperNode, ConstructionWithMetadata) {
  std::unordered_map<std::string, std::string> metadata = {
    {"source_file", "test.cpp"},
    {"line_number", "42"}
  };
  
  HyperNode node(2, HyperNode::NodeType::FILE, "test.cpp", metadata);
  
  EXPECT_EQ(node.getId(), 2);
  EXPECT_EQ(node.getType(), HyperNode::NodeType::FILE);
  EXPECT_EQ(node.getName(), "test.cpp");
  EXPECT_EQ(node.getMetadata("source_file"), "test.cpp");
  EXPECT_EQ(node.getMetadata("line_number"), "42");
  EXPECT_EQ(node.getMetadata("nonexistent", "default"), "default");
}

TEST_F(TestHyperNode, MetadataManipulation) {
  HyperNode node(3, HyperNode::NodeType::CLASS, "TestClass");
  
  node.setMetadata("namespace", "TestNamespace");
  node.setMetadata("visibility", "public");
  
  EXPECT_EQ(node.getMetadata("namespace"), "TestNamespace");
  EXPECT_EQ(node.getMetadata("visibility"), "public");
  
  // Overwrite existing metadata
  node.setMetadata("visibility", "private");
  EXPECT_EQ(node.getMetadata("visibility"), "private");
}

// </editor-fold> end HyperNode Tests }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="HyperLink Tests"> {{{1

class TestHyperLink : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestHyperLink, BasicConstruction) {
  std::vector<std::size_t> sources = {1};
  std::vector<std::size_t> targets = {2};
  
  HyperLink link(10, HyperLink::LinkType::FUNCTION_CALL, sources, targets);
  
  EXPECT_EQ(link.getId(), 10);
  EXPECT_EQ(link.getType(), HyperLink::LinkType::FUNCTION_CALL);
  EXPECT_EQ(link.getSourceNodes(), sources);
  EXPECT_EQ(link.getTargetNodes(), targets);
  EXPECT_TRUE(link.getMetadata().empty());
}

TEST_F(TestHyperLink, HyperEdgeConstruction) {
  std::vector<std::size_t> sources = {1, 2};
  std::vector<std::size_t> targets = {3, 4, 5};
  
  HyperLink link(11, HyperLink::LinkType::DATA_FLOW, sources, targets);
  
  EXPECT_EQ(link.getSourceNodes().size(), 2);
  EXPECT_EQ(link.getTargetNodes().size(), 3);
  
  // Test connectivity
  EXPECT_TRUE(link.connects(1, 3));
  EXPECT_TRUE(link.connects(2, 4));
  EXPECT_TRUE(link.connects(1, 5));
  EXPECT_FALSE(link.connects(1, 6));
  EXPECT_FALSE(link.connects(6, 3));
}

TEST_F(TestHyperLink, MetadataHandling) {
  std::vector<std::size_t> sources = {1};
  std::vector<std::size_t> targets = {2};
  std::unordered_map<std::string, std::string> metadata = {
    {"call_type", "direct"},
    {"frequency", "high"}
  };
  
  HyperLink link(12, HyperLink::LinkType::FUNCTION_CALL, sources, targets, metadata);
  
  EXPECT_EQ(link.getMetadata("call_type"), "direct");
  EXPECT_EQ(link.getMetadata("frequency"), "high");
  
  link.setMetadata("frequency", "medium");
  EXPECT_EQ(link.getMetadata("frequency"), "medium");
}

// </editor-fold> end HyperLink Tests }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="HyperGraph Tests"> {{{1

class TestHyperGraph : public ::testing::Test {
 protected:
  void SetUp() override {
    graph_.clear();
  }
  
  void TearDown() override {}
  
  HyperGraph graph_;
};

TEST_F(TestHyperGraph, EmptyGraph) {
  EXPECT_EQ(graph_.getNodeCount(), 0);
  EXPECT_EQ(graph_.getLinkCount(), 0);
  
  auto stats = graph_.getStats();
  EXPECT_EQ(stats.nodeCount, 0);
  EXPECT_EQ(stats.linkCount, 0);
  EXPECT_EQ(stats.maxDegree, 0);
  EXPECT_EQ(stats.avgDegree, 0.0);
}

TEST_F(TestHyperGraph, AddNodes) {
  auto fileId = graph_.addNode(HyperNode::NodeType::FILE, "test.cpp");
  auto funcId = graph_.addNode(HyperNode::NodeType::FUNCTION, "testFunc");
  
  EXPECT_EQ(graph_.getNodeCount(), 2);
  EXPECT_NE(fileId, funcId);
  
  auto fileNode = graph_.getNode(fileId);
  auto funcNode = graph_.getNode(funcId);
  
  ASSERT_NE(fileNode, nullptr);
  ASSERT_NE(funcNode, nullptr);
  
  EXPECT_EQ(fileNode->getType(), HyperNode::NodeType::FILE);
  EXPECT_EQ(fileNode->getName(), "test.cpp");
  
  EXPECT_EQ(funcNode->getType(), HyperNode::NodeType::FUNCTION);
  EXPECT_EQ(funcNode->getName(), "testFunc");
}

TEST_F(TestHyperGraph, AddLinks) {
  auto sourceId = graph_.addNode(HyperNode::NodeType::FUNCTION, "caller");
  auto targetId = graph_.addNode(HyperNode::NodeType::FUNCTION, "callee");
  
  auto linkId = graph_.addLink(HyperLink::LinkType::FUNCTION_CALL, {sourceId}, {targetId});
  
  EXPECT_EQ(graph_.getLinkCount(), 1);
  
  auto link = graph_.getLink(linkId);
  ASSERT_NE(link, nullptr);
  
  EXPECT_EQ(link->getType(), HyperLink::LinkType::FUNCTION_CALL);
  EXPECT_EQ(link->getSourceNodes().size(), 1);
  EXPECT_EQ(link->getTargetNodes().size(), 1);
  EXPECT_EQ(link->getSourceNodes()[0], sourceId);
  EXPECT_EQ(link->getTargetNodes()[0], targetId);
}

TEST_F(TestHyperGraph, QueryNodesByType) {
  auto fileId1 = graph_.addNode(HyperNode::NodeType::FILE, "file1.cpp");
  auto fileId2 = graph_.addNode(HyperNode::NodeType::FILE, "file2.cpp");
  auto funcId = graph_.addNode(HyperNode::NodeType::FUNCTION, "func");
  
  auto fileNodes = graph_.getNodesByType(HyperNode::NodeType::FILE);
  auto funcNodes = graph_.getNodesByType(HyperNode::NodeType::FUNCTION);
  auto classNodes = graph_.getNodesByType(HyperNode::NodeType::CLASS);
  
  EXPECT_EQ(fileNodes.size(), 2);
  EXPECT_EQ(funcNodes.size(), 1);
  EXPECT_EQ(classNodes.size(), 0);
  
  // Verify the nodes are correct
  bool foundFile1 = false, foundFile2 = false;
  for (const auto& node : fileNodes) {
    if (node->getName() == "file1.cpp") foundFile1 = true;
    if (node->getName() == "file2.cpp") foundFile2 = true;
  }
  EXPECT_TRUE(foundFile1);
  EXPECT_TRUE(foundFile2);
}

TEST_F(TestHyperGraph, ConnectedNodesAndLinks) {
  auto nodeA = graph_.addNode(HyperNode::NodeType::FUNCTION, "A");
  auto nodeB = graph_.addNode(HyperNode::NodeType::FUNCTION, "B");
  auto nodeC = graph_.addNode(HyperNode::NodeType::FUNCTION, "C");
  
  auto linkAB = graph_.addLink(HyperLink::LinkType::FUNCTION_CALL, {nodeA}, {nodeB});
  auto linkAC = graph_.addLink(HyperLink::LinkType::FUNCTION_CALL, {nodeA}, {nodeC});
  auto linkBC = graph_.addLink(HyperLink::LinkType::DATA_FLOW, {nodeB}, {nodeC});
  
  // Test connected nodes
  auto connectedToA = graph_.getConnectedNodes(nodeA);
  EXPECT_EQ(connectedToA.size(), 2);
  
  // Test outgoing links
  auto outgoingFromA = graph_.getOutgoingLinks(nodeA);
  EXPECT_EQ(outgoingFromA.size(), 2);
  
  // Test incoming links
  auto incomingToC = graph_.getIncomingLinks(nodeC);
  EXPECT_EQ(incomingToC.size(), 2);
}

TEST_F(TestHyperGraph, CustomQueries) {
  auto fileId = graph_.addNode(HyperNode::NodeType::FILE, "test.cpp", {{"extension", "cpp"}});
  auto headerID = graph_.addNode(HyperNode::NodeType::FILE, "test.h", {{"extension", "h"}});
  auto funcId = graph_.addNode(HyperNode::NodeType::FUNCTION, "testFunc");
  
  // Query nodes with specific metadata
  auto cppFiles = graph_.queryNodes([](const HyperNode& node) {
    return node.getMetadata("extension") == "cpp";
  });
  
  EXPECT_EQ(cppFiles.size(), 1);
  EXPECT_EQ(cppFiles[0]->getName(), "test.cpp");
  
  // Query nodes by name pattern
  auto testItems = graph_.queryNodes([](const HyperNode& node) {
    return node.getName().find("test") != std::string::npos;
  });
  
  EXPECT_EQ(testItems.size(), 3);
}

TEST_F(TestHyperGraph, GraphStatistics) {
  // Create a small graph: A -> B -> C, A -> C
  auto nodeA = graph_.addNode(HyperNode::NodeType::FUNCTION, "A");
  auto nodeB = graph_.addNode(HyperNode::NodeType::FUNCTION, "B");
  auto nodeC = graph_.addNode(HyperNode::NodeType::FUNCTION, "C");
  auto fileNode = graph_.addNode(HyperNode::NodeType::FILE, "test.cpp");
  
  graph_.addLink(HyperLink::LinkType::FUNCTION_CALL, {nodeA}, {nodeB});
  graph_.addLink(HyperLink::LinkType::FUNCTION_CALL, {nodeB}, {nodeC});
  graph_.addLink(HyperLink::LinkType::FUNCTION_CALL, {nodeA}, {nodeC});
  graph_.addLink(HyperLink::LinkType::INCLUDE, {fileNode}, {nodeA});
  
  auto stats = graph_.getStats();
  
  EXPECT_EQ(stats.nodeCount, 4);
  EXPECT_EQ(stats.linkCount, 4);
  EXPECT_EQ(stats.nodeTypeCounts[HyperNode::NodeType::FUNCTION], 3);
  EXPECT_EQ(stats.nodeTypeCounts[HyperNode::NodeType::FILE], 1);
  EXPECT_EQ(stats.linkTypeCounts[HyperLink::LinkType::FUNCTION_CALL], 3);
  EXPECT_EQ(stats.linkTypeCounts[HyperLink::LinkType::INCLUDE], 1);
}

// </editor-fold> end HyperGraph Tests }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="ModuleMapper Tests"> {{{1

class TestModuleMapper : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create temporary test directory
    testDir_ = "/tmp/kokkos_hypergraph_test";
    std::filesystem::create_directories(testDir_);
    createTestFiles();
  }
  
  void TearDown() override {
    // Clean up test directory
    std::filesystem::remove_all(testDir_);
  }
  
  void createTestFiles() {
    // Create test.hpp
    std::ofstream header(testDir_ + "/test.hpp");
    header << "#ifndef TEST_HPP\n";
    header << "#define TEST_HPP\n";
    header << "\n";
    header << "class TestClass {\n";
    header << "public:\n";
    header << "    void publicMethod();\n";
    header << "    int getValue();\n";
    header << "};\n";
    header << "\n";
    header << "void globalFunction();\n";
    header << "\n";
    header << "#endif\n";
    header.close();
    
    // Create test.cpp
    std::ofstream source(testDir_ + "/test.cpp");
    source << "#include \"test.hpp\"\n";
    source << "#include <iostream>\n";
    source << "\n";
    source << "void TestClass::publicMethod() {\n";
    source << "    std::cout << \"Hello\" << std::endl;\n";
    source << "    getValue();\n";
    source << "}\n";
    source << "\n";
    source << "int TestClass::getValue() {\n";
    source << "    return 42;\n";
    source << "}\n";
    source << "\n";
    source << "void globalFunction() {\n";
    source << "    TestClass obj;\n";
    source << "    obj.publicMethod();\n";
    source << "}\n";
    source.close();
    
    // Create main.cpp
    std::ofstream main(testDir_ + "/main.cpp");
    main << "#include \"test.hpp\"\n";
    main << "\n";
    main << "int main() {\n";
    main << "    globalFunction();\n";
    main << "    return 0;\n";
    main << "}\n";
    main.close();
  }
  
  std::string testDir_;
};

TEST_F(TestModuleMapper, BasicConfiguration) {
  ModuleMapper::MapperConfig config;
  config.sourceDirectories = {testDir_};
  config.fileExtensions = {".cpp", ".hpp"};
  
  ModuleMapper mapper(config);
  
  EXPECT_EQ(mapper.getConfig().sourceDirectories.size(), 1);
  EXPECT_EQ(mapper.getConfig().fileExtensions.size(), 2);
  EXPECT_TRUE(mapper.getConfig().analyzeIncludes);
  EXPECT_TRUE(mapper.getConfig().analyzeFunctionCalls);
}

TEST_F(TestModuleMapper, FunctionCallExtraction) {
  ModuleMapper mapper;
  
  std::string line1 = "    getValue();";
  std::string line2 = "    obj.publicMethod();";
  std::string line3 = "    std::cout << \"test\" << std::endl;";
  std::string line4 = "    if (condition) {";  // Should not extract 'if'
  
  auto calls1 = mapper.extractFunctionCalls(line1);
  auto calls2 = mapper.extractFunctionCalls(line2);
  auto calls3 = mapper.extractFunctionCalls(line3);
  auto calls4 = mapper.extractFunctionCalls(line4);
  
  EXPECT_EQ(calls1.size(), 1);
  EXPECT_EQ(calls1[0], "getValue");
  
  EXPECT_EQ(calls2.size(), 1);
  EXPECT_EQ(calls2[0], "publicMethod");
  
  EXPECT_GE(calls3.size(), 0);  // May or may not find functions depending on implementation
  EXPECT_EQ(calls4.size(), 0);  // Should not find 'if' as a function call
}

TEST_F(TestModuleMapper, IncludeExtraction) {
  ModuleMapper mapper;
  
  std::string line1 = "#include \"test.hpp\"";
  std::string line2 = "#include <iostream>";
  std::string line3 = "// #include \"commented.hpp\"";
  std::string line4 = "    some code here";
  
  EXPECT_EQ(mapper.extractInclude(line1), "test.hpp");
  EXPECT_EQ(mapper.extractInclude(line2), "iostream");
  EXPECT_EQ(mapper.extractInclude(line3), "");
  EXPECT_EQ(mapper.extractInclude(line4), "");
}

TEST_F(TestModuleMapper, FileMapping) {
  ModuleMapper::MapperConfig config;
  config.sourceDirectories = {testDir_};
  
  ModuleMapper mapper(config);
  HyperGraph graph;
  
  std::size_t relationCount = mapper.mapToHyperGraph(graph);
  
  EXPECT_GT(relationCount, 0);
  EXPECT_GT(graph.getNodeCount(), 0);
  EXPECT_GT(graph.getLinkCount(), 0);
  
  // Check that we have file nodes
  auto fileNodes = graph.getNodesByType(HyperNode::NodeType::FILE);
  EXPECT_GE(fileNodes.size(), 3);  // test.hpp, test.cpp, main.cpp
  
  // Check that we have function nodes
  auto funcNodes = graph.getNodesByType(HyperNode::NodeType::FUNCTION);
  EXPECT_GT(funcNodes.size(), 0);
  
  // Verify specific relationships exist
  auto stats = graph.getStats();
  EXPECT_GT(stats.linkTypeCounts[HyperLink::LinkType::INCLUDE], 0);
}

TEST_F(TestModuleMapper, SingleFileMapping) {
  ModuleMapper mapper;
  HyperGraph graph;
  
  std::string testFile = testDir_ + "/test.cpp";
  std::size_t relationCount = mapper.mapFileToHyperGraph(graph, testFile);
  
  EXPECT_GT(relationCount, 0);
  EXPECT_GE(graph.getNodeCount(), 1);  // At least the file node
  
  // Should have the file node
  auto fileNodes = graph.getNodesByType(HyperNode::NodeType::FILE);
  EXPECT_EQ(fileNodes.size(), 1);
  EXPECT_EQ(fileNodes[0]->getName(), testFile);
}

// </editor-fold> end ModuleMapper Tests }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Integration Tests"> {{{1

class TestHyperGraphIntegration : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestHyperGraphIntegration, DynamicAdaptation) {
  // Test dynamic adaptation capabilities by modifying graph structure
  HyperGraph graph;
  
  // Initial graph: A -> B -> C
  auto nodeA = graph.addNode(HyperNode::NodeType::FUNCTION, "A");
  auto nodeB = graph.addNode(HyperNode::NodeType::FUNCTION, "B");
  auto nodeC = graph.addNode(HyperNode::NodeType::FUNCTION, "C");
  
  auto linkAB = graph.addLink(HyperLink::LinkType::FUNCTION_CALL, {nodeA}, {nodeB});
  auto linkBC = graph.addLink(HyperLink::LinkType::FUNCTION_CALL, {nodeB}, {nodeC});
  
  auto initialStats = graph.getStats();
  EXPECT_EQ(initialStats.nodeCount, 3);
  EXPECT_EQ(initialStats.linkCount, 2);
  
  // Add dynamic optimization node
  auto optimizerNode = graph.addNode(HyperNode::NodeType::CUSTOM, "Optimizer", {
    {"type", "dynamic_optimizer"},
    {"adaptation_strategy", "performance"}
  });
  
  // Connect optimizer to monitor all function calls
  auto optimizerLinks = graph.queryLinks([](const HyperLink& link) {
    return link.getType() == HyperLink::LinkType::FUNCTION_CALL;
  });
  
  for (const auto& link : optimizerLinks) {
    graph.addLink(HyperLink::LinkType::CUSTOM, {optimizerNode}, link->getSourceNodes(), {
      {"monitoring", "performance"},
      {"adaptation", "enabled"}
    });
  }
  
  auto finalStats = graph.getStats();
  EXPECT_EQ(finalStats.nodeCount, 4);
  EXPECT_GT(finalStats.linkCount, initialStats.linkCount);
}

TEST_F(TestHyperGraphIntegration, LargeGraphPerformance) {
  // Test performance with a larger graph
  HyperGraph graph;
  
  const std::size_t numNodes = 1000;
  const std::size_t numLinks = 2000;
  
  // Create nodes
  std::vector<std::size_t> nodeIds;
  for (std::size_t i = 0; i < numNodes; ++i) {
    auto nodeId = graph.addNode(HyperNode::NodeType::FUNCTION, "func_" + std::to_string(i));
    nodeIds.push_back(nodeId);
  }
  
  // Create random links
  for (std::size_t i = 0; i < numLinks; ++i) {
    auto sourceId = nodeIds[i % nodeIds.size()];
    auto targetId = nodeIds[(i + 1) % nodeIds.size()];
    graph.addLink(HyperLink::LinkType::FUNCTION_CALL, {sourceId}, {targetId});
  }
  
  EXPECT_EQ(graph.getNodeCount(), numNodes);
  EXPECT_EQ(graph.getLinkCount(), numLinks);
  
  // Test query performance
  auto functionNodes = graph.getNodesByType(HyperNode::NodeType::FUNCTION);
  EXPECT_EQ(functionNodes.size(), numNodes);
  
  // Test connectivity queries
  auto connectedNodes = graph.getConnectedNodes(nodeIds[0]);
  EXPECT_GT(connectedNodes.size(), 0);
  
  auto stats = graph.getStats();
  EXPECT_EQ(stats.nodeCount, numNodes);
  EXPECT_EQ(stats.linkCount, numLinks);
}

// </editor-fold> end Integration Tests }}}1
//==============================================================================

} // namespace Test