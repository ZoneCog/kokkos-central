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

#ifndef KOKKOS_HYPERGRAPH_MODULE_MAPPER_HPP
#define KOKKOS_HYPERGRAPH_MODULE_MAPPER_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_MODULE_MAPPER
#endif

#include <Kokkos_HyperGraph.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <fstream>
#include <filesystem>

namespace Kokkos {
namespace Experimental {

//==============================================================================
// <editor-fold desc="ModuleMapper"> {{{1

/// @brief Utility class for mapping inter-module relations into a hypergraph
/// Analyzes source code to extract files, functions, and their relationships
class ModuleMapper {
 public:
  /// @brief Configuration for module mapping
  struct MapperConfig {
    std::vector<std::string> sourceDirectories;   ///< Directories to scan
    std::vector<std::string> fileExtensions;      ///< File extensions to analyze
    bool analyzeIncludes;                          ///< Parse #include directives
    bool analyzeFunctionCalls;                     ///< Extract function call relationships
    bool analyzeDataFlow;                          ///< Extract data dependencies
    bool analyzeInheritance;                       ///< Extract class inheritance
    std::unordered_set<std::string> excludePatterns; ///< Patterns to exclude
    
    MapperConfig() 
        : analyzeIncludes(true)
        , analyzeFunctionCalls(true)
        , analyzeDataFlow(true)
        , analyzeInheritance(true) {}
  };

  /// @brief Construct a ModuleMapper with configuration
  explicit ModuleMapper(MapperConfig config = MapperConfig{}) 
      : m_config(std::move(config)) {
    // Set default file extensions if none provided
    if (m_config.fileExtensions.empty()) {
      m_config.fileExtensions = {".hpp", ".cpp", ".h", ".c", ".cc", ".cxx"};
    }
  }

  /// @brief Map source code to hypergraph
  /// @param graph Target hypergraph to populate
  /// @return Number of relations mapped
  std::size_t mapToHyperGraph(HyperGraph& graph) {
    std::size_t relationCount = 0;
    std::unordered_map<std::string, std::size_t> fileNodes;
    std::unordered_map<std::string, std::size_t> functionNodes;
    
    // Phase 1: Discover all files and create file nodes
    auto sourceFiles = discoverSourceFiles();
    for (const auto& filePath : sourceFiles) {
      auto nodeId = graph.addNode(HyperNode::NodeType::FILE, filePath, {
        {"path", filePath},
        {"type", "source_file"}
      });
      fileNodes[filePath] = nodeId;
    }

    // Phase 2: Analyze each file for functions and relationships
    for (const auto& filePath : sourceFiles) {
      relationCount += analyzeFile(graph, filePath, fileNodes, functionNodes);
    }

    return relationCount;
  }

  /// @brief Map specific file to hypergraph
  /// @param graph Target hypergraph
  /// @param filePath Path to the file to analyze
  /// @return Number of relations found
  std::size_t mapFileToHyperGraph(HyperGraph& graph, const std::string& filePath) {
    std::unordered_map<std::string, std::size_t> fileNodes;
    std::unordered_map<std::string, std::size_t> functionNodes;
    
    // Create file node
    auto fileNodeId = graph.addNode(HyperNode::NodeType::FILE, filePath, {
      {"path", filePath},
      {"type", "source_file"}
    });
    fileNodes[filePath] = fileNodeId;
    
    return analyzeFile(graph, filePath, fileNodes, functionNodes);
  }

  /// @brief Add function call relationship to hypergraph
  /// @param graph Target hypergraph
  /// @param callerFunction Name of calling function
  /// @param calledFunction Name of called function
  /// @param sourceFile Source file containing the call
  /// @param functionNodes Map of function name to node ID
  /// @return Link ID
  std::size_t addFunctionCall(HyperGraph& graph, 
                              const std::string& callerFunction,
                              const std::string& calledFunction,
                              const std::string& sourceFile,
                              std::unordered_map<std::string, std::size_t>& functionNodes) {
    // Ensure function nodes exist
    if (functionNodes.find(callerFunction) == functionNodes.end()) {
      auto callerId = graph.addNode(HyperNode::NodeType::FUNCTION, callerFunction, {
        {"name", callerFunction},
        {"source_file", sourceFile}
      });
      functionNodes[callerFunction] = callerId;
    }
    
    if (functionNodes.find(calledFunction) == functionNodes.end()) {
      auto calledId = graph.addNode(HyperNode::NodeType::FUNCTION, calledFunction, {
        {"name", calledFunction},
        {"source_file", sourceFile}
      });
      functionNodes[calledFunction] = calledId;
    }
    
    // Create function call link
    return graph.addLink(HyperLink::LinkType::FUNCTION_CALL,
                        {functionNodes[callerFunction]},
                        {functionNodes[calledFunction]}, {
                          {"caller", callerFunction},
                          {"called", calledFunction},
                          {"source_file", sourceFile}
                        });
  }

  /// @brief Add include relationship to hypergraph
  /// @param graph Target hypergraph
  /// @param sourceFile File that includes
  /// @param includedFile File that is included
  /// @param fileNodes Map of file path to node ID
  /// @return Link ID
  std::size_t addIncludeRelation(HyperGraph& graph,
                                 const std::string& sourceFile,
                                 const std::string& includedFile,
                                 std::unordered_map<std::string, std::size_t>& fileNodes) {
    // Ensure file nodes exist
    if (fileNodes.find(sourceFile) == fileNodes.end()) {
      auto sourceId = graph.addNode(HyperNode::NodeType::FILE, sourceFile, {
        {"path", sourceFile},
        {"type", "source_file"}
      });
      fileNodes[sourceFile] = sourceId;
    }
    
    if (fileNodes.find(includedFile) == fileNodes.end()) {
      auto includedId = graph.addNode(HyperNode::NodeType::FILE, includedFile, {
        {"path", includedFile},
        {"type", "header_file"}
      });
      fileNodes[includedFile] = includedId;
    }
    
    // Create include link
    return graph.addLink(HyperLink::LinkType::INCLUDE,
                        {fileNodes[sourceFile]},
                        {fileNodes[includedFile]}, {
                          {"source", sourceFile},
                          {"included", includedFile},
                          {"relationship", "include"}
                        });
  }

  /// @brief Extract function calls from source line
  /// @param line Source code line
  /// @return Vector of function names called
  std::vector<std::string> extractFunctionCalls(const std::string& line) {
    std::vector<std::string> calls;
    
    // Simple regex for function calls - matches identifier followed by (
    std::regex funcCallRegex(R"(\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\()");
    std::sregex_iterator iter(line.begin(), line.end(), funcCallRegex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
      std::string functionName = iter->str(1);
      // Filter out common keywords and operators
      if (!isKeyword(functionName) && !isOperator(functionName)) {
        calls.push_back(functionName);
      }
    }
    
    return calls;
  }

  /// @brief Extract include directives from source line
  /// @param line Source code line
  /// @return Include file path if found, empty string otherwise
  std::string extractInclude(const std::string& line) {
    std::regex includeRegex(R"(^\s*#\s*include\s*[<"]([^>"]+)[>"])");
    std::smatch match;
    
    if (std::regex_match(line, match, includeRegex)) {
      return match[1].str();
    }
    
    return "";
  }

  /// @brief Get mapping configuration
  const MapperConfig& getConfig() const { return m_config; }

  /// @brief Set mapping configuration
  void setConfig(MapperConfig config) { m_config = std::move(config); }

 private:
  MapperConfig m_config;

  /// @brief Discover all source files in configured directories
  std::vector<std::string> discoverSourceFiles() {
    std::vector<std::string> files;
    
    for (const auto& directory : m_config.sourceDirectories) {
      if (std::filesystem::exists(directory)) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
          if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            if (shouldIncludeFile(filePath)) {
              files.push_back(filePath);
            }
          }
        }
      }
    }
    
    return files;
  }

  /// @brief Check if file should be included based on configuration
  bool shouldIncludeFile(const std::string& filePath) {
    // Check extension
    bool hasValidExtension = false;
    for (const auto& ext : m_config.fileExtensions) {
      if (filePath.size() >= ext.size() && 
          filePath.substr(filePath.size() - ext.size()) == ext) {
        hasValidExtension = true;
        break;
      }
    }
    
    if (!hasValidExtension) return false;
    
    // Check exclude patterns
    for (const auto& pattern : m_config.excludePatterns) {
      if (filePath.find(pattern) != std::string::npos) {
        return false;
      }
    }
    
    return true;
  }

  /// @brief Analyze single file for relationships
  std::size_t analyzeFile(HyperGraph& graph, 
                          const std::string& filePath,
                          std::unordered_map<std::string, std::size_t>& fileNodes,
                          std::unordered_map<std::string, std::size_t>& functionNodes) {
    std::size_t relationCount = 0;
    std::ifstream file(filePath);
    if (!file.is_open()) return 0;
    
    std::string line;
    std::string currentFunction;
    
    while (std::getline(file, line)) {
      // Extract include relationships
      if (m_config.analyzeIncludes) {
        std::string includedFile = extractInclude(line);
        if (!includedFile.empty()) {
          addIncludeRelation(graph, filePath, includedFile, fileNodes);
          relationCount++;
        }
      }
      
      // Extract function definitions (simple heuristic)
      std::string functionName = extractFunctionDefinition(line);
      if (!functionName.empty()) {
        currentFunction = functionName;
        if (functionNodes.find(functionName) == functionNodes.end()) {
          auto funcId = graph.addNode(HyperNode::NodeType::FUNCTION, functionName, {
            {"name", functionName},
            {"source_file", filePath}
          });
          functionNodes[functionName] = funcId;
        }
      }
      
      // Extract function calls
      if (m_config.analyzeFunctionCalls && !currentFunction.empty()) {
        auto calls = extractFunctionCalls(line);
        for (const auto& calledFunc : calls) {
          addFunctionCall(graph, currentFunction, calledFunc, filePath, functionNodes);
          relationCount++;
        }
      }
    }
    
    return relationCount;
  }

  /// @brief Extract function definition from line (simple heuristic)
  std::string extractFunctionDefinition(const std::string& line) {
    // Simple regex for function definitions
    std::regex funcDefRegex(R"(^\s*(?:.*\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{?)");
    std::smatch match;
    
    if (std::regex_match(line, match, funcDefRegex)) {
      std::string funcName = match[1].str();
      if (!isKeyword(funcName) && !isOperator(funcName)) {
        return funcName;
      }
    }
    
    return "";
  }

  /// @brief Check if identifier is a C++ keyword
  bool isKeyword(const std::string& identifier) {
    static const std::unordered_set<std::string> keywords = {
      "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
      "return", "class", "struct", "namespace", "using", "typedef", "template",
      "typename", "public", "private", "protected", "virtual", "static", "const",
      "volatile", "mutable", "explicit", "inline", "extern", "auto", "register",
      "sizeof", "new", "delete", "throw", "try", "catch", "true", "false",
      "nullptr", "void", "bool", "char", "int", "float", "double", "long", "short"
    };
    
    return keywords.find(identifier) != keywords.end();
  }

  /// @brief Check if identifier is an operator
  bool isOperator(const std::string& identifier) {
    return identifier.find("operator") == 0;
  }
};

// </editor-fold> end ModuleMapper }}}1
//==============================================================================

} // namespace Experimental
} // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_MODULE_MAPPER
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_MODULE_MAPPER
#endif

#endif // KOKKOS_HYPERGRAPH_MODULE_MAPPER_HPP