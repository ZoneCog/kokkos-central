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

#ifndef KOKKOS_HYPERGRAPH_HPP
#define KOKKOS_HYPERGRAPH_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_HYPERGRAPH
#endif

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <impl/Kokkos_Error.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace Kokkos {
namespace Experimental {

//==============================================================================
// <editor-fold desc="HyperNode"> {{{1

/// @brief Represents a node in the AtomSpace-style hypergraph
/// Can represent files, functions, modules, or other code entities
class HyperNode {
 public:
  enum class NodeType {
    FILE,         ///< Source file node
    FUNCTION,     ///< Function node  
    MODULE,       ///< Module/namespace node
    CLASS,        ///< Class/struct node
    VARIABLE,     ///< Variable/data node
    CUSTOM        ///< User-defined node type
  };

  /// @brief Construct a HyperNode
  /// @param id Unique identifier for the node
  /// @param type Type of the node
  /// @param name Human-readable name
  /// @param metadata Additional metadata as key-value pairs
  HyperNode(std::size_t id, NodeType type, std::string name,
            std::unordered_map<std::string, std::string> metadata = {})
      : m_id(id), m_type(type), m_name(std::move(name)), m_metadata(std::move(metadata)) {}

  std::size_t getId() const { return m_id; }
  NodeType getType() const { return m_type; }
  const std::string& getName() const { return m_name; }
  const std::unordered_map<std::string, std::string>& getMetadata() const { return m_metadata; }
  
  void setMetadata(const std::string& key, const std::string& value) {
    m_metadata[key] = value;
  }
  
  std::string getMetadata(const std::string& key, const std::string& defaultValue = "") const {
    auto it = m_metadata.find(key);
    return (it != m_metadata.end()) ? it->second : defaultValue;
  }

 private:
  std::size_t m_id;
  NodeType m_type;
  std::string m_name;
  std::unordered_map<std::string, std::string> m_metadata;
};

// </editor-fold> end HyperNode }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="HyperLink"> {{{1

/// @brief Represents a hyperedge/link connecting multiple nodes
/// Can represent function calls, data flow, inheritance, etc.
class HyperLink {
 public:
  enum class LinkType {
    FUNCTION_CALL,    ///< Function call relationship
    DATA_FLOW,        ///< Data dependency relationship
    INHERITANCE,      ///< Class inheritance relationship
    COMPOSITION,      ///< Component relationship
    REFERENCE,        ///< Reference/pointer relationship
    INCLUDE,          ///< File include relationship
    CUSTOM            ///< User-defined link type
  };

  /// @brief Construct a HyperLink
  /// @param id Unique identifier for the link
  /// @param type Type of the link
  /// @param sourceNodes Source nodes (typically single node for most relationships)
  /// @param targetNodes Target nodes (can be multiple for hyperedges)
  /// @param metadata Additional metadata
  HyperLink(std::size_t id, LinkType type,
            std::vector<std::size_t> sourceNodes,
            std::vector<std::size_t> targetNodes,
            std::unordered_map<std::string, std::string> metadata = {})
      : m_id(id), m_type(type), m_sourceNodes(std::move(sourceNodes)),
        m_targetNodes(std::move(targetNodes)), m_metadata(std::move(metadata)) {}

  std::size_t getId() const { return m_id; }
  LinkType getType() const { return m_type; }
  const std::vector<std::size_t>& getSourceNodes() const { return m_sourceNodes; }
  const std::vector<std::size_t>& getTargetNodes() const { return m_targetNodes; }
  const std::unordered_map<std::string, std::string>& getMetadata() const { return m_metadata; }

  void setMetadata(const std::string& key, const std::string& value) {
    m_metadata[key] = value;
  }
  
  std::string getMetadata(const std::string& key, const std::string& defaultValue = "") const {
    auto it = m_metadata.find(key);
    return (it != m_metadata.end()) ? it->second : defaultValue;
  }

  /// @brief Check if this link connects the given nodes
  bool connects(std::size_t nodeA, std::size_t nodeB) const {
    auto hasNode = [](const std::vector<std::size_t>& nodes, std::size_t id) {
      return std::find(nodes.begin(), nodes.end(), id) != nodes.end();
    };
    return (hasNode(m_sourceNodes, nodeA) && hasNode(m_targetNodes, nodeB)) ||
           (hasNode(m_sourceNodes, nodeB) && hasNode(m_targetNodes, nodeA));
  }

 private:
  std::size_t m_id;
  LinkType m_type;
  std::vector<std::size_t> m_sourceNodes;
  std::vector<std::size_t> m_targetNodes;
  std::unordered_map<std::string, std::string> m_metadata;
};

// </editor-fold> end HyperLink }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="HyperGraph"> {{{1

/// @brief AtomSpace-style hypergraph for representing inter-module relations
/// Provides efficient storage and querying of complex relationships between
/// code entities (files, functions, modules) for dynamic agentic adaptation
class HyperGraph {
 public:
  using NodePtr = std::shared_ptr<HyperNode>;
  using LinkPtr = std::shared_ptr<HyperLink>;
  using NodeQueryCallback = std::function<bool(const HyperNode&)>;
  using LinkQueryCallback = std::function<bool(const HyperLink&)>;

  HyperGraph() : m_nextNodeId(0), m_nextLinkId(0) {}

  /// @brief Add a node to the hypergraph
  /// @param type Node type
  /// @param name Node name
  /// @param metadata Optional metadata
  /// @return Node ID
  std::size_t addNode(HyperNode::NodeType type, const std::string& name,
                      std::unordered_map<std::string, std::string> metadata = {}) {
    std::size_t id = m_nextNodeId++;
    auto node = std::make_shared<HyperNode>(id, type, name, std::move(metadata));
    m_nodes[id] = node;
    m_nodesByType[type].insert(id);
    return id;
  }

  /// @brief Add a link between nodes
  /// @param type Link type
  /// @param sourceNodes Source node IDs
  /// @param targetNodes Target node IDs  
  /// @param metadata Optional metadata
  /// @return Link ID
  std::size_t addLink(HyperLink::LinkType type,
                      std::vector<std::size_t> sourceNodes,
                      std::vector<std::size_t> targetNodes,
                      std::unordered_map<std::string, std::string> metadata = {}) {
    std::size_t id = m_nextLinkId++;
    auto link = std::make_shared<HyperLink>(id, type, sourceNodes, targetNodes, std::move(metadata));
    m_links[id] = link;
    
    // Update adjacency information for efficient queries
    for (auto sourceId : sourceNodes) {
      m_outgoingLinks[sourceId].insert(id);
      for (auto targetId : targetNodes) {
        m_adjacency[sourceId].insert(targetId);
      }
    }
    for (auto targetId : targetNodes) {
      m_incomingLinks[targetId].insert(id);
    }
    
    return id;
  }

  /// @brief Get node by ID
  NodePtr getNode(std::size_t nodeId) const {
    auto it = m_nodes.find(nodeId);
    return (it != m_nodes.end()) ? it->second : nullptr;
  }

  /// @brief Get link by ID
  LinkPtr getLink(std::size_t linkId) const {
    auto it = m_links.find(linkId);
    return (it != m_links.end()) ? it->second : nullptr;
  }

  /// @brief Find nodes by type
  std::vector<NodePtr> getNodesByType(HyperNode::NodeType type) const {
    std::vector<NodePtr> result;
    auto it = m_nodesByType.find(type);
    if (it != m_nodesByType.end()) {
      for (auto nodeId : it->second) {
        auto node = getNode(nodeId);
        if (node) result.push_back(node);
      }
    }
    return result;
  }

  /// @brief Find nodes connected to a given node
  std::vector<NodePtr> getConnectedNodes(std::size_t nodeId) const {
    std::vector<NodePtr> result;
    auto it = m_adjacency.find(nodeId);
    if (it != m_adjacency.end()) {
      for (auto connectedId : it->second) {
        auto node = getNode(connectedId);
        if (node) result.push_back(node);
      }
    }
    return result;
  }

  /// @brief Get outgoing links from a node
  std::vector<LinkPtr> getOutgoingLinks(std::size_t nodeId) const {
    std::vector<LinkPtr> result;
    auto it = m_outgoingLinks.find(nodeId);
    if (it != m_outgoingLinks.end()) {
      for (auto linkId : it->second) {
        auto link = getLink(linkId);
        if (link) result.push_back(link);
      }
    }
    return result;
  }

  /// @brief Get incoming links to a node
  std::vector<LinkPtr> getIncomingLinks(std::size_t nodeId) const {
    std::vector<LinkPtr> result;
    auto it = m_incomingLinks.find(nodeId);
    if (it != m_incomingLinks.end()) {
      for (auto linkId : it->second) {
        auto link = getLink(linkId);
        if (link) result.push_back(link);
      }
    }
    return result;
  }

  /// @brief Query nodes with custom predicate
  std::vector<NodePtr> queryNodes(NodeQueryCallback predicate) const {
    std::vector<NodePtr> result;
    for (const auto& pair : m_nodes) {
      if (predicate(*pair.second)) {
        result.push_back(pair.second);
      }
    }
    return result;
  }

  /// @brief Query links with custom predicate
  std::vector<LinkPtr> queryLinks(LinkQueryCallback predicate) const {
    std::vector<LinkPtr> result;
    for (const auto& pair : m_links) {
      if (predicate(*pair.second)) {
        result.push_back(pair.second);
      }
    }
    return result;
  }

  /// @brief Get graph statistics for optimization
  struct GraphStats {
    std::size_t nodeCount;
    std::size_t linkCount;
    std::size_t maxDegree;
    double avgDegree;
    std::unordered_map<HyperNode::NodeType, std::size_t> nodeTypeCounts;
    std::unordered_map<HyperLink::LinkType, std::size_t> linkTypeCounts;
  };

  GraphStats getStats() const {
    GraphStats stats;
    stats.nodeCount = m_nodes.size();
    stats.linkCount = m_links.size();
    stats.maxDegree = 0;
    
    std::size_t totalDegree = 0;
    for (const auto& pair : m_adjacency) {
      std::size_t degree = pair.second.size();
      stats.maxDegree = std::max(stats.maxDegree, degree);
      totalDegree += degree;
    }
    
    stats.avgDegree = stats.nodeCount > 0 ? static_cast<double>(totalDegree) / stats.nodeCount : 0.0;
    
    // Count by types
    for (const auto& pair : m_nodes) {
      stats.nodeTypeCounts[pair.second->getType()]++;
    }
    for (const auto& pair : m_links) {
      stats.linkTypeCounts[pair.second->getType()]++;
    }
    
    return stats;
  }

  /// @brief Clear all nodes and links
  void clear() {
    m_nodes.clear();
    m_links.clear();
    m_nodesByType.clear();
    m_adjacency.clear();
    m_outgoingLinks.clear();
    m_incomingLinks.clear();
    m_nextNodeId = 0;
    m_nextLinkId = 0;
  }

  /// @brief Get total number of nodes
  std::size_t getNodeCount() const { return m_nodes.size(); }

  /// @brief Get total number of links
  std::size_t getLinkCount() const { return m_links.size(); }

 private:
  // Core storage
  std::unordered_map<std::size_t, NodePtr> m_nodes;
  std::unordered_map<std::size_t, LinkPtr> m_links;
  
  // Indexing for efficient queries
  std::unordered_map<HyperNode::NodeType, std::unordered_set<std::size_t>> m_nodesByType;
  std::unordered_map<std::size_t, std::unordered_set<std::size_t>> m_adjacency;
  std::unordered_map<std::size_t, std::unordered_set<std::size_t>> m_outgoingLinks;
  std::unordered_map<std::size_t, std::unordered_set<std::size_t>> m_incomingLinks;
  
  // ID generation
  std::size_t m_nextNodeId;
  std::size_t m_nextLinkId;
};

// </editor-fold> end HyperGraph }}}1
//==============================================================================

} // namespace Experimental
} // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_HYPERGRAPH
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_HYPERGRAPH
#endif

#endif // KOKKOS_HYPERGRAPH_HPP