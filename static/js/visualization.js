/**
 * GraphRAG - Knowledge Graph Visualization
 * This file handles the visualization of knowledge graphs using D3.js
 */

// Define color scheme for entity types
const nodeColors = {
  PERSON: '#4e79a7',
  ORG: '#f28e2c',
  ORGANIZATION: '#f28e2c',
  GPE: '#e15759',
  LOCATION: '#e15759',
  DATE: '#76b7b2',
  TIME: '#76b7b2',
  PRODUCT: '#59a14f',
  EVENT: '#edc949',
  WORK_OF_ART: '#af7aa1',
  LAW: '#ff9da7',
  MONEY: '#9c755f',
  QUANTITY: '#bab0ab',
  DEFAULT: '#1f77b4'
};

// Global visualization variables
let svg, simulation, linkElements, nodeElements, textElements;
let tooltip;
let width, height;
let nodes = [];
let links = [];
let selectedNode = null;

/**
 * Initialize the knowledge graph visualization
 */
function initializeGraph() {
  const container = document.getElementById('graph-container');
  if (!container) return;
  
  // Get container dimensions
  const containerRect = container.getBoundingClientRect();
  width = containerRect.width;
  height = containerRect.height;
  
  // Create SVG element
  svg = d3.select('#graph-container')
    .append('svg')
    .attr('width', width)
    .attr('height', height);
    
  // Define arrow markers for links
  svg.append('defs').append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '-0 -5 10 10')
    .attr('refX', 25)
    .attr('refY', 0)
    .attr('orient', 'auto')
    .attr('markerWidth', 8)
    .attr('markerHeight', 8)
    .attr('xoverflow', 'visible')
    .append('svg:path')
    .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
    .attr('fill', '#999')
    .style('stroke', 'none');
    
  // Create groups for links, nodes, and text
  const linkGroup = svg.append('g').attr('class', 'links');
  const nodeGroup = svg.append('g').attr('class', 'nodes');
  const textGroup = svg.append('g').attr('class', 'texts');
  
  // Create tooltip
  tooltip = d3.select('body').append('div')
    .attr('class', 'entity-tooltip')
    .style('opacity', 0);
    
  // Create force simulation
  simulation = d3.forceSimulation()
    .force('link', d3.forceLink().id(d => d.id).distance(120))
    .force('charge', d3.forceManyBody().strength(-800))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(50));
    
  // Handle window resize
  window.addEventListener('resize', () => {
    const containerRect = container.getBoundingClientRect();
    width = containerRect.width;
    height = containerRect.height;
    
    svg.attr('width', width).attr('height', height);
    simulation.force('center', d3.forceCenter(width / 2, height / 2));
    simulation.restart();
  });
  
  // Handle zoom and pan
  const zoom = d3.zoom()
    .scaleExtent([0.2, 8])
    .on('zoom', event => {
      linkGroup.attr('transform', event.transform);
      nodeGroup.attr('transform', event.transform);
      textGroup.attr('transform', event.transform);
    });
    
  svg.call(zoom);
  
  // Load initial data if available
  const initialEntities = document.getElementById('initial-entities');
  if (initialEntities) {
    const entitiesData = JSON.parse(initialEntities.getAttribute('data-entities') || '[]');
    if (entitiesData.length > 0) {
      loadEntityGraph(entitiesData[0].id);
    }
  }
  
  // Initialize controls
  initializeGraphControls();
}

/**
 * Update the graph with new data
 * @param {Array} graphData - The nodes and edges of the graph
 */
function updateGraph(graphData) {
  if (!graphData || !graphData.nodes || !graphData.edges) return;
  
  nodes = graphData.nodes.map(node => ({
    ...node,
    radius: getNodeRadius(node)
  }));
  
  links = graphData.edges.map(edge => ({
    ...edge,
    value: edge.confidence || 0.5
  }));
  
  // Update force simulation
  simulation.nodes(nodes);
  simulation.force('link').links(links);
  
  // Create links
  linkElements = svg.select('.links')
    .selectAll('line')
    .data(links)
    .join(
      enter => enter.append('line')
        .attr('stroke-width', d => Math.max(1, d.value * 3))
        .attr('stroke', '#999')
        .attr('stroke-opacity', 0.6)
        .attr('marker-end', 'url(#arrowhead)')
        .append('title')
        .text(d => d.label),
      update => update
        .attr('stroke-width', d => Math.max(1, d.value * 3)),
      exit => exit.remove()
    );
    
  // Create node elements
  nodeElements = svg.select('.nodes')
    .selectAll('circle')
    .data(nodes)
    .join(
      enter => enter.append('circle')
        .attr('r', d => d.radius)
        .attr('fill', d => getNodeColor(d))
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5)
        .attr('class', 'entity-node')
        .call(d3.drag()
          .on('start', dragStarted)
          .on('drag', dragging)
          .on('end', dragEnded))
        .on('mouseover', showTooltip)
        .on('mouseout', hideTooltip)
        .on('click', nodeClicked),
      update => update
        .attr('r', d => d.radius)
        .attr('fill', d => getNodeColor(d)),
      exit => exit.remove()
    );
    
  // Create text elements
  textElements = svg.select('.texts')
    .selectAll('text')
    .data(nodes)
    .join(
      enter => enter.append('text')
        .text(d => d.label)
        .attr('font-size', 10)
        .attr('dx', d => d.radius + 5)
        .attr('dy', 4)
        .attr('fill', '#aaa'),
      update => update
        .text(d => d.label),
      exit => exit.remove()
    );
    
  // Restart simulation
  simulation.alpha(0.3).restart();
  
  // Update positions on tick
  simulation.on('tick', () => {
    // Constrain nodes to the viewport
    nodes.forEach(node => {
      node.x = Math.max(node.radius, Math.min(width - node.radius, node.x));
      node.y = Math.max(node.radius, Math.min(height - node.radius, node.y));
    });
    
    linkElements
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);
      
    nodeElements
      .attr('cx', d => d.x)
      .attr('cy', d => d.y);
      
    textElements
      .attr('x', d => d.x)
      .attr('y', d => d.y);
  });
}

/**
 * Get the radius of a node based on its type
 */
function getNodeRadius(node) {
  // Base radius
  let radius = 8;
  
  // Adjust based on entity type
  if (node.entity_type) {
    switch (node.entity_type) {
      case 'PERSON':
      case 'ORG':
      case 'ORGANIZATION':
        radius = 10;
        break;
      case 'EVENT':
      case 'WORK_OF_ART':
        radius = 12;
        break;
      default:
        radius = 8;
    }
  }
  
  return radius;
}

/**
 * Get color for a node based on its type
 */
function getNodeColor(node) {
  if (!node.entity_type) return nodeColors.DEFAULT;
  return nodeColors[node.entity_type] || nodeColors.DEFAULT;
}

/**
 * Handle drag start event
 */
function dragStarted(event, d) {
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

/**
 * Handle dragging event
 */
function dragging(event, d) {
  d.fx = event.x;
  d.fy = event.y;
}

/**
 * Handle drag end event
 */
function dragEnded(event, d) {
  if (!event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}

/**
 * Show tooltip on node hover
 */
function showTooltip(event, d) {
  tooltip.transition()
    .duration(200)
    .style('opacity', 0.9);
    
  let tooltipContent = `
    <div>
      <strong>${d.label}</strong><br>
      <span class="badge bg-info">${d.entity_type || 'Entity'}</span>
    </div>
  `;
  
  tooltip.html(tooltipContent)
    .style('left', (event.pageX + 10) + 'px')
    .style('top', (event.pageY - 28) + 'px');
}

/**
 * Hide tooltip when not hovering
 */
function hideTooltip() {
  tooltip.transition()
    .duration(500)
    .style('opacity', 0);
}

/**
 * Handle node click event
 */
function nodeClicked(event, d) {
  event.preventDefault();
  
  // Toggle selected state
  if (selectedNode === d.id) {
    selectedNode = null;
    nodeElements.attr('stroke', '#fff');
  } else {
    selectedNode = d.id;
    nodeElements.attr('stroke', n => n.id === d.id ? '#ff0' : '#fff');
    
    // Extract entity ID from node ID (format: "entity_123")
    const entityId = d.id.split('_')[1];
    
    // Load entity neighborhood
    if (entityId) {
      loadEntityGraph(entityId);
    }
  }
}

/**
 * Load entity graph from API
 */
function loadEntityGraph(entityId, maxHops = 2) {
  // Show loading state
  d3.select('#graph-loading-indicator').style('display', 'block');
  
  // Update UI to show which entity is being explored
  d3.select('#current-entity-display').text(`Exploring entity ID: ${entityId}`);
  
  fetch(`/api/knowledge-graph/traverse`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      entity_id: entityId,
      max_hops: maxHops
    })
  })
  .then(response => response.json())
  .then(data => {
    // Hide loading indicator
    d3.select('#graph-loading-indicator').style('display', 'none');
    
    // Update graph
    updateGraph(data);
    
    // Update entity info panel
    updateEntityInfoPanel(entityId);
  })
  .catch(error => {
    console.error('Error loading entity graph:', error);
    d3.select('#graph-loading-indicator').style('display', 'none');
    
    // Show error message
    d3.select('#graph-error-message')
      .style('display', 'block')
      .text('Error loading graph data. Please try again.');
      
    setTimeout(() => {
      d3.select('#graph-error-message').style('display', 'none');
    }, 3000);
  });
}

/**
 * Update entity info panel with connections
 */
function updateEntityInfoPanel(entityId) {
  const infoPanel = d3.select('#entity-info-panel');
  
  // Show loading state
  infoPanel.html('<div class="loader-container"><div class="loader"></div></div>');
  
  fetch(`/api/knowledge-graph/entities/${entityId}`)
    .then(response => response.json())
    .then(data => {
      const entityConnections = data.connections || [];
      
      if (entityConnections.length === 0) {
        infoPanel.html('<p class="text-center">No connections found for this entity.</p>');
        return;
      }
      
      // Group by relationship type
      const relationshipGroups = {};
      entityConnections.forEach(conn => {
        if (!relationshipGroups[conn.relationship]) {
          relationshipGroups[conn.relationship] = [];
        }
        relationshipGroups[conn.relationship].push(conn);
      });
      
      // Build the panel content
      let panelContent = '<div class="list-group">';
      
      for (const [relationship, connections] of Object.entries(relationshipGroups)) {
        panelContent += `<h6 class="mt-3">${formatRelationshipName(relationship)} (${connections.length})</h6>`;
        
        connections.forEach(conn => {
          const confidenceClass = getConfidenceClass(conn.confidence);
          panelContent += `
            <a href="#" class="list-group-item list-group-item-action bg-dark connection-item" 
               data-entity-id="${conn.entity_id}" onclick="exploreEntity(${conn.entity_id}); return false;">
              <div class="d-flex justify-content-between align-items-center">
                <div>
                  <span class="badge bg-info me-2">${conn.entity_type}</span>
                  <strong>${conn.name}</strong>
                </div>
                <span class="badge ${confidenceClass}">${Math.round(conn.confidence * 100)}%</span>
              </div>
            </a>
          `;
        });
      }
      
      panelContent += '</div>';
      infoPanel.html(panelContent);
    })
    .catch(error => {
      console.error('Error fetching entity data:', error);
      infoPanel.html(`
        <div class="alert alert-danger">
          Error loading entity information.
        </div>
      `);
    });
}

/**
 * Explore a new entity from the info panel
 */
function exploreEntity(entityId) {
  loadEntityGraph(entityId);
}

/**
 * Initialize graph controls
 */
function initializeGraphControls() {
  // Initialize max hops slider
  const maxHopsSlider = document.getElementById('max-hops-slider');
  const maxHopsValue = document.getElementById('max-hops-value');
  
  if (maxHopsSlider && maxHopsValue) {
    maxHopsSlider.addEventListener('input', () => {
      const value = maxHopsSlider.value;
      maxHopsValue.textContent = value;
      
      // If an entity is selected, reload with new max hops
      if (selectedNode) {
        const entityId = selectedNode.split('_')[1];
        loadEntityGraph(entityId, parseInt(value));
      }
    });
  }
  
  // Initialize entity search
  const entitySearchInput = document.getElementById('entity-search');
  const entitySearchResults = document.getElementById('entity-search-results');
  
  if (entitySearchInput && entitySearchResults) {
    entitySearchInput.addEventListener('input', () => {
      const searchTerm = entitySearchInput.value.trim();
      
      if (searchTerm.length < 3) {
        entitySearchResults.innerHTML = '';
        return;
      }
      
      // Search for entities
      fetch(`/api/entities/search?q=${encodeURIComponent(searchTerm)}`)
        .then(response => response.json())
        .then(data => {
          if (!data.entities || data.entities.length === 0) {
            entitySearchResults.innerHTML = '<p class="text-secondary">No entities found</p>';
            return;
          }
          
          // Build search results
          let resultsHtml = '<div class="list-group">';
          data.entities.slice(0, 5).forEach(entity => {
            resultsHtml += `
              <a href="#" class="list-group-item list-group-item-action bg-dark"
                 onclick="exploreEntity(${entity.id}); return false;">
                <div class="d-flex justify-content-between align-items-center">
                  <div>
                    <span class="badge bg-info me-2">${entity.entity_type}</span>
                    <strong>${entity.name}</strong>
                  </div>
                  <span class="badge bg-secondary">${entity.document_title}</span>
                </div>
              </a>
            `;
          });
          resultsHtml += '</div>';
          
          entitySearchResults.innerHTML = resultsHtml;
        })
        .catch(error => {
          console.error('Error searching entities:', error);
          entitySearchResults.innerHTML = '<p class="text-danger">Error searching entities</p>';
        });
    });
  }
  
  // Initialize entity type filter
  const entityTypeFilter = document.getElementById('entity-type-filter');
  
  if (entityTypeFilter) {
    entityTypeFilter.addEventListener('change', () => {
      const selectedType = entityTypeFilter.value;
      
      // Apply filter to visualization
      if (selectedType === 'all') {
        nodeElements.style('opacity', 1);
        linkElements.style('opacity', 0.6);
        textElements.style('opacity', 1);
      } else {
        // Show only nodes of selected type and their connections
        const selectedNodes = new Set();
        
        // First identify nodes of the selected type
        nodes.forEach(node => {
          if (node.entity_type === selectedType) {
            selectedNodes.add(node.id);
          }
        });
        
        // Then add their direct connections
        links.forEach(link => {
          if (selectedNodes.has(link.source.id)) {
            selectedNodes.add(link.target.id);
          }
          if (selectedNodes.has(link.target.id)) {
            selectedNodes.add(link.source.id);
          }
        });
        
        // Apply filter
        nodeElements.style('opacity', d => selectedNodes.has(d.id) ? 1 : 0.2);
        linkElements.style('opacity', d => 
          selectedNodes.has(d.source.id) && selectedNodes.has(d.target.id) ? 0.6 : 0.1);
        textElements.style('opacity', d => selectedNodes.has(d.id) ? 1 : 0.2);
      }
    });
  }
  
  // Initialize relationship filter
  const relationshipFilter = document.getElementById('relationship-filter');
  
  if (relationshipFilter) {
    relationshipFilter.addEventListener('change', () => {
      const selectedRelationship = relationshipFilter.value;
      
      // Apply filter to links
      if (selectedRelationship === 'all') {
        linkElements.style('opacity', 0.6);
        nodeElements.style('opacity', 1);
        textElements.style('opacity', 1);
      } else {
        // Show only links of selected type and their nodes
        const connectedNodes = new Set();
        
        links.forEach(link => {
          if (link.label === selectedRelationship) {
            connectedNodes.add(link.source.id);
            connectedNodes.add(link.target.id);
          }
        });
        
        // Apply filter
        linkElements.style('opacity', d => d.label === selectedRelationship ? 0.8 : 0.1);
        nodeElements.style('opacity', d => connectedNodes.has(d.id) ? 1 : 0.2);
        textElements.style('opacity', d => connectedNodes.has(d.id) ? 1 : 0.2);
      }
    });
  }
  
  // Initialize apply layout button
  const applyLayoutButton = document.getElementById('apply-layout-button');
  
  if (applyLayoutButton) {
    applyLayoutButton.addEventListener('click', () => {
      // Reset simulation with stronger forces temporarily
      simulation
        .alpha(0.8)
        .force('charge', d3.forceManyBody().strength(-800))
        .restart();
    });
  }
  
  // Initialize reset view button
  const resetViewButton = document.getElementById('reset-view-button');
  
  if (resetViewButton) {
    resetViewButton.addEventListener('click', () => {
      // Reset zoom and pan
      svg.transition().duration(750).call(
        d3.zoom().transform,
        d3.zoomIdentity.translate(0, 0).scale(1)
      );
    });
  }
}

// Helper functions
function formatRelationshipName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase());
}

function getConfidenceClass(confidence) {
  if (confidence >= 0.8) {
    return 'bg-success';
  } else if (confidence >= 0.6) {
    return 'bg-info';
  } else if (confidence >= 0.4) {
    return 'bg-warning';
  } else {
    return 'bg-secondary';
  }
}

// Initialize the graph when the page is loaded
document.addEventListener('DOMContentLoaded', function() {
  const graphContainer = document.getElementById('graph-container');
  if (graphContainer) {
    initializeGraph();
  }
});
