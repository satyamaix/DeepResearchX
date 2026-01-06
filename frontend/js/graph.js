/**
 * DRX Argument Graph Visualization Module
 * Interactive knowledge graph with Cytoscape.js
 *
 * Provides visualization of entities, relations, and claims
 * with citation hovercards and filtering capabilities.
 */

// Cytoscape.js loaded via CDN in index.html

const DRXGraph = (function() {
    'use strict';

    // ==========================================================================
    // Private State
    // ==========================================================================

    let cy = null;
    let container = null;
    let tooltipEl = null;

    // ==========================================================================
    // Color Configurations
    // ==========================================================================

    /**
     * Entity type colors matching backend EntityType definitions
     */
    const ENTITY_COLORS = {
        person: '#4A90D9',
        organization: '#50C878',
        concept: '#9B59B6',
        event: '#E67E22',
        location: '#E74C3C',
        document: '#6366F1',
        claim: '#EC4899'
    };

    /**
     * Claim status colors for verification state
     */
    const CLAIM_COLORS = {
        supported: '#27AE60',
        contested: '#F39C12',
        refuted: '#E74C3C',
        unverified: '#8B949E'
    };

    // ==========================================================================
    // Cytoscape Style Configuration
    // ==========================================================================

    /**
     * Get Cytoscape style array
     * @returns {Array} Cytoscape style configuration
     */
    function getCytoscapeStyle() {
        return [
            // Base node styles
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '11px',
                    'font-family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                    'text-wrap': 'ellipsis',
                    'text-max-width': '80px',
                    'color': '#ffffff',
                    'text-outline-width': 1,
                    'text-outline-color': 'data(bgColor)',
                    'width': 'mapData(weight, 0, 100, 40, 80)',
                    'height': 'mapData(weight, 0, 100, 40, 80)',
                    'background-color': 'data(bgColor)',
                    'border-width': 2,
                    'border-color': 'data(borderColor)',
                    'transition-property': 'width, height, border-width',
                    'transition-duration': '0.2s'
                }
            },

            // Entity type specific styles
            {
                selector: 'node[type="person"]',
                style: {
                    'background-color': ENTITY_COLORS.person,
                    'border-color': '#3A7AC5'
                }
            },
            {
                selector: 'node[type="organization"]',
                style: {
                    'background-color': ENTITY_COLORS.organization,
                    'border-color': '#3DAD60'
                }
            },
            {
                selector: 'node[type="concept"]',
                style: {
                    'background-color': ENTITY_COLORS.concept,
                    'border-color': '#7B4798'
                }
            },
            {
                selector: 'node[type="event"]',
                style: {
                    'background-color': ENTITY_COLORS.event,
                    'border-color': '#C76A1A'
                }
            },
            {
                selector: 'node[type="location"]',
                style: {
                    'background-color': ENTITY_COLORS.location,
                    'border-color': '#C73A3A'
                }
            },
            {
                selector: 'node[type="document"]',
                style: {
                    'background-color': ENTITY_COLORS.document,
                    'border-color': '#4F51D9'
                }
            },

            // Claim nodes with diamond shape
            {
                selector: 'node[nodeType="claim"]',
                style: {
                    'shape': 'diamond',
                    'background-color': ENTITY_COLORS.claim,
                    'border-color': '#C73980'
                }
            },
            {
                selector: 'node[nodeType="claim"][status="supported"]',
                style: {
                    'background-color': CLAIM_COLORS.supported,
                    'border-color': '#1E8449'
                }
            },
            {
                selector: 'node[nodeType="claim"][status="contested"]',
                style: {
                    'background-color': CLAIM_COLORS.contested,
                    'border-color': '#D68910'
                }
            },
            {
                selector: 'node[nodeType="claim"][status="refuted"]',
                style: {
                    'background-color': CLAIM_COLORS.refuted,
                    'border-color': '#C0392B'
                }
            },
            {
                selector: 'node[nodeType="claim"][status="unverified"]',
                style: {
                    'background-color': CLAIM_COLORS.unverified,
                    'border-color': '#6E7681'
                }
            },

            // Edge styles
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': '#6E7681',
                    'target-arrow-color': '#6E7681',
                    'target-arrow-shape': 'triangle',
                    'arrow-scale': 0.8,
                    'curve-style': 'bezier',
                    'label': 'data(label)',
                    'font-size': '9px',
                    'font-family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                    'color': '#8B949E',
                    'text-rotation': 'autorotate',
                    'text-margin-y': -8,
                    'text-background-color': 'var(--bg-secondary, #161b22)',
                    'text-background-opacity': 0.8,
                    'text-background-padding': '2px',
                    'transition-property': 'line-color, target-arrow-color, width',
                    'transition-duration': '0.2s'
                }
            },

            // Edge confidence styling
            {
                selector: 'edge[confidence >= 0.8]',
                style: {
                    'width': 3,
                    'line-style': 'solid'
                }
            },
            {
                selector: 'edge[confidence >= 0.5][confidence < 0.8]',
                style: {
                    'width': 2,
                    'line-style': 'solid'
                }
            },
            {
                selector: 'edge[confidence < 0.5]',
                style: {
                    'width': 1.5,
                    'line-style': 'dashed'
                }
            },

            // Hover states
            {
                selector: 'node:active',
                style: {
                    'overlay-opacity': 0.2,
                    'overlay-color': '#58a6ff'
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'border-width': 4,
                    'border-color': '#58a6ff'
                }
            },

            // Highlighted path
            {
                selector: '.highlighted',
                style: {
                    'line-color': '#58a6ff',
                    'target-arrow-color': '#58a6ff',
                    'width': 4,
                    'z-index': 10
                }
            },
            {
                selector: 'node.highlighted',
                style: {
                    'border-width': 4,
                    'border-color': '#58a6ff',
                    'z-index': 10
                }
            },

            // Hidden nodes (filtered out)
            {
                selector: '.hidden',
                style: {
                    'display': 'none'
                }
            },

            // Dimmed nodes (not in current filter focus)
            {
                selector: '.dimmed',
                style: {
                    'opacity': 0.3
                }
            }
        ];
    }

    // ==========================================================================
    // Initialization
    // ==========================================================================

    /**
     * Initialize the graph visualization
     * @param {string} containerId - DOM element ID for the graph
     * @param {Object} options - Optional configuration
     */
    function init(containerId, options = {}) {
        container = document.getElementById(containerId);
        if (!container) {
            console.error('[DRXGraph] Container not found:', containerId);
            return;
        }

        // Check if Cytoscape is loaded
        if (typeof cytoscape === 'undefined') {
            console.error('[DRXGraph] Cytoscape.js not loaded. Include it via CDN.');
            return;
        }

        // Destroy existing instance if any
        if (cy) {
            cy.destroy();
        }

        // Create Cytoscape instance
        cy = cytoscape({
            container: container,
            style: getCytoscapeStyle(),
            layout: {
                name: 'cose',
                animate: true,
                animationDuration: 500,
                nodeDimensionsIncludeLabels: true,
                randomize: false,
                componentSpacing: 100,
                nodeRepulsion: function() { return 8000; },
                idealEdgeLength: function() { return 100; },
                edgeElasticity: function() { return 100; },
                nestingFactor: 5,
                gravity: 80,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0
            },
            // Interaction settings
            userZoomingEnabled: true,
            userPanningEnabled: true,
            boxSelectionEnabled: true,
            selectionType: 'single',
            minZoom: 0.2,
            maxZoom: 3,
            wheelSensitivity: 0.3,
            ...options
        });

        // Create tooltip element
        createTooltip();

        // Set up event handlers
        setupEventHandlers();

        console.log('[DRXGraph] Initialized');
    }

    /**
     * Create tooltip DOM element
     */
    function createTooltip() {
        // Remove existing tooltip if any
        const existing = container.querySelector('.graph-tooltip');
        if (existing) {
            existing.remove();
        }

        tooltipEl = document.createElement('div');
        tooltipEl.className = 'graph-tooltip';
        container.appendChild(tooltipEl);
    }

    // ==========================================================================
    // Event Handlers
    // ==========================================================================

    /**
     * Set up Cytoscape event handlers
     */
    function setupEventHandlers() {
        if (!cy) return;

        // Node hover - show tooltip
        cy.on('mouseover', 'node', function(event) {
            const node = event.target;
            showTooltip(node, event);
        });

        cy.on('mouseout', 'node', function() {
            hideTooltip();
        });

        // Edge hover - show tooltip
        cy.on('mouseover', 'edge', function(event) {
            const edge = event.target;
            showEdgeTooltip(edge, event);
        });

        cy.on('mouseout', 'edge', function() {
            hideTooltip();
        });

        // Node click - show detail panel
        cy.on('tap', 'node', function(event) {
            const node = event.target;
            showNodeDetail(node);

            // Highlight connected nodes
            highlightConnected(node);
        });

        // Edge click - show edge detail
        cy.on('tap', 'edge', function(event) {
            const edge = event.target;
            showEdgeDetail(edge);
        });

        // Background click - clear selection
        cy.on('tap', function(event) {
            if (event.target === cy) {
                clearHighlights();
                hideDetailPanel();
            }
        });

        // Track mouse position for tooltip positioning
        cy.on('mousemove', function(event) {
            if (tooltipEl && tooltipEl.classList.contains('visible')) {
                updateTooltipPosition(event);
            }
        });
    }

    // ==========================================================================
    // Tooltip Functions
    // ==========================================================================

    /**
     * Show tooltip for a node
     * @param {Object} node - Cytoscape node
     * @param {Object} event - Mouse event
     */
    function showTooltip(node, event) {
        const data = node.data();

        let html = `<strong>${escapeHtml(data.label || 'Unknown')}</strong>`;

        if (data.type) {
            html += `<div class="tooltip-type">${escapeHtml(data.type)}</div>`;
        }

        if (data.nodeType === 'claim' && data.status) {
            html += `<div class="tooltip-status ${data.status}">${escapeHtml(data.status)}</div>`;
        }

        if (data.confidence !== undefined) {
            const pct = (data.confidence * 100).toFixed(0);
            html += `<div class="tooltip-confidence">Confidence: ${pct}%</div>`;
        }

        if (data.sources && data.sources.length > 0) {
            html += `<div class="tooltip-sources">${data.sources.length} source(s)</div>`;
        }

        tooltipEl.innerHTML = html;
        tooltipEl.classList.add('visible');
        updateTooltipPosition(event);
    }

    /**
     * Show tooltip for an edge
     * @param {Object} edge - Cytoscape edge
     * @param {Object} event - Mouse event
     */
    function showEdgeTooltip(edge, event) {
        const data = edge.data();
        const sourceNode = cy.getElementById(data.source);
        const targetNode = cy.getElementById(data.target);

        let html = `<div class="tooltip-relation">`;
        html += `<strong>${escapeHtml(sourceNode.data('label') || data.source)}</strong>`;
        html += ` <span class="relation-arrow">&rarr;</span> `;
        html += `<span class="relation-type">${escapeHtml(data.label || 'relates to')}</span>`;
        html += ` <span class="relation-arrow">&rarr;</span> `;
        html += `<strong>${escapeHtml(targetNode.data('label') || data.target)}</strong>`;
        html += `</div>`;

        if (data.confidence !== undefined) {
            const pct = (data.confidence * 100).toFixed(0);
            html += `<div class="tooltip-confidence">Confidence: ${pct}%</div>`;
        }

        tooltipEl.innerHTML = html;
        tooltipEl.classList.add('visible');
        updateTooltipPosition(event);
    }

    /**
     * Update tooltip position
     * @param {Object} event - Mouse event
     */
    function updateTooltipPosition(event) {
        if (!tooltipEl || !container) return;

        const containerRect = container.getBoundingClientRect();
        const x = event.renderedPosition ? event.renderedPosition.x : (event.originalEvent.clientX - containerRect.left);
        const y = event.renderedPosition ? event.renderedPosition.y : (event.originalEvent.clientY - containerRect.top);

        // Position tooltip above and to the right of cursor
        let left = x + 15;
        let top = y - tooltipEl.offsetHeight - 10;

        // Keep tooltip within container bounds
        if (left + tooltipEl.offsetWidth > containerRect.width) {
            left = x - tooltipEl.offsetWidth - 15;
        }
        if (top < 0) {
            top = y + 15;
        }

        tooltipEl.style.left = `${left}px`;
        tooltipEl.style.top = `${top}px`;
    }

    /**
     * Hide tooltip
     */
    function hideTooltip() {
        if (tooltipEl) {
            tooltipEl.classList.remove('visible');
        }
    }

    // ==========================================================================
    // Detail Panel Functions
    // ==========================================================================

    /**
     * Show node detail panel with citations
     * @param {Object} node - Cytoscape node
     */
    function showNodeDetail(node) {
        const panel = document.getElementById('graph-detail-panel');
        if (!panel) return;

        const data = node.data();

        let html = `<button class="detail-close" onclick="DRXGraph.hideDetailPanel()" aria-label="Close">&times;</button>`;
        html += `<h4>${escapeHtml(data.label || 'Unknown')}</h4>`;

        // Type badge
        const displayType = data.nodeType === 'claim' ? 'claim' : (data.type || 'entity');
        html += `<span class="node-type type-${displayType}">${escapeHtml(displayType)}</span>`;

        // Status for claims
        if (data.nodeType === 'claim' && data.status) {
            html += `<span class="claim-status status-${data.status}">${escapeHtml(data.status)}</span>`;
        }

        // Confidence
        if (data.confidence !== undefined) {
            const pct = (data.confidence * 100).toFixed(0);
            html += `<div class="detail-confidence">
                <span class="confidence-label">Confidence:</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${pct}%"></div>
                </div>
                <span class="confidence-value">${pct}%</span>
            </div>`;
        }

        // Evidence/Description
        if (data.evidence) {
            html += `<div class="detail-evidence">
                <strong>Evidence:</strong>
                <p>${escapeHtml(data.evidence)}</p>
            </div>`;
        }

        if (data.statement) {
            html += `<div class="detail-statement">
                <strong>Statement:</strong>
                <p>${escapeHtml(data.statement)}</p>
            </div>`;
        }

        // Properties
        if (data.properties && Object.keys(data.properties).length > 0) {
            html += `<div class="detail-properties"><strong>Properties:</strong><ul>`;
            for (const [key, value] of Object.entries(data.properties)) {
                if (!['id', 'label', 'type', 'bgColor', 'borderColor', 'weight'].includes(key)) {
                    html += `<li><span class="prop-key">${escapeHtml(key)}:</span> ${escapeHtml(String(value))}</li>`;
                }
            }
            html += `</ul></div>`;
        }

        // Sources with citation hovercards
        if (data.sources && data.sources.length > 0) {
            html += `<div class="detail-sources">
                <strong>Sources (${data.sources.length}):</strong>
                <ul class="source-list">`;

            data.sources.forEach((src, idx) => {
                const title = src.title || src.url || `Source ${idx + 1}`;
                const url = src.url || '#';
                html += `<li class="source-item">
                    <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer"
                       class="source-link" title="${escapeHtml(src.title || url)}">
                        ${escapeHtml(title)}
                    </a>
                    ${src.domain ? `<span class="source-domain">${escapeHtml(src.domain)}</span>` : ''}
                </li>`;
            });

            html += `</ul></div>`;
        }

        // Connected nodes
        const connectedEdges = node.connectedEdges();
        if (connectedEdges.length > 0) {
            html += `<div class="detail-connections">
                <strong>Connections (${connectedEdges.length}):</strong>
                <ul class="connection-list">`;

            connectedEdges.forEach(edge => {
                const edgeData = edge.data();
                const otherNodeId = edgeData.source === data.id ? edgeData.target : edgeData.source;
                const otherNode = cy.getElementById(otherNodeId);
                const direction = edgeData.source === data.id ? '&rarr;' : '&larr;';

                html += `<li class="connection-item" onclick="DRXGraph.focusNode('${escapeHtml(otherNodeId)}')">
                    <span class="connection-direction">${direction}</span>
                    <span class="connection-relation">${escapeHtml(edgeData.label || 'relates to')}</span>
                    <span class="connection-target">${escapeHtml(otherNode.data('label') || otherNodeId)}</span>
                </li>`;
            });

            html += `</ul></div>`;
        }

        panel.innerHTML = html;
        panel.classList.add('visible');
    }

    /**
     * Show edge detail panel
     * @param {Object} edge - Cytoscape edge
     */
    function showEdgeDetail(edge) {
        const panel = document.getElementById('graph-detail-panel');
        if (!panel) return;

        const data = edge.data();
        const sourceNode = cy.getElementById(data.source);
        const targetNode = cy.getElementById(data.target);

        let html = `<button class="detail-close" onclick="DRXGraph.hideDetailPanel()" aria-label="Close">&times;</button>`;
        html += `<h4>Relationship</h4>`;

        html += `<div class="edge-detail-nodes">
            <div class="edge-source" onclick="DRXGraph.focusNode('${escapeHtml(data.source)}')">
                ${escapeHtml(sourceNode.data('label') || data.source)}
            </div>
            <div class="edge-relation">
                <span class="relation-arrow">&darr;</span>
                <span class="relation-label">${escapeHtml(data.label || 'relates to')}</span>
                <span class="relation-arrow">&darr;</span>
            </div>
            <div class="edge-target" onclick="DRXGraph.focusNode('${escapeHtml(data.target)}')">
                ${escapeHtml(targetNode.data('label') || data.target)}
            </div>
        </div>`;

        if (data.confidence !== undefined) {
            const pct = (data.confidence * 100).toFixed(0);
            html += `<div class="detail-confidence">
                <span class="confidence-label">Confidence:</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${pct}%"></div>
                </div>
                <span class="confidence-value">${pct}%</span>
            </div>`;
        }

        if (data.evidence) {
            html += `<div class="detail-evidence">
                <strong>Evidence:</strong>
                <p>${escapeHtml(data.evidence)}</p>
            </div>`;
        }

        panel.innerHTML = html;
        panel.classList.add('visible');
    }

    /**
     * Hide detail panel
     */
    function hideDetailPanel() {
        const panel = document.getElementById('graph-detail-panel');
        if (panel) {
            panel.classList.remove('visible');
        }
    }

    // ==========================================================================
    // Data Loading
    // ==========================================================================

    /**
     * Load graph from Cytoscape JSON format (from backend)
     * @param {Object} cytoscapeJson - Cytoscape-compatible graph data
     * @param {Object} options - Layout options
     */
    function loadFromCytoscape(cytoscapeJson, options = {}) {
        if (!cy) {
            console.error('[DRXGraph] Graph not initialized. Call init() first.');
            return;
        }

        // Clear existing elements
        cy.elements().remove();

        // Process nodes to add computed properties
        const elements = {
            nodes: [],
            edges: []
        };

        if (cytoscapeJson.nodes) {
            elements.nodes = cytoscapeJson.nodes.map(node => {
                const data = { ...node.data };

                // Add computed styling properties
                data.bgColor = ENTITY_COLORS[data.type] || ENTITY_COLORS.concept;
                data.borderColor = darkenColor(data.bgColor, 20);
                data.weight = data.weight || 50;

                return { data };
            });
        }

        if (cytoscapeJson.edges) {
            elements.edges = cytoscapeJson.edges.map(edge => ({
                data: { ...edge.data }
            }));
        }

        // Add claims as nodes if provided separately
        if (cytoscapeJson.claims) {
            cytoscapeJson.claims.forEach(claim => {
                elements.nodes.push({
                    data: {
                        id: claim.id,
                        label: truncateText(claim.statement, 30),
                        statement: claim.statement,
                        nodeType: 'claim',
                        status: claim.status || 'unverified',
                        confidence: claim.confidence,
                        bgColor: CLAIM_COLORS[claim.status] || CLAIM_COLORS.unverified,
                        borderColor: darkenColor(CLAIM_COLORS[claim.status] || CLAIM_COLORS.unverified, 20),
                        weight: 40
                    }
                });
            });
        }

        // Add elements to graph
        cy.add(elements);

        // Run layout
        const layoutOptions = {
            name: options.layout || 'cose',
            animate: options.animate !== false,
            animationDuration: options.animationDuration || 500,
            fit: true,
            padding: 40,
            nodeDimensionsIncludeLabels: true,
            ...options.layoutOptions
        };

        cy.layout(layoutOptions).run();

        // Fit to viewport after layout
        setTimeout(() => {
            cy.fit(undefined, 40);
        }, layoutOptions.animationDuration + 100);

        console.log(`[DRXGraph] Loaded ${elements.nodes.length} nodes and ${elements.edges.length} edges`);
    }

    /**
     * Add nodes incrementally
     * @param {Array} nodes - Array of node data
     */
    function addNodes(nodes) {
        if (!cy) return;

        const elements = nodes.map(node => {
            const data = { ...node.data || node };
            data.bgColor = ENTITY_COLORS[data.type] || ENTITY_COLORS.concept;
            data.borderColor = darkenColor(data.bgColor, 20);
            data.weight = data.weight || 50;
            return { data };
        });

        cy.add(elements);
    }

    /**
     * Add edges incrementally
     * @param {Array} edges - Array of edge data
     */
    function addEdges(edges) {
        if (!cy) return;

        const elements = edges.map(edge => ({
            data: { ...edge.data || edge }
        }));

        cy.add(elements);
    }

    // ==========================================================================
    // Filtering Functions
    // ==========================================================================

    /**
     * Filter graph by entity type(s)
     * @param {Array|string} types - Entity type(s) to show, empty array shows all
     */
    function filterByType(types) {
        if (!cy) return;

        const typeArray = Array.isArray(types) ? types : [types];
        const showAll = typeArray.length === 0 || typeArray.includes('');

        cy.batch(() => {
            cy.nodes().forEach(node => {
                const nodeType = node.data('type') || node.data('nodeType');

                if (showAll || typeArray.includes(nodeType)) {
                    node.removeClass('hidden dimmed');
                } else {
                    node.addClass('hidden');
                }
            });

            // Hide edges connected to hidden nodes
            cy.edges().forEach(edge => {
                const source = cy.getElementById(edge.data('source'));
                const target = cy.getElementById(edge.data('target'));

                if (source.hasClass('hidden') || target.hasClass('hidden')) {
                    edge.addClass('hidden');
                } else {
                    edge.removeClass('hidden');
                }
            });
        });
    }

    /**
     * Filter claims by status
     * @param {Array|string} statuses - Status(es) to show
     */
    function filterByClaimStatus(statuses) {
        if (!cy) return;

        const statusArray = Array.isArray(statuses) ? statuses : [statuses];
        const showAll = statusArray.length === 0;

        cy.batch(() => {
            cy.nodes('[nodeType="claim"]').forEach(node => {
                const status = node.data('status');

                if (showAll || statusArray.includes(status)) {
                    node.removeClass('hidden dimmed');
                } else {
                    node.addClass('hidden');
                }
            });
        });
    }

    /**
     * Filter by confidence threshold
     * @param {number} minConfidence - Minimum confidence (0-1)
     */
    function filterByConfidence(minConfidence) {
        if (!cy) return;

        cy.batch(() => {
            cy.edges().forEach(edge => {
                const confidence = edge.data('confidence') || 1;

                if (confidence >= minConfidence) {
                    edge.removeClass('dimmed');
                } else {
                    edge.addClass('dimmed');
                }
            });
        });
    }

    /**
     * Reset all filters
     */
    function resetFilters() {
        if (!cy) return;

        cy.batch(() => {
            cy.elements().removeClass('hidden dimmed');
        });
    }

    // ==========================================================================
    // Highlighting Functions
    // ==========================================================================

    /**
     * Highlight nodes connected to a given node
     * @param {Object} node - Cytoscape node
     */
    function highlightConnected(node) {
        if (!cy) return;

        clearHighlights();

        const connected = node.closedNeighborhood();

        cy.batch(() => {
            // Dim all elements
            cy.elements().addClass('dimmed');

            // Highlight connected elements
            connected.removeClass('dimmed').addClass('highlighted');
        });
    }

    /**
     * Highlight path between two nodes
     * @param {string} sourceId - Source node ID
     * @param {string} targetId - Target node ID
     * @returns {boolean} Whether a path was found
     */
    function highlightPath(sourceId, targetId) {
        if (!cy) return false;

        const source = cy.getElementById(sourceId);
        const target = cy.getElementById(targetId);

        if (source.empty() || target.empty()) {
            console.warn('[DRXGraph] Source or target node not found');
            return false;
        }

        // Find shortest path using A* algorithm
        const path = cy.elements().aStar({
            root: source,
            goal: target,
            weight: edge => 1 / (edge.data('confidence') || 0.5)
        });

        if (path.found) {
            clearHighlights();

            cy.batch(() => {
                cy.elements().addClass('dimmed');
                path.path.removeClass('dimmed').addClass('highlighted');
            });

            return true;
        }

        console.log('[DRXGraph] No path found between nodes');
        return false;
    }

    /**
     * Clear all highlights
     */
    function clearHighlights() {
        if (!cy) return;

        cy.batch(() => {
            cy.elements().removeClass('highlighted dimmed');
        });
    }

    // ==========================================================================
    // Navigation Functions
    // ==========================================================================

    /**
     * Focus on a specific node
     * @param {string} nodeId - Node ID to focus on
     */
    function focusNode(nodeId) {
        if (!cy) return;

        const node = cy.getElementById(nodeId);
        if (node.empty()) {
            console.warn('[DRXGraph] Node not found:', nodeId);
            return;
        }

        // Center on node
        cy.animate({
            center: { eles: node },
            zoom: 1.5,
            duration: 300
        });

        // Select and show details
        node.select();
        showNodeDetail(node);
        highlightConnected(node);
    }

    /**
     * Fit graph to viewport
     * @param {number} padding - Padding around the graph
     */
    function fit(padding = 40) {
        if (!cy) return;
        cy.fit(undefined, padding);
    }

    /**
     * Center graph
     */
    function center() {
        if (!cy) return;
        cy.center();
    }

    /**
     * Reset zoom to default
     */
    function resetZoom() {
        if (!cy) return;
        cy.zoom(1);
        cy.center();
    }

    // ==========================================================================
    // Layout Functions
    // ==========================================================================

    /**
     * Re-run layout
     * @param {string} layoutName - Layout name (cose, grid, circle, etc.)
     * @param {Object} options - Layout options
     */
    function runLayout(layoutName = 'cose', options = {}) {
        if (!cy) return;

        const layoutOptions = {
            name: layoutName,
            animate: true,
            animationDuration: 500,
            fit: true,
            padding: 40,
            nodeDimensionsIncludeLabels: true,
            ...options
        };

        cy.layout(layoutOptions).run();
    }

    // ==========================================================================
    // Export Functions
    // ==========================================================================

    /**
     * Export graph as PNG image
     * @param {Object} options - Export options
     */
    function exportPNG(options = {}) {
        if (!cy) return;

        const defaults = {
            scale: 2,
            full: true,
            bg: getComputedStyle(document.documentElement)
                .getPropertyValue('--bg-secondary').trim() || '#161b22'
        };

        const exportOptions = { ...defaults, ...options };
        const png = cy.png(exportOptions);

        // Create download link
        const link = document.createElement('a');
        link.href = png;
        link.download = `knowledge-graph-${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        console.log('[DRXGraph] PNG exported');
    }

    /**
     * Export graph as JSON
     * @returns {Object} Cytoscape JSON format
     */
    function exportJSON() {
        if (!cy) return null;

        return cy.json().elements;
    }

    // ==========================================================================
    // Utility Functions
    // ==========================================================================

    /**
     * Clear all elements from graph
     */
    function clear() {
        if (cy) {
            cy.elements().remove();
        }
        hideDetailPanel();
        hideTooltip();
    }

    /**
     * Destroy graph instance
     */
    function destroy() {
        if (cy) {
            cy.destroy();
            cy = null;
        }
        if (tooltipEl) {
            tooltipEl.remove();
            tooltipEl = null;
        }
        container = null;
    }

    /**
     * Get graph statistics
     * @returns {Object} Graph statistics
     */
    function getStats() {
        if (!cy) return null;

        const nodes = cy.nodes();
        const edges = cy.edges();

        const typeCounts = {};
        nodes.forEach(node => {
            const type = node.data('type') || node.data('nodeType') || 'unknown';
            typeCounts[type] = (typeCounts[type] || 0) + 1;
        });

        return {
            nodeCount: nodes.length,
            edgeCount: edges.length,
            typeCounts,
            avgDegree: nodes.length > 0 ? (edges.length * 2 / nodes.length).toFixed(2) : 0
        };
    }

    /**
     * Escape HTML to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    function escapeHtml(text) {
        if (typeof text !== 'string') return String(text);
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Darken a hex color
     * @param {string} hex - Hex color
     * @param {number} percent - Darkening percentage
     * @returns {string} Darkened hex color
     */
    function darkenColor(hex, percent) {
        const num = parseInt(hex.replace('#', ''), 16);
        const amt = Math.round(2.55 * percent);
        const R = Math.max(0, (num >> 16) - amt);
        const G = Math.max(0, ((num >> 8) & 0x00FF) - amt);
        const B = Math.max(0, (num & 0x0000FF) - amt);
        return '#' + (0x1000000 + R * 0x10000 + G * 0x100 + B).toString(16).slice(1);
    }

    /**
     * Truncate text with ellipsis
     * @param {string} text - Text to truncate
     * @param {number} maxLength - Maximum length
     * @returns {string} Truncated text
     */
    function truncateText(text, maxLength) {
        if (!text || text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }

    // ==========================================================================
    // Public API
    // ==========================================================================

    return {
        // Initialization
        init,
        destroy,

        // Data loading
        loadFromCytoscape,
        addNodes,
        addEdges,

        // Filtering
        filterByType,
        filterByClaimStatus,
        filterByConfidence,
        resetFilters,

        // Highlighting
        highlightConnected,
        highlightPath,
        clearHighlights,

        // Navigation
        focusNode,
        fit,
        center,
        resetZoom,

        // Layout
        runLayout,

        // Export
        exportPNG,
        exportJSON,

        // Utility
        clear,
        getStats,
        hideDetailPanel,

        // Constants
        ENTITY_COLORS,
        CLAIM_COLORS
    };
})();

// Export to window for global access
window.DRXGraph = DRXGraph;
