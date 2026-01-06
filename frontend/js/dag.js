/**
 * DRX DAG Visualization Module
 * Real-time directed acyclic graph visualization for research workflow
 *
 * Uses D3.js for rendering with hierarchical layout optimized for DAGs.
 * Supports real-time updates via SSE events.
 */

const DRXDAG = (function() {
    'use strict';

    // ==========================================================================
    // Configuration
    // ==========================================================================

    const CONFIG = {
        // Node dimensions
        nodeWidth: 140,
        nodeHeight: 50,
        nodeRadius: 8,

        // Layout spacing
        levelHeight: 100,
        nodeSpacing: 180,
        padding: 40,

        // Animation durations (ms)
        transitionDuration: 300,
        pulseInterval: 1500,

        // Colors (will be overridden by CSS custom properties)
        colors: {
            pending: '#8b949e',
            running: '#1f6feb',
            completed: '#238636',
            failed: '#f85149',
            edge: '#30363d',
            edgeActive: '#1f6feb',
            background: '#0d1117',
            text: '#e6edf3'
        },

        // Agent type colors
        agentColors: {
            planner: '#8957e5',
            searcher: '#1f6feb',
            reader: '#d29922',
            synthesizer: '#238636',
            critic: '#f85149',
            reporter: '#58a6ff'
        }
    };

    // Node states
    const NODE_STATES = {
        PENDING: 'pending',
        RUNNING: 'running',
        COMPLETED: 'completed',
        FAILED: 'failed'
    };

    // ==========================================================================
    // Private State
    // ==========================================================================

    let svg = null;
    let svgGroup = null;
    let zoom = null;
    let nodes = [];
    let edges = [];
    let nodeMap = new Map();
    let containerId = null;
    let containerEl = null;
    let tooltipEl = null;
    let isInitialized = false;
    let animationFrameId = null;

    // ==========================================================================
    // Initialization
    // ==========================================================================

    /**
     * Initialize the DAG visualization
     * @param {string} id - Container element ID
     * @param {object} options - Configuration options
     */
    function init(id, options = {}) {
        containerId = id;
        containerEl = document.getElementById(id);

        if (!containerEl) {
            console.error('[DRXDAG] Container element not found:', id);
            return false;
        }

        // Check for D3.js
        if (typeof d3 === 'undefined') {
            console.error('[DRXDAG] D3.js is required but not loaded');
            return false;
        }

        // Merge options
        Object.assign(CONFIG, options);

        // Clear existing content
        clear();

        // Get container dimensions
        const rect = containerEl.getBoundingClientRect();
        const width = rect.width || 600;
        const height = rect.height || 400;

        // Create SVG element
        svg = d3.select(containerEl)
            .append('svg')
            .attr('class', 'dag-svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('preserveAspectRatio', 'xMidYMid meet');

        // Add defs for markers (arrowheads)
        const defs = svg.append('defs');

        // Arrow marker for edges
        defs.append('marker')
            .attr('id', 'dag-arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('class', 'dag-arrow-path');

        // Active arrow marker
        defs.append('marker')
            .attr('id', 'dag-arrow-active')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('class', 'dag-arrow-path active');

        // Create main group for zoom/pan
        svgGroup = svg.append('g')
            .attr('class', 'dag-main-group');

        // Add edge group (rendered first, behind nodes)
        svgGroup.append('g').attr('class', 'dag-edges');

        // Add node group
        svgGroup.append('g').attr('class', 'dag-nodes');

        // Setup zoom behavior
        zoom = d3.zoom()
            .scaleExtent([0.3, 2])
            .on('zoom', (event) => {
                svgGroup.attr('transform', event.transform);
            });

        svg.call(zoom);

        // Create tooltip element
        createTooltip();

        isInitialized = true;
        console.log('[DRXDAG] Initialized');

        return true;
    }

    /**
     * Create tooltip element
     */
    function createTooltip() {
        // Remove existing tooltip
        const existing = containerEl.querySelector('.dag-tooltip');
        if (existing) existing.remove();

        tooltipEl = document.createElement('div');
        tooltipEl.className = 'dag-tooltip';
        containerEl.appendChild(tooltipEl);
    }

    // ==========================================================================
    // Node Management
    // ==========================================================================

    /**
     * Add a node to the graph
     * @param {string} id - Unique node ID
     * @param {string} label - Display label
     * @param {string} type - Agent type (planner, searcher, etc.)
     * @param {string} status - Node status (pending, running, completed, failed)
     * @param {object} metadata - Additional node metadata
     */
    function addNode(id, label, type, status = NODE_STATES.PENDING, metadata = {}) {
        if (!isInitialized) {
            console.warn('[DRXDAG] Not initialized');
            return;
        }

        // Check if node already exists
        if (nodeMap.has(id)) {
            updateNodeStatus(id, status, metadata);
            return;
        }

        const node = {
            id,
            label: label || id,
            type: type || 'unknown',
            status: status || NODE_STATES.PENDING,
            metadata,
            x: 0,
            y: 0,
            level: 0,
            dependencies: metadata.dependencies || []
        };

        nodes.push(node);
        nodeMap.set(id, node);

        // Recalculate layout
        calculateLayout();
        render();
    }

    /**
     * Add multiple nodes at once
     * @param {Array} nodeList - Array of node objects
     */
    function addNodes(nodeList) {
        if (!isInitialized || !Array.isArray(nodeList)) return;

        nodeList.forEach(n => {
            if (!nodeMap.has(n.id)) {
                const node = {
                    id: n.id,
                    label: n.label || n.id,
                    type: n.type || n.agent_type || 'unknown',
                    status: n.status || NODE_STATES.PENDING,
                    metadata: n.metadata || {},
                    x: 0,
                    y: 0,
                    level: 0,
                    dependencies: n.dependencies || []
                };
                nodes.push(node);
                nodeMap.set(n.id, node);
            }
        });

        // Rebuild edges from dependencies
        rebuildEdges();
        calculateLayout();
        render();
    }

    /**
     * Add an edge between nodes
     * @param {string} sourceId - Source node ID
     * @param {string} targetId - Target node ID
     */
    function addEdge(sourceId, targetId) {
        if (!isInitialized) return;

        // Check if edge already exists
        const exists = edges.some(e => e.source === sourceId && e.target === targetId);
        if (exists) return;

        // Verify both nodes exist
        if (!nodeMap.has(sourceId) || !nodeMap.has(targetId)) {
            console.warn('[DRXDAG] Cannot add edge - node not found:', sourceId, targetId);
            return;
        }

        edges.push({
            source: sourceId,
            target: targetId,
            active: false
        });

        // Recalculate layout
        calculateLayout();
        render();
    }

    /**
     * Rebuild edges from node dependencies
     */
    function rebuildEdges() {
        edges = [];
        nodes.forEach(node => {
            if (node.dependencies && node.dependencies.length > 0) {
                node.dependencies.forEach(depId => {
                    if (nodeMap.has(depId)) {
                        edges.push({
                            source: depId,
                            target: node.id,
                            active: false
                        });
                    }
                });
            }
        });
    }

    /**
     * Update node status with animation
     * @param {string} nodeId - Node ID
     * @param {string} status - New status
     * @param {object} metadata - Additional metadata
     */
    function updateNodeStatus(nodeId, status, metadata = {}) {
        const node = nodeMap.get(nodeId);
        if (!node) {
            console.warn('[DRXDAG] Node not found:', nodeId);
            return;
        }

        const oldStatus = node.status;
        node.status = status;
        Object.assign(node.metadata, metadata);

        // Update edges involving this node
        edges.forEach(edge => {
            if (edge.target === nodeId && status === NODE_STATES.RUNNING) {
                edge.active = true;
            } else if (edge.source === nodeId && status === NODE_STATES.COMPLETED) {
                // Keep edge active until target starts
            }
        });

        // Re-render with transition
        render(true);

        console.log(`[DRXDAG] Node ${nodeId}: ${oldStatus} -> ${status}`);
    }

    // ==========================================================================
    // Layout Calculation
    // ==========================================================================

    /**
     * Calculate hierarchical layout for DAG
     * Uses topological sorting to assign levels
     */
    function calculateLayout() {
        if (nodes.length === 0) return;

        // Build adjacency list for topological sort
        const inDegree = new Map();
        const adjList = new Map();

        nodes.forEach(node => {
            inDegree.set(node.id, 0);
            adjList.set(node.id, []);
        });

        edges.forEach(edge => {
            const current = inDegree.get(edge.target) || 0;
            inDegree.set(edge.target, current + 1);

            const adj = adjList.get(edge.source) || [];
            adj.push(edge.target);
            adjList.set(edge.source, adj);
        });

        // Assign levels using BFS (Kahn's algorithm)
        const queue = [];
        const levels = new Map();

        // Start with nodes that have no dependencies
        nodes.forEach(node => {
            if (inDegree.get(node.id) === 0) {
                queue.push(node.id);
                levels.set(node.id, 0);
            }
        });

        while (queue.length > 0) {
            const nodeId = queue.shift();
            const nodeLevel = levels.get(nodeId);
            const neighbors = adjList.get(nodeId) || [];

            neighbors.forEach(neighborId => {
                const newDegree = inDegree.get(neighborId) - 1;
                inDegree.set(neighborId, newDegree);

                // Set level to max of current level and parent level + 1
                const currentLevel = levels.get(neighborId) || 0;
                levels.set(neighborId, Math.max(currentLevel, nodeLevel + 1));

                if (newDegree === 0) {
                    queue.push(neighborId);
                }
            });
        }

        // Group nodes by level
        const levelGroups = new Map();
        nodes.forEach(node => {
            const level = levels.get(node.id) || 0;
            node.level = level;

            if (!levelGroups.has(level)) {
                levelGroups.set(level, []);
            }
            levelGroups.get(level).push(node);
        });

        // Calculate positions
        const maxLevel = Math.max(...levelGroups.keys(), 0);
        const rect = containerEl.getBoundingClientRect();
        const width = rect.width || 600;
        const height = rect.height || 400;

        levelGroups.forEach((levelNodes, level) => {
            const nodeCount = levelNodes.length;
            const totalWidth = nodeCount * CONFIG.nodeSpacing;
            const startX = (width - totalWidth) / 2 + CONFIG.nodeSpacing / 2;
            const y = CONFIG.padding + level * CONFIG.levelHeight;

            levelNodes.forEach((node, index) => {
                node.x = startX + index * CONFIG.nodeSpacing;
                node.y = y;
            });
        });

        // Update viewBox to fit content
        const maxY = CONFIG.padding + (maxLevel + 1) * CONFIG.levelHeight;
        svg.attr('viewBox', `0 0 ${width} ${Math.max(height, maxY + CONFIG.padding)}`);
    }

    // ==========================================================================
    // Rendering
    // ==========================================================================

    /**
     * Render the current state
     * @param {boolean} animate - Whether to animate transitions
     */
    function render(animate = false) {
        if (!isInitialized || !svg) return;

        const duration = animate ? CONFIG.transitionDuration : 0;

        renderEdges(duration);
        renderNodes(duration);
    }

    /**
     * Render edges (lines with arrows)
     */
    function renderEdges(duration) {
        const edgeGroup = svgGroup.select('.dag-edges');

        // Prepare edge data with coordinates
        const edgeData = edges.map(edge => {
            const source = nodeMap.get(edge.source);
            const target = nodeMap.get(edge.target);
            return {
                ...edge,
                sourceNode: source,
                targetNode: target,
                x1: source ? source.x : 0,
                y1: source ? source.y + CONFIG.nodeHeight / 2 : 0,
                x2: target ? target.x : 0,
                y2: target ? target.y - CONFIG.nodeHeight / 2 : 0
            };
        }).filter(e => e.sourceNode && e.targetNode);

        // Bind data
        const edgeSelection = edgeGroup.selectAll('.dag-edge')
            .data(edgeData, d => `${d.source}-${d.target}`);

        // Enter new edges
        const edgeEnter = edgeSelection.enter()
            .append('path')
            .attr('class', 'dag-edge')
            .attr('marker-end', 'url(#dag-arrow)')
            .attr('d', d => createCurvedPath(d.x1, d.y1, d.x2, d.y2))
            .style('opacity', 0);

        // Update all edges
        edgeSelection.merge(edgeEnter)
            .transition()
            .duration(duration)
            .attr('d', d => createCurvedPath(d.x1, d.y1, d.x2, d.y2))
            .attr('class', d => `dag-edge ${d.active ? 'active' : ''}`)
            .attr('marker-end', d => d.active ? 'url(#dag-arrow-active)' : 'url(#dag-arrow)')
            .style('opacity', 1);

        // Remove old edges
        edgeSelection.exit()
            .transition()
            .duration(duration)
            .style('opacity', 0)
            .remove();
    }

    /**
     * Create curved path for edge
     */
    function createCurvedPath(x1, y1, x2, y2) {
        const midY = (y1 + y2) / 2;
        return `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;
    }

    /**
     * Render nodes (rectangles with labels)
     */
    function renderNodes(duration) {
        const nodeGroup = svgGroup.select('.dag-nodes');

        // Bind data
        const nodeSelection = nodeGroup.selectAll('.dag-node')
            .data(nodes, d => d.id);

        // Enter new nodes
        const nodeEnter = nodeSelection.enter()
            .append('g')
            .attr('class', d => `dag-node ${d.status}`)
            .attr('transform', d => `translate(${d.x}, ${d.y})`)
            .style('opacity', 0)
            .on('mouseenter', showTooltip)
            .on('mouseleave', hideTooltip)
            .on('click', handleNodeClick);

        // Add background rectangle
        nodeEnter.append('rect')
            .attr('class', 'dag-node-bg')
            .attr('x', -CONFIG.nodeWidth / 2)
            .attr('y', -CONFIG.nodeHeight / 2)
            .attr('width', CONFIG.nodeWidth)
            .attr('height', CONFIG.nodeHeight)
            .attr('rx', CONFIG.nodeRadius);

        // Add agent type indicator (colored bar)
        nodeEnter.append('rect')
            .attr('class', 'dag-node-type')
            .attr('x', -CONFIG.nodeWidth / 2)
            .attr('y', -CONFIG.nodeHeight / 2)
            .attr('width', 4)
            .attr('height', CONFIG.nodeHeight)
            .attr('rx', 2)
            .style('fill', d => CONFIG.agentColors[d.type] || CONFIG.colors.pending);

        // Add label
        nodeEnter.append('text')
            .attr('class', 'dag-label')
            .attr('dy', '0.35em')
            .text(d => truncateLabel(d.label, 16));

        // Add status icon
        nodeEnter.append('text')
            .attr('class', 'dag-status-icon')
            .attr('x', CONFIG.nodeWidth / 2 - 16)
            .attr('y', 0)
            .attr('dy', '0.35em')
            .text(d => getStatusIcon(d.status));

        // Update all nodes
        const nodeUpdate = nodeSelection.merge(nodeEnter);

        nodeUpdate
            .transition()
            .duration(duration)
            .attr('class', d => `dag-node ${d.status}`)
            .attr('transform', d => `translate(${d.x}, ${d.y})`)
            .style('opacity', 1);

        // Update status icons
        nodeUpdate.select('.dag-status-icon')
            .text(d => getStatusIcon(d.status));

        // Update type indicator color
        nodeUpdate.select('.dag-node-type')
            .style('fill', d => CONFIG.agentColors[d.type] || CONFIG.colors.pending);

        // Remove old nodes
        nodeSelection.exit()
            .transition()
            .duration(duration)
            .style('opacity', 0)
            .remove();
    }

    /**
     * Get status icon character
     */
    function getStatusIcon(status) {
        switch (status) {
            case NODE_STATES.RUNNING:
                return '\u25CF'; // Filled circle
            case NODE_STATES.COMPLETED:
                return '\u2713'; // Check mark
            case NODE_STATES.FAILED:
                return '\u2717'; // X mark
            default:
                return '\u25CB'; // Empty circle
        }
    }

    /**
     * Truncate label to fit
     */
    function truncateLabel(label, maxLength) {
        if (!label) return '';
        if (label.length <= maxLength) return label;
        return label.substring(0, maxLength - 3) + '...';
    }

    // ==========================================================================
    // Tooltip Handling
    // ==========================================================================

    /**
     * Show tooltip on hover
     */
    function showTooltip(event, d) {
        if (!tooltipEl) return;

        const node = d;
        let content = `<strong>${node.label}</strong><br>`;
        content += `Type: ${node.type}<br>`;
        content += `Status: ${node.status}`;

        if (node.metadata) {
            if (node.metadata.execution_time_ms) {
                content += `<br>Time: ${(node.metadata.execution_time_ms / 1000).toFixed(1)}s`;
            }
            if (node.metadata.error) {
                content += `<br><span class="error">Error: ${node.metadata.error}</span>`;
            }
        }

        tooltipEl.innerHTML = content;
        tooltipEl.classList.add('visible');

        // Position tooltip
        const rect = containerEl.getBoundingClientRect();
        const x = event.clientX - rect.left + 10;
        const y = event.clientY - rect.top + 10;

        tooltipEl.style.left = `${x}px`;
        tooltipEl.style.top = `${y}px`;
    }

    /**
     * Hide tooltip
     */
    function hideTooltip() {
        if (tooltipEl) {
            tooltipEl.classList.remove('visible');
        }
    }

    /**
     * Handle node click
     */
    function handleNodeClick(event, d) {
        console.log('[DRXDAG] Node clicked:', d);
        // Could emit custom event here for external handling
        const customEvent = new CustomEvent('dag-node-click', { detail: d });
        containerEl.dispatchEvent(customEvent);
    }

    // ==========================================================================
    // SSE Event Integration
    // ==========================================================================

    /**
     * Update graph from SSE event
     * @param {object} event - SSE event data
     */
    function updateFromEvent(event) {
        if (!isInitialized) {
            console.warn('[DRXDAG] Cannot update - not initialized');
            return;
        }

        const eventType = event.type || event.event_type;

        switch (eventType) {
            case 'workflow_state':
            case 'dag_state':
                handleWorkflowState(event);
                break;

            case 'agent_start':
                handleAgentStart(event);
                break;

            case 'agent_end':
            case 'agent_complete':
                handleAgentComplete(event);
                break;

            case 'task_start':
                handleTaskStart(event);
                break;

            case 'task_complete':
            case 'task_end':
                handleTaskComplete(event);
                break;

            case 'task_error':
            case 'task_failed':
                handleTaskFailed(event);
                break;

            case 'plan_created':
            case 'dag_created':
                handlePlanCreated(event);
                break;

            default:
                // Try to infer from event structure
                if (event.dag_nodes || event.nodes) {
                    handleWorkflowState(event);
                } else if (event.node_id || event.task_id) {
                    handleGenericNodeUpdate(event);
                }
        }
    }

    /**
     * Handle workflow state update (full DAG)
     */
    function handleWorkflowState(event) {
        const dagNodes = event.dag_nodes || event.nodes || [];

        if (dagNodes.length === 0) return;

        // Clear existing nodes and rebuild
        nodes = [];
        edges = [];
        nodeMap.clear();

        dagNodes.forEach(n => {
            const node = {
                id: n.id,
                label: n.label || n.description || n.id,
                type: n.agent_type || n.type || 'unknown',
                status: mapStatus(n.status),
                metadata: {
                    description: n.description,
                    inputs: n.inputs,
                    outputs: n.outputs,
                    error: n.error,
                    execution_time_ms: n.execution_time_ms
                },
                x: 0,
                y: 0,
                level: 0,
                dependencies: n.dependencies || []
            };
            nodes.push(node);
            nodeMap.set(n.id, node);
        });

        rebuildEdges();
        calculateLayout();
        render(true);
    }

    /**
     * Handle plan created event
     */
    function handlePlanCreated(event) {
        const plan = event.plan || event;
        const dagNodes = plan.dag_nodes || plan.tasks || [];

        if (dagNodes.length > 0) {
            handleWorkflowState({ dag_nodes: dagNodes });
        }
    }

    /**
     * Handle agent start
     */
    function handleAgentStart(event) {
        const nodeId = event.node_id || event.task_id || event.agent;
        const agent = event.agent || event.agent_type;

        // Try to find node by ID or agent type
        let node = nodeMap.get(nodeId);

        if (!node && agent) {
            // Find first pending node of this agent type
            node = nodes.find(n => n.type === agent && n.status === NODE_STATES.PENDING);
        }

        if (node) {
            updateNodeStatus(node.id, NODE_STATES.RUNNING, {
                started_at: new Date().toISOString()
            });
        }
    }

    /**
     * Handle agent complete
     */
    function handleAgentComplete(event) {
        const nodeId = event.node_id || event.task_id || event.agent;
        const agent = event.agent || event.agent_type;

        let node = nodeMap.get(nodeId);

        if (!node && agent) {
            // Find running node of this agent type
            node = nodes.find(n => n.type === agent && n.status === NODE_STATES.RUNNING);
        }

        if (node) {
            updateNodeStatus(node.id, NODE_STATES.COMPLETED, {
                summary: event.summary,
                completed_at: new Date().toISOString()
            });
        }
    }

    /**
     * Handle task start
     */
    function handleTaskStart(event) {
        const nodeId = event.task_id || event.node_id;
        if (nodeId && nodeMap.has(nodeId)) {
            updateNodeStatus(nodeId, NODE_STATES.RUNNING);
        }
    }

    /**
     * Handle task complete
     */
    function handleTaskComplete(event) {
        const nodeId = event.task_id || event.node_id;
        if (nodeId && nodeMap.has(nodeId)) {
            updateNodeStatus(nodeId, NODE_STATES.COMPLETED, {
                outputs: event.outputs,
                execution_time_ms: event.execution_time_ms
            });
        }
    }

    /**
     * Handle task failed
     */
    function handleTaskFailed(event) {
        const nodeId = event.task_id || event.node_id;
        if (nodeId && nodeMap.has(nodeId)) {
            updateNodeStatus(nodeId, NODE_STATES.FAILED, {
                error: event.error || event.message
            });
        }
    }

    /**
     * Handle generic node update
     */
    function handleGenericNodeUpdate(event) {
        const nodeId = event.node_id || event.task_id;
        const status = mapStatus(event.status);

        if (nodeId && nodeMap.has(nodeId)) {
            updateNodeStatus(nodeId, status, event);
        }
    }

    /**
     * Map backend status to frontend status
     */
    function mapStatus(status) {
        if (!status) return NODE_STATES.PENDING;

        const statusLower = status.toLowerCase();

        switch (statusLower) {
            case 'running':
            case 'in_progress':
            case 'active':
            case 'executing':
                return NODE_STATES.RUNNING;
            case 'completed':
            case 'done':
            case 'success':
            case 'finished':
                return NODE_STATES.COMPLETED;
            case 'failed':
            case 'error':
            case 'cancelled':
                return NODE_STATES.FAILED;
            default:
                return NODE_STATES.PENDING;
        }
    }

    // ==========================================================================
    // View Controls
    // ==========================================================================

    /**
     * Reset zoom to fit all content
     */
    function resetView() {
        if (!svg || !zoom) return;

        svg.transition()
            .duration(CONFIG.transitionDuration)
            .call(zoom.transform, d3.zoomIdentity);
    }

    /**
     * Fit graph to container
     */
    function fitToContainer() {
        if (!svg || !svgGroup || nodes.length === 0) return;

        const rect = containerEl.getBoundingClientRect();
        const bounds = svgGroup.node().getBBox();

        const width = rect.width;
        const height = rect.height;
        const fullWidth = bounds.width + CONFIG.padding * 2;
        const fullHeight = bounds.height + CONFIG.padding * 2;

        const scale = Math.min(
            width / fullWidth,
            height / fullHeight,
            1
        ) * 0.9;

        const translateX = (width - bounds.width * scale) / 2 - bounds.x * scale;
        const translateY = (height - bounds.height * scale) / 2 - bounds.y * scale;

        svg.transition()
            .duration(CONFIG.transitionDuration)
            .call(
                zoom.transform,
                d3.zoomIdentity
                    .translate(translateX, translateY)
                    .scale(scale)
            );
    }

    /**
     * Clear the visualization
     */
    function clear() {
        nodes = [];
        edges = [];
        nodeMap.clear();

        if (svgGroup) {
            svgGroup.selectAll('.dag-node').remove();
            svgGroup.selectAll('.dag-edge').remove();
        }

        if (tooltipEl) {
            tooltipEl.classList.remove('visible');
        }
    }

    /**
     * Destroy the visualization
     */
    function destroy() {
        clear();

        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }

        if (svg) {
            svg.remove();
            svg = null;
        }

        if (tooltipEl) {
            tooltipEl.remove();
            tooltipEl = null;
        }

        svgGroup = null;
        zoom = null;
        isInitialized = false;
    }

    // ==========================================================================
    // Export Functionality
    // ==========================================================================

    /**
     * Export current graph as PNG
     */
    function exportPNG() {
        if (!svg || !isInitialized) {
            console.warn('[DRXDAG] Cannot export - not initialized');
            return;
        }

        // Get SVG element
        const svgElement = svg.node();
        const svgData = new XMLSerializer().serializeToString(svgElement);

        // Create canvas
        const canvas = document.createElement('canvas');
        const rect = containerEl.getBoundingClientRect();
        canvas.width = rect.width * 2; // 2x for retina
        canvas.height = rect.height * 2;

        const ctx = canvas.getContext('2d');
        ctx.scale(2, 2);

        // Create image from SVG
        const img = new Image();
        const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);

        img.onload = function() {
            // Fill background
            ctx.fillStyle = getComputedStyle(document.documentElement)
                .getPropertyValue('--bg-primary') || CONFIG.colors.background;
            ctx.fillRect(0, 0, rect.width, rect.height);

            // Draw SVG
            ctx.drawImage(img, 0, 0, rect.width, rect.height);

            // Create download link
            const a = document.createElement('a');
            a.download = `drx-dag-${Date.now()}.png`;
            a.href = canvas.toDataURL('image/png');
            a.click();

            URL.revokeObjectURL(url);
        };

        img.src = url;
    }

    /**
     * Export current graph as SVG
     */
    function exportSVG() {
        if (!svg || !isInitialized) {
            console.warn('[DRXDAG] Cannot export - not initialized');
            return;
        }

        const svgElement = svg.node();
        const svgData = new XMLSerializer().serializeToString(svgElement);
        const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });

        const a = document.createElement('a');
        a.download = `drx-dag-${Date.now()}.svg`;
        a.href = URL.createObjectURL(blob);
        a.click();

        URL.revokeObjectURL(a.href);
    }

    // ==========================================================================
    // Utility Functions
    // ==========================================================================

    /**
     * Get current graph state
     */
    function getState() {
        return {
            nodes: nodes.map(n => ({ ...n })),
            edges: edges.map(e => ({ ...e })),
            isInitialized
        };
    }

    /**
     * Get node by ID
     */
    function getNode(id) {
        return nodeMap.get(id);
    }

    /**
     * Check if initialized
     */
    function isReady() {
        return isInitialized;
    }

    // ==========================================================================
    // Public API
    // ==========================================================================

    return {
        // Initialization
        init,
        destroy,
        isReady,

        // Node management
        addNode,
        addNodes,
        addEdge,
        updateNodeStatus,

        // SSE integration
        updateFromEvent,

        // View controls
        render,
        clear,
        resetView,
        fitToContainer,

        // Export
        exportPNG,
        exportSVG,

        // State access
        getState,
        getNode,

        // Constants
        NODE_STATES,
        CONFIG
    };
})();

// Make available globally
if (typeof window !== 'undefined') {
    window.DRXDAG = DRXDAG;
}

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DRXDAG;
}
