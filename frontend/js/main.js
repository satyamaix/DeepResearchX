/**
 * DRX Main Entry Point
 * Bridges HTML with module-based architecture
 */

// Global state
let currentResearch = null;
let sseConnection = null;
let reportContent = '';
let startTime = null;
let progressTimer = null;
let currentKnowledgeGraph = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);

/**
 * Initialize the application
 */
function init() {
    console.log('[DRX] Initializing...');

    // Load saved settings
    loadSettings();

    // Set up event listeners
    setupEventListeners();

    // Check API health
    checkHealth();

    // Initialize theme
    initTheme();

    console.log('[DRX] Ready');
}

/**
 * Load settings from localStorage
 */
function loadSettings() {
    const settings = DRXState.get();

    // Model selects
    const defaultModel = document.getElementById('default-model');
    if (defaultModel && settings.model) {
        defaultModel.value = settings.model;
    }

    // Sliders
    const maxIterations = document.getElementById('max-iterations');
    if (maxIterations && settings.steerability?.maxIterations) {
        maxIterations.value = settings.steerability.maxIterations;
        document.getElementById('max-iterations-value').textContent = maxIterations.value;
    }

    const maxSources = document.getElementById('max-sources');
    if (maxSources && settings.researchParams?.maxSources) {
        maxSources.value = settings.researchParams.maxSources;
        document.getElementById('max-sources-value').textContent = maxSources.value;
    }

    // Load history
    updateHistoryList();
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Query input
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const charCount = document.getElementById('char-count');

    if (queryInput) {
        queryInput.addEventListener('input', () => {
            // Auto-resize
            queryInput.style.height = 'auto';
            queryInput.style.height = Math.min(queryInput.scrollHeight, 200) + 'px';

            // Update char count
            const len = queryInput.value.length;
            charCount.textContent = `${len} / 10000`;

            // Enable/disable send button
            sendBtn.disabled = len < 10;
        });

        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuery();
            }
        });
    }

    // Slider value displays
    setupSlider('max-iterations', 'max-iterations-value', v => v);
    setupSlider('max-sources', 'max-sources-value', v => v);
    setupSlider('token-budget', 'token-budget-value', v => (v / 1000) + 'K');
    setupSlider('timeout', 'timeout-value', v => Math.floor(v / 60) + ' min');

    // Theme toggle
    document.getElementById('dark-mode')?.addEventListener('change', (e) => {
        const theme = e.target.checked ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', theme);
        DRXState.set('theme', theme);
    });

    // Model selection
    document.getElementById('default-model')?.addEventListener('change', (e) => {
        DRXState.set('model', e.target.value);
    });

    // Tag inputs
    setupTagInput('focus-input', 'focus-tags', 'researchParams.focusAreas', 10);
    setupTagInput('exclude-input', 'exclude-tags', 'researchParams.excludeTopics', 10);
    setupTagInput('domain-input', 'domain-tags', 'researchParams.preferredDomains', 20);

    // Custom instructions
    const instructions = document.getElementById('custom-instructions');
    const instructionsCount = document.getElementById('instructions-count');
    if (instructions && instructionsCount) {
        instructions.addEventListener('input', () => {
            instructionsCount.textContent = instructions.value.length;
            DRXState.set('customInstructions', instructions.value);
        });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
}

/**
 * Set up slider with value display
 */
function setupSlider(sliderId, valueId, formatter) {
    const slider = document.getElementById(sliderId);
    const valueEl = document.getElementById(valueId);

    if (slider && valueEl) {
        slider.addEventListener('input', () => {
            valueEl.textContent = formatter(parseInt(slider.value));
        });
    }
}

/**
 * Set up tag input
 */
function setupTagInput(inputId, containerId, statePath, maxTags) {
    const input = document.getElementById(inputId);
    const container = document.getElementById(containerId);

    if (!input || !container) return;

    // Load existing tags
    const existing = DRXState.get(statePath) || [];
    existing.forEach(tag => addTag(container, tag, statePath));

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && input.value.trim()) {
            e.preventDefault();
            const tags = DRXState.get(statePath) || [];
            if (tags.length < maxTags) {
                const value = input.value.trim();
                if (!tags.includes(value)) {
                    addTag(container, value, statePath);
                    DRXState.set(statePath, [...tags, value]);
                }
            }
            input.value = '';
        }
    });
}

/**
 * Add tag element
 */
function addTag(container, value, statePath) {
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.innerHTML = `${DRXMarkdown.escapeHtml(value)}<span class="tag-remove" onclick="removeTag(this, '${statePath}')">&times;</span>`;
    tag.dataset.value = value;
    container.appendChild(tag);
}

/**
 * Remove tag
 */
function removeTag(removeBtn, statePath) {
    const tag = removeBtn.parentElement;
    const value = tag.dataset.value;
    tag.remove();

    const tags = DRXState.get(statePath) || [];
    DRXState.set(statePath, tags.filter(t => t !== value));
}

/**
 * Check API health
 */
async function checkHealth() {
    try {
        await DRXApi.getHealth();
        updateConnectionStatus('connected');
    } catch (error) {
        console.error('[DRX] Health check failed:', error);
        updateConnectionStatus('disconnected');
        showToast('Unable to connect to API server', 'error');
    }
}

/**
 * Initialize theme
 */
function initTheme() {
    const theme = DRXState.get('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    const darkModeToggle = document.getElementById('dark-mode');
    if (darkModeToggle) {
        darkModeToggle.checked = theme === 'dark';
    }
}

/**
 * Toggle sidebar
 */
function toggleSidebar(side) {
    const sidebar = document.getElementById(`sidebar-${side}`);
    const headerBtn = document.getElementById(`toggle-${side}-sidebar`);

    if (sidebar) {
        const isCollapsing = !sidebar.classList.contains('collapsed');
        sidebar.classList.toggle('collapsed');

        // Show/hide header button based on sidebar state
        if (headerBtn) {
            if (isCollapsing) {
                headerBtn.classList.add('visible');
            } else {
                headerBtn.classList.remove('visible');
            }
        }
    }
}

/**
 * Set query from example
 */
function setQuery(text) {
    const input = document.getElementById('query-input');
    if (input) {
        input.value = text;
        input.dispatchEvent(new Event('input'));
        input.focus();
    }
}

/**
 * Send research query
 */
async function sendQuery() {
    const input = document.getElementById('query-input');
    const query = input?.value?.trim();

    if (!query || query.length < 10) {
        showToast('Please enter at least 10 characters', 'warning');
        return;
    }

    if (currentResearch) {
        showToast('Research already in progress', 'warning');
        return;
    }

    console.log('[DRX] Starting research:', query);

    // Add user message
    addMessage('user', query);

    // Clear input
    input.value = '';
    input.style.height = 'auto';
    document.getElementById('send-btn').disabled = true;
    document.getElementById('char-count').textContent = '0 / 10000';

    // Show progress
    showProgress('Initializing research...', 0);
    startTime = Date.now();
    startProgressTimer();

    // Initialize DAG visualization
    initDAGVisualization();

    try {
        // Get settings
        const settings = DRXState.get();

        // Create interaction
        const response = await DRXApi.createInteraction(query, {
            model: settings.model,
            maxIterations: parseInt(document.getElementById('max-iterations')?.value) || 5,
            maxSources: parseInt(document.getElementById('max-sources')?.value) || 20,
            tokenBudget: parseInt(document.getElementById('token-budget')?.value) || 500000,
            timeout: parseInt(document.getElementById('timeout')?.value) || 600,
            tone: document.getElementById('tone')?.value || 'technical',
            format: document.getElementById('format')?.value || 'markdown',
            language: document.getElementById('language')?.value || 'en',
            focusAreas: settings.researchParams?.focusAreas || [],
            excludeTopics: settings.researchParams?.excludeTopics || [],
            preferredDomains: settings.researchParams?.preferredDomains || [],
            customInstructions: settings.customInstructions || ''
        });

        currentResearch = {
            id: response.interaction_id,
            query,
            iteration: 0,
            maxIterations: parseInt(document.getElementById('max-iterations')?.value) || 5
        };

        console.log('[DRX] Created interaction:', currentResearch.id);

        // Connect to SSE stream
        connectToStream(currentResearch.id);

    } catch (error) {
        console.error('[DRX] Failed to start research:', error);
        handleError(error);
    }
}

/**
 * Connect to SSE stream
 */
function connectToStream(interactionId) {
    sseConnection = DRXStream.connect(interactionId);

    sseConnection.on('open', () => {
        console.log('[DRX] SSE connected');
        updateConnectionStatus('connected');
    });

    sseConnection.on('error', (data) => {
        if (data.permanent) {
            handleError(new Error('Connection lost'));
        }
    });

    sseConnection.on('reconnect', (data) => {
        showToast(`Reconnecting... (${data.attempt}/${data.maxAttempts})`, 'warning');
        updateConnectionStatus('reconnecting');
    });

    // Agent events
    sseConnection.on('agent_start', handleAgentStart);
    sseConnection.on('agent_end', handleAgentEnd);
    sseConnection.on('search_result', handleSearchResult);

    // Progress events
    sseConnection.on('progress', handleProgressEvent);
    sseConnection.on('iteration_start', handleIterationStart);
    sseConnection.on('iteration_end', handleIterationEnd);

    // Content events
    sseConnection.on('content_chunk', handleContentChunk);
    sseConnection.on('report_chunk', handleReportChunk);
    sseConnection.on('thinking', handleThinking);

    // Completion
    sseConnection.on('complete', handleComplete);
    sseConnection.on('error_event', handleErrorEvent);

    // DAG events (for workflow plan updates)
    sseConnection.on('plan_created', handlePlanCreated);
    sseConnection.on('dag_created', handlePlanCreated);
    sseConnection.on('workflow_state', handleWorkflowState);
    sseConnection.on('dag_state', handleWorkflowState);
}

/**
 * Handle plan created event - update DAG with actual workflow
 */
function handlePlanCreated(data) {
    console.log('[DRX] Plan created:', data);

    if (typeof DRXDAG === 'undefined' || !DRXDAG.isReady()) return;

    const plan = data.plan || data;
    const dagNodes = plan.dag_nodes || plan.tasks || [];

    if (dagNodes.length > 0) {
        // Clear existing nodes and rebuild with actual plan
        DRXDAG.clear();
        DRXDAG.addNodes(dagNodes);
        DRXDAG.fitToContainer();
    }
}

/**
 * Handle workflow state update
 */
function handleWorkflowState(data) {
    console.log('[DRX] Workflow state update:', data);

    if (typeof DRXDAG === 'undefined' || !DRXDAG.isReady()) return;

    // Let DRXDAG handle the full state update
    DRXDAG.updateFromEvent(data);
}

/**
 * Initialize DAG visualization
 */
function initDAGVisualization() {
    // Show DAG section
    const dagSection = document.getElementById('dag-section');
    if (dagSection) {
        dagSection.style.display = 'block';
    }

    // Initialize DRXDAG if available
    if (typeof DRXDAG !== 'undefined') {
        const success = DRXDAG.init('dag-container');
        if (success) {
            console.log('[DRX] DAG visualization initialized');

            // Add initial nodes for the research workflow
            addInitialDAGNodes();
        } else {
            console.warn('[DRX] Failed to initialize DAG visualization');
        }
    } else {
        console.warn('[DRX] DRXDAG module not available');
    }
}

/**
 * Add initial DAG nodes representing the research workflow
 */
function addInitialDAGNodes() {
    if (typeof DRXDAG === 'undefined' || !DRXDAG.isReady()) return;

    // Create a basic workflow structure
    // This will be replaced when the actual plan is received from the backend
    const initialNodes = [
        {
            id: 'planner',
            label: 'Planning',
            type: 'planner',
            status: 'pending',
            dependencies: []
        },
        {
            id: 'searcher',
            label: 'Searching',
            type: 'searcher',
            status: 'pending',
            dependencies: ['planner']
        },
        {
            id: 'reader',
            label: 'Reading',
            type: 'reader',
            status: 'pending',
            dependencies: ['searcher']
        },
        {
            id: 'synthesizer',
            label: 'Synthesizing',
            type: 'synthesizer',
            status: 'pending',
            dependencies: ['reader']
        },
        {
            id: 'critic',
            label: 'Reviewing',
            type: 'critic',
            status: 'pending',
            dependencies: ['synthesizer']
        },
        {
            id: 'reporter',
            label: 'Reporting',
            type: 'reporter',
            status: 'pending',
            dependencies: ['critic']
        }
    ];

    DRXDAG.addNodes(initialNodes);
    DRXDAG.fitToContainer();
}

/**
 * Hide DAG visualization
 */
function hideDAGVisualization() {
    const dagSection = document.getElementById('dag-section');
    if (dagSection) {
        dagSection.style.display = 'none';
    }

    if (typeof DRXDAG !== 'undefined' && DRXDAG.isReady()) {
        DRXDAG.clear();
    }
}

/**
 * Handle agent start
 */
function handleAgentStart(data) {
    const agent = data.agent || data.node || 'Agent';
    console.log('[DRX] Agent started:', agent);

    updateAgentStatus(agent, 'active');
    showProgress(`${agent} working...`);
    addAgentActivity(agent, 'started');

    // Update DAG visualization
    updateDAGForAgentEvent(agent, 'running');
}

/**
 * Handle agent end
 */
function handleAgentEnd(data) {
    const agent = data.agent || data.node || 'Agent';
    console.log('[DRX] Agent completed:', agent);

    updateAgentStatus(agent, 'done');
    addAgentActivity(agent, 'completed', data.summary);

    // Update DAG visualization
    updateDAGForAgentEvent(agent, 'completed');
}

/**
 * Update DAG visualization for agent events
 */
function updateDAGForAgentEvent(agent, status) {
    if (typeof DRXDAG === 'undefined' || !DRXDAG.isReady()) return;

    // Map agent name to node ID
    const agentLower = agent.toLowerCase();
    const nodeId = agentLower; // Node IDs match agent names

    // Check if this is a known agent type
    const validAgents = ['planner', 'searcher', 'reader', 'synthesizer', 'critic', 'reporter'];

    if (validAgents.includes(agentLower)) {
        DRXDAG.updateNodeStatus(nodeId, status);
    }
}

/**
 * Mark all DAG nodes as completed (for final state)
 */
function completeAllDAGNodes() {
    if (typeof DRXDAG === 'undefined' || !DRXDAG.isReady()) return;

    const state = DRXDAG.getState();
    if (state && state.nodes) {
        state.nodes.forEach(node => {
            if (node.status !== 'completed' && node.status !== 'failed') {
                DRXDAG.updateNodeStatus(node.id, 'completed');
            }
        });
    }
}

/**
 * Handle search results
 */
function handleSearchResult(data) {
    if (data.results?.length) {
        addAgentActivity('Searcher', `found ${data.results.length} results`);
    }
}

/**
 * Handle progress event
 */
function handleProgressEvent(data) {
    const percent = data.percent || 0;
    const message = data.message || 'Processing...';
    showProgress(message, percent);
}

/**
 * Handle iteration start
 */
function handleIterationStart(data) {
    if (currentResearch) {
        currentResearch.iteration = data.iteration || currentResearch.iteration + 1;
    }
    const iter = currentResearch?.iteration || 1;
    const max = currentResearch?.maxIterations || 5;

    document.getElementById('progress-iteration').textContent = `Iteration ${iter}/${max}`;
    showProgress('Starting iteration...', (iter - 1) / max * 100);
}

/**
 * Handle iteration end
 */
function handleIterationEnd(data) {
    if (data.gaps_found) {
        addAgentActivity('Critic', `found ${data.gaps_found} gaps, continuing...`);
    }
}

/**
 * Handle content chunk (streaming)
 */
function handleContentChunk(data) {
    if (data.content) {
        reportContent += data.content;
        updateStreamingReport();
    }
}

/**
 * Handle report chunk
 */
function handleReportChunk(data) {
    const chunk = data.content || data.chunk || '';
    if (chunk) {
        reportContent += chunk;
        updateStreamingReport();
    }
}

/**
 * Handle thinking event
 */
function handleThinking(data) {
    if (data.thinking && document.getElementById('thinking-summaries')?.checked) {
        // Could show thinking in a collapsible section
        console.log('[DRX] Thinking:', data.thinking.substring(0, 100) + '...');
    }
}

/**
 * Handle completion
 */
function handleComplete(data) {
    console.log('[DRX] Research complete');

    stopProgressTimer();

    // Mark all remaining DAG nodes as completed
    completeAllDAGNodes();

    // Use final report if provided
    if (data.report) {
        reportContent = data.report;
    }

    // Store knowledge graph data if provided
    if (data.knowledge_graph) {
        currentKnowledgeGraph = data.knowledge_graph;
        console.log('[DRX] Knowledge graph received with',
            (data.knowledge_graph.nodes || []).length, 'nodes and',
            (data.knowledge_graph.edges || []).length, 'edges');
    }

    // Hide progress (but keep DAG visible for a bit)
    hideProgress();

    // Update final message
    if (reportContent) {
        updateStreamingReport(true);

        // Add view full report button
        const viewBtn = document.createElement('button');
        viewBtn.className = 'btn btn-primary view-report-btn';
        viewBtn.textContent = 'View Full Report';
        viewBtn.onclick = () => showReportModal();

        const lastMessage = document.querySelector('.message.assistant:last-child .message-content');
        if (lastMessage) {
            lastMessage.appendChild(viewBtn);
        }
    }

    // Save to history (including knowledge graph if available)
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    DRXState.addToHistory({
        id: currentResearch?.id || Date.now().toString(),
        query: currentResearch?.query,
        report: reportContent,
        knowledge_graph: currentKnowledgeGraph || null,
        status: 'completed',
        duration,
        timestamp: new Date().toISOString()
    });
    updateHistoryList();

    // Cleanup
    currentResearch = null;
    if (sseConnection) {
        sseConnection.close();
        sseConnection = null;
    }

    updateConnectionStatus('completed');
    showToast(`Research completed in ${duration}s`, 'success');
}

/**
 * Handle error event
 */
function handleErrorEvent(data) {
    handleError(new Error(data.message || data.error || 'Unknown error'));
}

/**
 * Handle errors
 */
function handleError(error) {
    console.error('[DRX] Error:', error);

    stopProgressTimer();
    hideProgress();

    const message = error.body?.detail || error.message || 'An error occurred';
    addMessage('system', `**Error:** ${message}\n\nPlease try again.`);
    showToast(message, 'error');

    // Save to history
    if (currentResearch) {
        DRXState.addToHistory({
            id: currentResearch.id || Date.now().toString(),
            query: currentResearch.query,
            status: 'error',
            error: message,
            timestamp: new Date().toISOString()
        });
        updateHistoryList();
    }

    // Cleanup
    currentResearch = null;
    reportContent = '';
    if (sseConnection) {
        sseConnection.close();
        sseConnection = null;
    }

    updateConnectionStatus('disconnected');
}

/**
 * Add message to chat
 */
let streamingMessageEl = null;

function addMessage(type, content) {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    const message = document.createElement('div');
    message.className = `message ${type}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (type === 'assistant' || type === 'system') {
        contentDiv.innerHTML = DRXMarkdown.render(content);
    } else {
        contentDiv.textContent = content;
    }

    message.appendChild(contentDiv);
    container.appendChild(message);

    // Scroll to bottom
    container.scrollTop = container.scrollHeight;

    return message;
}

/**
 * Update streaming report
 */
function updateStreamingReport(isFinal = false) {
    const container = document.getElementById('chat-messages');

    if (!streamingMessageEl) {
        streamingMessageEl = document.createElement('div');
        streamingMessageEl.className = 'message assistant';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content streaming';
        streamingMessageEl.appendChild(contentDiv);

        container.appendChild(streamingMessageEl);
    }

    const contentDiv = streamingMessageEl.querySelector('.message-content');
    contentDiv.innerHTML = DRXMarkdown.render(reportContent);

    if (isFinal) {
        contentDiv.classList.remove('streaming');
        streamingMessageEl = null;
    }

    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
}

/**
 * Show progress
 */
function showProgress(phase, percent = null) {
    const container = document.getElementById('progress-container');
    const phaseEl = document.getElementById('progress-phase');
    const bar = document.getElementById('progress-bar');

    if (container) {
        container.style.display = 'block';
    }

    if (phaseEl) {
        phaseEl.textContent = phase;
    }

    if (bar && percent !== null) {
        bar.style.width = `${Math.min(percent, 100)}%`;
    }
}

/**
 * Hide progress
 */
function hideProgress() {
    const container = document.getElementById('progress-container');
    if (container) {
        container.style.display = 'none';
    }
}

/**
 * Start progress timer
 */
function startProgressTimer() {
    const timeEl = document.getElementById('progress-time');
    if (!timeEl) return;

    progressTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        timeEl.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }, 1000);
}

/**
 * Stop progress timer
 */
function stopProgressTimer() {
    if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
    }
}

/**
 * Add agent activity
 */
function addAgentActivity(agent, action, details = '') {
    const container = document.getElementById('agent-activity');
    if (!container) return;

    const activity = document.createElement('div');
    activity.className = 'activity-item';
    activity.innerHTML = `
        <span class="activity-agent">${agent}</span>
        <span class="activity-action">${action}</span>
        ${details ? `<span class="activity-details">${details}</span>` : ''}
    `;

    container.appendChild(activity);

    // Keep only last 5 activities
    while (container.children.length > 5) {
        container.removeChild(container.firstChild);
    }
}

/**
 * Update agent status indicator
 */
function updateAgentStatus(agent, status) {
    const statusEl = document.getElementById(`agent-${agent.toLowerCase()}-status`);
    if (statusEl) {
        statusEl.className = `agent-status ${status}`;
        statusEl.textContent = status === 'active' ? '●' : status === 'done' ? '✓' : '';
    }
}

/**
 * Update connection status
 */
function updateConnectionStatus(status) {
    const statusEl = document.getElementById('connection-status');
    if (!statusEl) return;

    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('.status-text');

    if (dot) {
        dot.className = `status-dot ${status}`;
    }

    if (text) {
        const labels = {
            connected: 'Connected',
            connecting: 'Connecting...',
            reconnecting: 'Reconnecting...',
            disconnected: 'Disconnected',
            completed: 'Completed'
        };
        text.textContent = labels[status] || status;
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-message">${message}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">&times;</button>
    `;

    container.appendChild(toast);

    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

/**
 * Show report modal
 */
function showReportModal() {
    const modal = document.getElementById('report-modal');
    const content = document.getElementById('report-content');
    const meta = document.getElementById('report-meta');

    if (modal && content) {
        content.innerHTML = DRXMarkdown.render(reportContent);
        modal.classList.add('active');

        if (meta && currentResearch) {
            const duration = ((Date.now() - startTime) / 1000).toFixed(1);
            meta.textContent = `Duration: ${duration}s | Iterations: ${currentResearch?.iteration || '?'}`;
        }

        // Initialize knowledge graph if data is available
        initKnowledgeGraph();
    }
}

/**
 * Initialize knowledge graph visualization in the modal
 */
function initKnowledgeGraph() {
    const graphSection = document.getElementById('graph-section');
    const graphContainer = document.getElementById('graph-container');

    if (!graphSection || !graphContainer) return;

    // Check if we have knowledge graph data
    if (currentKnowledgeGraph &&
        ((currentKnowledgeGraph.nodes && currentKnowledgeGraph.nodes.length > 0) ||
         (currentKnowledgeGraph.elements && currentKnowledgeGraph.elements.nodes))) {

        graphSection.style.display = 'block';

        // Initialize DRXGraph if available
        if (typeof DRXGraph !== 'undefined') {
            // Small delay to ensure modal is visible and container has dimensions
            setTimeout(() => {
                DRXGraph.init('graph-container');
                DRXGraph.loadFromCytoscape(currentKnowledgeGraph);
                console.log('[DRX] Knowledge graph visualization initialized');
            }, 100);
        } else {
            console.warn('[DRX] DRXGraph module not available');
        }
    } else {
        // Hide graph section if no data
        graphSection.style.display = 'none';
    }
}

/**
 * Handle graph filter button click
 * @param {HTMLElement} btn - The clicked filter button
 */
function handleGraphFilter(btn) {
    const type = btn.dataset.type || '';

    // Update active state on buttons
    const buttons = document.querySelectorAll('.graph-filter-btn[data-type]');
    buttons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    // Apply filter
    if (typeof DRXGraph !== 'undefined') {
        DRXGraph.filterByType(type ? [type] : []);
    }
}

/**
 * Close modal
 */
function closeModal() {
    const modal = document.getElementById('report-modal');
    if (modal) {
        modal.classList.remove('active');
    }

    // Clean up graph when modal closes
    if (typeof DRXGraph !== 'undefined') {
        DRXGraph.hideDetailPanel();
    }
}

/**
 * Copy report to clipboard
 */
async function copyReport() {
    try {
        await navigator.clipboard.writeText(reportContent);
        showToast('Report copied to clipboard', 'success');
    } catch (error) {
        showToast('Failed to copy report', 'error');
    }
}

/**
 * Download report
 */
function downloadReport() {
    const blob = new Blob([reportContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `drx-report-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Report downloaded', 'success');
}

/**
 * Update history list
 */
function updateHistoryList() {
    const container = document.getElementById('history-list');
    if (!container) return;

    const history = DRXState.getHistory();

    if (history.length === 0) {
        container.innerHTML = '<p class="history-empty">No recent research</p>';
        return;
    }

    container.innerHTML = history.slice(0, 10).map(item => `
        <div class="history-item" onclick="loadHistoryItem('${item.id}')">
            <div class="history-query">${DRXMarkdown.escapeHtml((item.query || '').substring(0, 60))}${(item.query || '').length > 60 ? '...' : ''}</div>
            <div class="history-meta">
                <span class="history-status ${item.status}">${item.status}</span>
                <span class="history-time">${formatRelativeTime(item.timestamp)}</span>
            </div>
        </div>
    `).join('');
}

/**
 * Load history item
 */
function loadHistoryItem(id) {
    const history = DRXState.getHistory();
    const item = history.find(h => h.id === id);

    if (item?.report) {
        reportContent = item.report;
        // Load knowledge graph from history if available
        currentKnowledgeGraph = item.knowledge_graph || null;
        showReportModal();
    }
}

/**
 * Format relative time
 */
function formatRelativeTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;

    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
}
