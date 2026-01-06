/**
 * DRX SSE Connection Manager
 * Handles Server-Sent Events with automatic reconnection and event buffering
 */

const DRXStream = (() => {
    const DEFAULT_BASE_URL = '/api/v1';
    const RECONNECT_DELAYS = [1000, 2000, 5000, 10000, 30000]; // Exponential backoff
    const MAX_RECONNECT_ATTEMPTS = 10;
    const HEARTBEAT_TIMEOUT = 60000; // 60 seconds

    class SSEConnection {
        constructor(interactionId, options = {}) {
            this.interactionId = interactionId;
            this.baseUrl = options.baseUrl || DEFAULT_BASE_URL;
            this.eventSource = null;
            this.reconnectAttempts = 0;
            this.lastEventId = null;
            this.isConnecting = false;
            this.isClosed = false;
            this.heartbeatTimer = null;
            this.reconnectTimer = null;

            // Event handlers
            this.handlers = {
                open: [],
                close: [],
                error: [],
                reconnect: [],
                message: [],
                // DRX-specific events
                'agent_start': [],
                'agent_end': [],
                'search_result': [],
                'content_chunk': [],
                'thinking': [],
                'progress': [],
                'iteration_start': [],
                'iteration_end': [],
                'report_chunk': [],
                'complete': [],
                'error_event': [],
                // DAG-specific events
                'workflow_state': [],
                'dag_state': [],
                'plan_created': [],
                'dag_created': [],
                'task_start': [],
                'task_complete': [],
                'task_end': [],
                'task_error': [],
                'task_failed': []
            };

            // Buffered events for replay
            this.eventBuffer = [];
            this.bufferSize = options.bufferSize || 1000;
        }

        /**
         * Connect to SSE stream
         */
        connect() {
            if (this.isClosed || this.isConnecting) return;

            this.isConnecting = true;
            this.updateConnectionState('connecting');

            const url = this.buildUrl();
            console.log(`[SSE] Connecting to ${url}`);

            try {
                this.eventSource = new EventSource(url);

                this.eventSource.onopen = () => {
                    console.log('[SSE] Connection opened');
                    this.isConnecting = false;
                    this.reconnectAttempts = 0;
                    this.updateConnectionState('connected');
                    this.startHeartbeat();
                    this.emit('open', { interactionId: this.interactionId });
                };

                this.eventSource.onerror = (error) => {
                    console.error('[SSE] Connection error:', error);
                    this.handleError(error);
                };

                this.eventSource.onmessage = (event) => {
                    this.handleMessage(event);
                };

                // Add specific event listeners for DRX events
                const eventTypes = [
                    'agent_start', 'agent_end', 'search_result', 'content_chunk',
                    'thinking', 'progress', 'iteration_start', 'iteration_end',
                    'report_chunk', 'complete', 'error',
                    // DAG-specific events
                    'workflow_state', 'dag_state', 'plan_created', 'dag_created',
                    'task_start', 'task_complete', 'task_end', 'task_error', 'task_failed'
                ];

                eventTypes.forEach(type => {
                    this.eventSource.addEventListener(type, (event) => {
                        this.handleTypedEvent(type, event);
                    });
                });

            } catch (error) {
                console.error('[SSE] Failed to create EventSource:', error);
                this.isConnecting = false;
                this.handleError(error);
            }
        }

        /**
         * Build SSE URL with last event ID for resumption
         */
        buildUrl() {
            const url = new URL(`${this.baseUrl}/interactions/${this.interactionId}/stream`, window.location.origin);

            if (this.lastEventId) {
                url.searchParams.set('last_event_id', this.lastEventId);
            }

            return url.toString();
        }

        /**
         * Handle incoming message
         */
        handleMessage(event) {
            this.resetHeartbeat();

            try {
                const data = JSON.parse(event.data);

                // Update last event ID
                if (event.lastEventId) {
                    this.lastEventId = event.lastEventId;
                } else if (data.event_id) {
                    this.lastEventId = data.event_id;
                }

                // Buffer event
                this.bufferEvent('message', data);

                // Emit to handlers
                this.emit('message', data);

            } catch (error) {
                console.warn('[SSE] Failed to parse message:', error, event.data);
            }
        }

        /**
         * Handle typed event
         */
        handleTypedEvent(type, event) {
            this.resetHeartbeat();

            try {
                const data = JSON.parse(event.data);

                // Update last event ID
                if (event.lastEventId) {
                    this.lastEventId = event.lastEventId;
                } else if (data.event_id) {
                    this.lastEventId = data.event_id;
                }

                // Buffer event
                this.bufferEvent(type, data);

                // Emit to type-specific handlers
                const eventName = type === 'error' ? 'error_event' : type;
                this.emit(eventName, data);

                // Also emit as generic message
                this.emit('message', { type, ...data });

                // Update DAG visualization for relevant events
                this.updateDAGVisualization(type, data);

                // Handle completion
                if (type === 'complete') {
                    this.handleComplete(data);
                }

            } catch (error) {
                console.warn(`[SSE] Failed to parse ${type} event:`, error, event.data);
            }
        }

        /**
         * Update DAG visualization based on event type
         */
        updateDAGVisualization(type, data) {
            // Check if DRXDAG is available
            if (typeof window !== 'undefined' && window.DRXDAG && window.DRXDAG.isReady()) {
                // List of event types that should update the DAG
                const dagEventTypes = [
                    'workflow_state', 'dag_state', 'plan_created', 'dag_created',
                    'agent_start', 'agent_end', 'agent_complete',
                    'task_start', 'task_complete', 'task_end', 'task_error', 'task_failed',
                    'iteration_start', 'iteration_end'
                ];

                if (dagEventTypes.includes(type)) {
                    try {
                        // Add event type to data for proper handling
                        const eventData = { ...data, type, event_type: type };
                        window.DRXDAG.updateFromEvent(eventData);
                    } catch (err) {
                        console.warn('[SSE] Failed to update DAG visualization:', err);
                    }
                }
            }
        }

        /**
         * Handle connection error
         */
        handleError(error) {
            this.isConnecting = false;
            this.stopHeartbeat();

            // Close existing connection
            if (this.eventSource) {
                this.eventSource.close();
                this.eventSource = null;
            }

            // Check if we should reconnect
            if (!this.isClosed && this.reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                this.scheduleReconnect();
            } else {
                this.updateConnectionState('disconnected');
                this.emit('error', { error, permanent: true });
            }
        }

        /**
         * Handle stream completion
         */
        handleComplete(data) {
            console.log('[SSE] Stream completed');
            this.stopHeartbeat();
            this.updateConnectionState('completed');
            this.close();
        }

        /**
         * Schedule reconnection attempt
         */
        scheduleReconnect() {
            const delayIndex = Math.min(this.reconnectAttempts, RECONNECT_DELAYS.length - 1);
            const delay = RECONNECT_DELAYS[delayIndex];

            console.log(`[SSE] Scheduling reconnect in ${delay}ms (attempt ${this.reconnectAttempts + 1})`);

            this.updateConnectionState('reconnecting');
            this.emit('reconnect', {
                attempt: this.reconnectAttempts + 1,
                delay,
                maxAttempts: MAX_RECONNECT_ATTEMPTS
            });

            this.reconnectTimer = setTimeout(() => {
                this.reconnectAttempts++;
                this.connect();
            }, delay);
        }

        /**
         * Start heartbeat monitoring
         */
        startHeartbeat() {
            this.stopHeartbeat();
            this.heartbeatTimer = setTimeout(() => {
                console.warn('[SSE] Heartbeat timeout, reconnecting...');
                this.handleError(new Error('Heartbeat timeout'));
            }, HEARTBEAT_TIMEOUT);
        }

        /**
         * Reset heartbeat timer
         */
        resetHeartbeat() {
            if (this.heartbeatTimer) {
                this.startHeartbeat();
            }
        }

        /**
         * Stop heartbeat monitoring
         */
        stopHeartbeat() {
            if (this.heartbeatTimer) {
                clearTimeout(this.heartbeatTimer);
                this.heartbeatTimer = null;
            }
        }

        /**
         * Buffer event for replay
         */
        bufferEvent(type, data) {
            this.eventBuffer.push({
                type,
                data,
                timestamp: Date.now()
            });

            // Trim buffer if needed
            if (this.eventBuffer.length > this.bufferSize) {
                this.eventBuffer.shift();
            }
        }

        /**
         * Get buffered events
         */
        getBufferedEvents(since = 0) {
            return this.eventBuffer.filter(e => e.timestamp > since);
        }

        /**
         * Update connection state
         */
        updateConnectionState(status) {
            if (typeof DRXState !== 'undefined') {
                DRXState.set('connection.status', status);
                if (status === 'connected') {
                    DRXState.set('connection.lastConnected', Date.now());
                    DRXState.set('connection.reconnectAttempts', 0);
                } else if (status === 'reconnecting') {
                    DRXState.set('connection.reconnectAttempts', this.reconnectAttempts);
                }
            }
        }

        /**
         * Register event handler
         */
        on(event, handler) {
            if (this.handlers[event]) {
                this.handlers[event].push(handler);
            }
            return () => this.off(event, handler);
        }

        /**
         * Remove event handler
         */
        off(event, handler) {
            if (this.handlers[event]) {
                this.handlers[event] = this.handlers[event].filter(h => h !== handler);
            }
        }

        /**
         * Emit event to handlers
         */
        emit(event, data) {
            if (this.handlers[event]) {
                for (const handler of this.handlers[event]) {
                    try {
                        handler(data);
                    } catch (error) {
                        console.error(`[SSE] Handler error for ${event}:`, error);
                    }
                }
            }
        }

        /**
         * Close connection
         */
        close() {
            this.isClosed = true;
            this.stopHeartbeat();

            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
            }

            if (this.eventSource) {
                this.eventSource.close();
                this.eventSource = null;
            }

            this.updateConnectionState('disconnected');
            this.emit('close', { interactionId: this.interactionId });
        }

        /**
         * Check if connection is active
         */
        isActive() {
            return this.eventSource && this.eventSource.readyState === EventSource.OPEN;
        }

        /**
         * Get connection status
         */
        getStatus() {
            if (this.isClosed) return 'closed';
            if (this.isConnecting) return 'connecting';
            if (this.eventSource) {
                switch (this.eventSource.readyState) {
                    case EventSource.CONNECTING: return 'connecting';
                    case EventSource.OPEN: return 'connected';
                    case EventSource.CLOSED: return 'disconnected';
                }
            }
            return 'disconnected';
        }
    }

    // Active connections map
    const connections = new Map();

    /**
     * Create new SSE connection for interaction
     */
    function connect(interactionId, options = {}) {
        // Close existing connection if any
        if (connections.has(interactionId)) {
            connections.get(interactionId).close();
        }

        const connection = new SSEConnection(interactionId, options);
        connections.set(interactionId, connection);
        connection.connect();

        return connection;
    }

    /**
     * Get existing connection
     */
    function get(interactionId) {
        return connections.get(interactionId);
    }

    /**
     * Close specific connection
     */
    function close(interactionId) {
        const connection = connections.get(interactionId);
        if (connection) {
            connection.close();
            connections.delete(interactionId);
        }
    }

    /**
     * Close all connections
     */
    function closeAll() {
        for (const connection of connections.values()) {
            connection.close();
        }
        connections.clear();
    }

    /**
     * Get all active connections
     */
    function getAll() {
        return Array.from(connections.entries()).map(([id, conn]) => ({
            interactionId: id,
            status: conn.getStatus(),
            eventCount: conn.eventBuffer.length
        }));
    }

    return {
        connect,
        get,
        close,
        closeAll,
        getAll,
        SSEConnection
    };
})();

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DRXStream;
}
