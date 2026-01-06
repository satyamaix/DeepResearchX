/**
 * DRX Connection Manager
 * Handles network monitoring, visibility changes, and connection quality
 */

const DRXConnection = (() => {
    let isOnline = navigator.onLine;
    let isVisible = document.visibilityState === 'visible';
    let lastPingTime = 0;
    let pingInterval = null;
    let quality = 'unknown';

    const handlers = {
        online: [],
        offline: [],
        visibilityChange: [],
        qualityChange: [],
        pendingInteractions: [],
    };

    // Network monitoring
    function setupNetworkMonitoring() {
        window.addEventListener('online', () => {
            isOnline = true;
            emit('online', { timestamp: Date.now() });
            updateQuality();
        });

        window.addEventListener('offline', () => {
            isOnline = false;
            quality = 'offline';
            emit('offline', { timestamp: Date.now() });
            emit('qualityChange', { quality: 'offline' });
        });
    }

    // Visibility monitoring
    function setupVisibilityMonitoring() {
        document.addEventListener('visibilitychange', () => {
            isVisible = document.visibilityState === 'visible';
            emit('visibilityChange', { visible: isVisible, timestamp: Date.now() });

            if (isVisible) {
                // Tab became visible - check for stale connections
                checkConnections();
            }
        });
    }

    // Connection quality assessment
    function updateQuality() {
        if (!isOnline) {
            quality = 'offline';
            return;
        }

        // Use Network Information API if available
        const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
        if (connection) {
            const effectiveType = connection.effectiveType;
            switch (effectiveType) {
                case '4g':
                    quality = 'excellent';
                    break;
                case '3g':
                    quality = 'good';
                    break;
                case '2g':
                case 'slow-2g':
                    quality = 'poor';
                    break;
                default:
                    quality = 'unknown';
            }
        } else {
            // Fallback: use ping-based quality detection
            quality = isOnline ? 'good' : 'offline';
        }

        emit('qualityChange', { quality });
    }

    // Check and recover stale connections
    function checkConnections() {
        if (typeof DRXPersistence !== 'undefined') {
            const pending = DRXPersistence.getPendingInteractions();
            if (pending.length > 0) {
                console.log('[Connection] Found pending interactions:', pending);
                // Emit event for UI to handle recovery prompt
                emit('pendingInteractions', { interactions: pending });
            }
        }

        // Also check active SSE connections via DRXStream
        if (typeof DRXStream !== 'undefined') {
            const connections = DRXStream.getAll();
            connections.forEach(conn => {
                if (conn.status === 'disconnected' || conn.status === 'closed') {
                    console.log(`[Connection] Found stale connection: ${conn.interactionId}`);
                }
            });
        }
    }

    // Recover a specific interaction
    async function recoverInteraction(interactionId, lastEventId) {
        if (typeof DRXStream !== 'undefined') {
            console.log(`[Connection] Recovering interaction ${interactionId} from event ${lastEventId}`);
            const connection = DRXStream.connect(interactionId, {
                lastEventId
            });
            return connection;
        }
        return null;
    }

    // Start ping monitoring
    function startPingMonitor(interval = 30000) {
        if (pingInterval) {
            clearInterval(pingInterval);
        }

        pingInterval = setInterval(async () => {
            if (!isOnline || !isVisible) return;

            try {
                const start = Date.now();
                const response = await fetch('/api/v1/health', {
                    method: 'GET',
                    cache: 'no-store'
                });

                if (response.ok) {
                    lastPingTime = Date.now() - start;

                    // Update quality based on latency
                    let newQuality;
                    if (lastPingTime < 100) {
                        newQuality = 'excellent';
                    } else if (lastPingTime < 300) {
                        newQuality = 'good';
                    } else {
                        newQuality = 'poor';
                    }

                    if (newQuality !== quality) {
                        quality = newQuality;
                        emit('qualityChange', { quality, latency: lastPingTime });
                    }
                }
            } catch (e) {
                console.warn('[Connection] Ping failed:', e);
                if (isOnline) {
                    quality = 'poor';
                    emit('qualityChange', { quality: 'poor', error: e.message });
                }
            }
        }, interval);
    }

    function stopPingMonitor() {
        if (pingInterval) {
            clearInterval(pingInterval);
            pingInterval = null;
        }
    }

    // Event handling
    function on(event, handler) {
        if (handlers[event]) {
            handlers[event].push(handler);
        }
        return () => off(event, handler);
    }

    function off(event, handler) {
        if (handlers[event]) {
            handlers[event] = handlers[event].filter(h => h !== handler);
        }
    }

    function emit(event, data) {
        if (handlers[event]) {
            for (const handler of handlers[event]) {
                try {
                    handler(data);
                } catch (e) {
                    console.error(`[Connection] Handler error for ${event}:`, e);
                }
            }
        }
    }

    // Get current status
    function getStatus() {
        return {
            online: isOnline,
            visible: isVisible,
            quality,
            lastPingTime,
        };
    }

    // Update UI connection indicator
    function updateIndicator() {
        const indicator = document.getElementById('connection-status');
        if (!indicator) return;

        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('.status-text');

        if (dot) {
            dot.className = 'status-dot';
            if (!isOnline) {
                dot.classList.add('disconnected');
            } else {
                switch (quality) {
                    case 'excellent':
                    case 'good':
                        dot.classList.add('connected');
                        break;
                    case 'poor':
                        dot.classList.add('warning');
                        break;
                    default:
                        dot.classList.add('connecting');
                }
            }
        }

        if (text) {
            if (!isOnline) {
                text.textContent = 'Offline';
            } else {
                switch (quality) {
                    case 'excellent':
                        text.textContent = 'Connected';
                        break;
                    case 'good':
                        text.textContent = 'Connected';
                        break;
                    case 'poor':
                        text.textContent = 'Poor Connection';
                        break;
                    default:
                        text.textContent = 'Connecting...';
                }
            }
        }
    }

    // Pause active SSE connections when tab is hidden
    function pauseConnections() {
        if (typeof DRXStream !== 'undefined') {
            const connections = DRXStream.getAll();
            console.log(`[Connection] Pausing ${connections.length} connections (tab hidden)`);
            // Note: SSE connections will naturally pause when tab is hidden
            // This is mainly for tracking and potential future optimizations
        }
    }

    // Resume connections when tab becomes visible
    function resumeConnections() {
        if (typeof DRXStream !== 'undefined') {
            const connections = DRXStream.getAll();
            console.log(`[Connection] Resuming ${connections.length} connections (tab visible)`);
            // Check for any stale connections that need recovery
            checkConnections();
        }
    }

    // Initialize
    function init() {
        setupNetworkMonitoring();
        setupVisibilityMonitoring();
        updateQuality();

        // Update indicator on quality change
        on('qualityChange', updateIndicator);
        on('online', updateIndicator);
        on('offline', updateIndicator);

        // Handle visibility changes for connection pause/resume
        on('visibilityChange', ({ visible }) => {
            if (visible) {
                resumeConnections();
                startPingMonitor();
            } else {
                pauseConnections();
                stopPingMonitor();
            }
        });

        // Initial indicator update
        updateIndicator();

        // Start ping monitor when visible
        if (isVisible) {
            startPingMonitor();
        }

        console.log('[Connection] DRXConnection initialized');
    }

    return {
        init,
        on,
        off,
        getStatus,
        recoverInteraction,
        checkConnections,
        updateIndicator,
        isOnline: () => isOnline,
        isVisible: () => isVisible,
        getQuality: () => quality,
    };
})();

// Auto-initialize
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', DRXConnection.init);
    } else {
        DRXConnection.init();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = DRXConnection;
}
