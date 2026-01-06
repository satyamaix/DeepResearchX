/**
 * DRX Reconnection UI Manager
 *
 * Provides visual feedback for connection state changes:
 * - Connection lost banner with retry button
 * - Reconnection progress indicator
 * - Countdown timer for next retry attempt
 * - Pending interactions list with resume option
 * - Toast notifications for connection events
 */

const DRXReconnectUI = (() => {
    // State
    let bannerElement = null;
    let toastContainer = null;
    let countdownInterval = null;
    let currentCountdown = 0;
    let isVisible = false;
    let pendingInteractions = [];

    // SVG Icons
    const ICONS = {
        offline: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="1" y1="1" x2="23" y2="23"/>
            <path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"/>
            <path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39"/>
            <path d="M10.71 5.05A16 16 0 0 1 22.58 9"/>
            <path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88"/>
            <path d="M8.53 16.11a6 6 0 0 1 6.95 0"/>
            <line x1="12" y1="20" x2="12.01" y2="20"/>
        </svg>`,
        reconnecting: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="23 4 23 10 17 10"/>
            <polyline points="1 20 1 14 7 14"/>
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
        </svg>`,
        success: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
            <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>`,
        warning: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>`,
        error: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>`,
        close: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>`,
        retry: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="23 4 23 10 17 10"/>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
        </svg>`,
        resume: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>`,
        buffered: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <line x1="9" y1="3" x2="9" y2="21"/>
        </svg>`
    };

    /**
     * Initialize the reconnect UI
     * Creates DOM elements and sets up event listeners
     */
    function init() {
        createBannerElement();
        setupEventListeners();
        console.log('[ReconnectUI] Initialized');
    }

    /**
     * Create the connection banner element
     */
    function createBannerElement() {
        // Check if banner already exists
        if (document.getElementById('connection-banner')) {
            bannerElement = document.getElementById('connection-banner');
            return;
        }

        bannerElement = document.createElement('div');
        bannerElement.id = 'connection-banner';
        bannerElement.className = 'connection-banner';
        bannerElement.innerHTML = `
            <div class="connection-progress-container">
                <div class="connection-progress-bar" id="connection-progress-bar"></div>
            </div>
            <div class="connection-banner-content">
                <div class="connection-banner-main">
                    <div class="connection-banner-icon" id="connection-banner-icon">
                        ${ICONS.offline}
                    </div>
                    <div class="connection-banner-message">
                        <div class="connection-banner-title" id="connection-banner-title">Connection Lost</div>
                        <div class="connection-banner-subtitle" id="connection-banner-subtitle">
                            <span id="connection-banner-message">Attempting to reconnect...</span>
                            <span class="connection-countdown connection-hidden" id="connection-countdown">
                                <span class="connection-countdown-label">Retry in</span>
                                <span id="countdown-value">0</span>s
                            </span>
                        </div>
                    </div>
                </div>
                <div class="connection-banner-actions">
                    <button class="connection-retry-btn" id="connection-retry-btn">
                        ${ICONS.retry}
                        <span>Retry Now</span>
                    </button>
                    <button class="connection-dismiss-btn" id="connection-dismiss-btn">
                        ${ICONS.close}
                    </button>
                </div>
            </div>
            <div class="pending-interactions-container connection-hidden" id="pending-interactions-container">
                <div class="pending-interactions-header">
                    <span class="pending-interactions-title">Pending Interactions</span>
                    <span class="pending-interactions-count" id="pending-interactions-count">0</span>
                </div>
                <div class="pending-interactions-list" id="pending-interactions-list"></div>
            </div>
        `;

        // Insert at the beginning of body
        document.body.insertBefore(bannerElement, document.body.firstChild);

        // Set up toast container reference
        toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }
    }

    /**
     * Set up event listeners for DRXConnection events
     */
    function setupEventListeners() {
        // Retry button click
        const retryBtn = document.getElementById('connection-retry-btn');
        if (retryBtn) {
            retryBtn.addEventListener('click', handleRetryClick);
        }

        // Dismiss button click
        const dismissBtn = document.getElementById('connection-dismiss-btn');
        if (dismissBtn) {
            dismissBtn.addEventListener('click', handleDismissClick);
        }

        // Listen to DRXConnection events if available
        if (typeof DRXConnection !== 'undefined') {
            DRXConnection.on('offline', handleOffline);
            DRXConnection.on('online', handleOnline);
            DRXConnection.on('pendingInteractions', handlePendingInteractions);
            DRXConnection.on('qualityChange', handleQualityChange);
        }

        // Also listen for native online/offline events as backup
        window.addEventListener('online', () => {
            if (isVisible) {
                showToast('Connection restored', 'success');
                hide();
            }
        });

        window.addEventListener('offline', () => {
            show({
                message: 'You are offline',
                canRetry: true
            });
        });
    }

    /**
     * Show the connection banner
     * @param {Object} options - Display options
     * @param {string} options.message - Message to display
     * @param {number} [options.countdown] - Countdown seconds until next retry
     * @param {boolean} [options.canRetry] - Whether retry button should be enabled
     * @param {string} [options.state] - Banner state: 'offline', 'reconnecting', 'success'
     */
    function show(options = {}) {
        const {
            message = 'Connection lost',
            countdown = 0,
            canRetry = true,
            state = 'offline'
        } = options;

        if (!bannerElement) {
            createBannerElement();
        }

        // Update banner state
        bannerElement.className = `connection-banner visible ${state}`;

        // Update icon based on state
        const iconEl = document.getElementById('connection-banner-icon');
        if (iconEl) {
            iconEl.innerHTML = ICONS[state] || ICONS.offline;
        }

        // Update title based on state
        const titleEl = document.getElementById('connection-banner-title');
        if (titleEl) {
            switch (state) {
                case 'offline':
                    titleEl.textContent = 'Connection Lost';
                    break;
                case 'reconnecting':
                    titleEl.textContent = 'Reconnecting...';
                    break;
                case 'success':
                    titleEl.textContent = 'Connected';
                    break;
                default:
                    titleEl.textContent = 'Connection Issue';
            }
        }

        // Update message
        const messageEl = document.getElementById('connection-banner-message');
        if (messageEl) {
            messageEl.textContent = message;
        }

        // Update retry button state
        const retryBtn = document.getElementById('connection-retry-btn');
        if (retryBtn) {
            retryBtn.disabled = !canRetry || state === 'reconnecting';
        }

        // Handle countdown
        if (countdown > 0) {
            startCountdown(countdown);
        } else {
            stopCountdown();
        }

        // Update progress bar
        const progressBar = document.getElementById('connection-progress-bar');
        if (progressBar) {
            if (state === 'reconnecting') {
                progressBar.classList.add('indeterminate');
            } else {
                progressBar.classList.remove('indeterminate');
                progressBar.style.width = '0%';
            }
        }

        isVisible = true;

        // Adjust main content position
        adjustMainContent(true);
    }

    /**
     * Hide the connection banner
     */
    function hide() {
        if (!bannerElement) return;

        stopCountdown();
        bannerElement.classList.remove('visible');
        isVisible = false;

        // Reset main content position
        adjustMainContent(false);

        // Clear pending interactions display
        const pendingContainer = document.getElementById('pending-interactions-container');
        if (pendingContainer) {
            pendingContainer.classList.add('connection-hidden');
        }
    }

    /**
     * Adjust main content position when banner is shown/hidden
     * @param {boolean} bannerVisible - Whether banner is visible
     */
    function adjustMainContent(bannerVisible) {
        const appContainer = document.querySelector('.app-container');
        if (appContainer) {
            if (bannerVisible) {
                // Get banner height dynamically
                const bannerHeight = bannerElement ? bannerElement.offsetHeight : 0;
                appContainer.style.marginTop = `${bannerHeight}px`;
                appContainer.style.height = `calc(100vh - ${bannerHeight}px)`;
            } else {
                appContainer.style.marginTop = '0';
                appContainer.style.height = '100vh';
            }
        }
    }

    /**
     * Start the countdown timer
     * @param {number} seconds - Seconds until next retry
     */
    function startCountdown(seconds) {
        stopCountdown();
        currentCountdown = seconds;

        const countdownEl = document.getElementById('connection-countdown');
        const countdownValueEl = document.getElementById('countdown-value');

        if (countdownEl && countdownValueEl) {
            countdownEl.classList.remove('connection-hidden');
            countdownValueEl.textContent = currentCountdown;

            countdownInterval = setInterval(() => {
                currentCountdown--;
                countdownValueEl.textContent = currentCountdown;

                if (currentCountdown <= 0) {
                    stopCountdown();
                    // Update message to show reconnecting
                    const messageEl = document.getElementById('connection-banner-message');
                    if (messageEl) {
                        messageEl.textContent = 'Attempting to reconnect...';
                    }
                }
            }, 1000);
        }
    }

    /**
     * Stop the countdown timer
     */
    function stopCountdown() {
        if (countdownInterval) {
            clearInterval(countdownInterval);
            countdownInterval = null;
        }

        const countdownEl = document.getElementById('connection-countdown');
        if (countdownEl) {
            countdownEl.classList.add('connection-hidden');
        }
    }

    /**
     * Update the countdown display
     * @param {number} seconds - Current countdown value
     */
    function updateCountdown(seconds) {
        if (seconds > 0) {
            startCountdown(seconds);
        } else {
            stopCountdown();
        }
    }

    /**
     * Show pending interactions in the banner
     * @param {Array} interactions - Array of pending interaction objects
     */
    function showPending(interactions) {
        pendingInteractions = interactions || [];

        const container = document.getElementById('pending-interactions-container');
        const list = document.getElementById('pending-interactions-list');
        const countEl = document.getElementById('pending-interactions-count');

        if (!container || !list) return;

        if (pendingInteractions.length === 0) {
            container.classList.add('connection-hidden');
            return;
        }

        container.classList.remove('connection-hidden');
        countEl.textContent = pendingInteractions.length;

        // Build list HTML
        list.innerHTML = pendingInteractions.map(interaction => {
            const timeAgo = formatTimeAgo(interaction.updatedAt);
            const shortId = interaction.id.substring(0, 8);

            return `
                <div class="pending-interaction-item" data-id="${interaction.id}">
                    <div class="pending-interaction-info">
                        <div class="pending-interaction-id">${shortId}...</div>
                        <div class="pending-interaction-meta">Last event: ${interaction.lastEventId || 'N/A'} - ${timeAgo}</div>
                    </div>
                    <button class="pending-interaction-resume" onclick="DRXReconnectUI.resumeInteraction('${interaction.id}', '${interaction.lastEventId}')">
                        ${ICONS.resume}
                        <span>Resume</span>
                    </button>
                </div>
            `;
        }).join('');
    }

    /**
     * Resume a pending interaction
     * @param {string} interactionId - The interaction ID to resume
     * @param {string} lastEventId - The last event ID received
     */
    async function resumeInteraction(interactionId, lastEventId) {
        if (typeof DRXConnection !== 'undefined') {
            try {
                showToast('Resuming interaction...', 'info');
                await DRXConnection.recoverInteraction(interactionId, lastEventId);
                showToast('Interaction resumed successfully', 'success');

                // Remove from pending list
                pendingInteractions = pendingInteractions.filter(i => i.id !== interactionId);
                showPending(pendingInteractions);
            } catch (error) {
                console.error('[ReconnectUI] Failed to resume interaction:', error);
                showToast('Failed to resume interaction', 'error');
            }
        }
    }

    /**
     * Show a toast notification
     * @param {string} message - Toast message
     * @param {string} type - Toast type: 'success', 'warning', 'error', 'info'
     * @param {Object} [options] - Additional options
     * @param {string} [options.title] - Optional title
     * @param {number} [options.duration] - Duration in ms (default: 5000)
     */
    function showToast(message, type = 'info', options = {}) {
        const {
            title = null,
            duration = 5000
        } = options;

        if (!toastContainer) {
            toastContainer = document.getElementById('toast-container');
            if (!toastContainer) return;
        }

        const toastId = `toast-${Date.now()}`;
        const iconHtml = ICONS[type] || ICONS.warning;

        const toastEl = document.createElement('div');
        toastEl.id = toastId;
        toastEl.className = `connection-toast ${type}`;
        toastEl.innerHTML = `
            <div class="connection-toast-icon">
                ${iconHtml}
            </div>
            <div class="connection-toast-content">
                ${title ? `<div class="connection-toast-title">${title}</div>` : ''}
                <div class="connection-toast-message">${message}</div>
            </div>
            <button class="connection-toast-close" onclick="DRXReconnectUI.dismissToast('${toastId}')">
                ${ICONS.close}
            </button>
        `;

        toastContainer.appendChild(toastEl);

        // Auto-dismiss after duration
        if (duration > 0) {
            setTimeout(() => {
                dismissToast(toastId);
            }, duration);
        }

        return toastId;
    }

    /**
     * Dismiss a toast notification
     * @param {string} toastId - The toast element ID
     */
    function dismissToast(toastId) {
        const toastEl = document.getElementById(toastId);
        if (toastEl) {
            toastEl.classList.add('fade-out');
            setTimeout(() => {
                toastEl.remove();
            }, 300);
        }
    }

    /**
     * Handle retry button click
     */
    function handleRetryClick() {
        const retryBtn = document.getElementById('connection-retry-btn');
        if (retryBtn) {
            retryBtn.disabled = true;
        }

        stopCountdown();

        // Update to reconnecting state
        show({
            message: 'Attempting to reconnect...',
            state: 'reconnecting',
            canRetry: false
        });

        // Trigger reconnection attempt
        if (typeof DRXStream !== 'undefined' && DRXStream.reconnectAll) {
            DRXStream.reconnectAll();
        }

        // If DRXConnection has a reconnect method
        if (typeof DRXConnection !== 'undefined' && DRXConnection.checkConnections) {
            DRXConnection.checkConnections();
        }

        // Also try to ping the server
        fetch('/api/v1/health', { method: 'GET', cache: 'no-store' })
            .then(response => {
                if (response.ok) {
                    handleOnline({ timestamp: Date.now() });
                } else {
                    throw new Error('Server not responding');
                }
            })
            .catch(() => {
                // Show offline state with retry countdown
                show({
                    message: 'Server unreachable. Will retry...',
                    countdown: 10,
                    canRetry: true,
                    state: 'offline'
                });
            });
    }

    /**
     * Handle dismiss button click
     */
    function handleDismissClick() {
        hide();
    }

    /**
     * Handle offline event from DRXConnection
     * @param {Object} data - Event data
     */
    function handleOffline(data) {
        console.log('[ReconnectUI] Offline event:', data);
        show({
            message: 'Check your internet connection',
            countdown: 5,
            canRetry: true,
            state: 'offline'
        });
    }

    /**
     * Handle online event from DRXConnection
     * @param {Object} data - Event data
     */
    function handleOnline(data) {
        console.log('[ReconnectUI] Online event:', data);

        // Briefly show success state
        show({
            message: 'Connection restored',
            state: 'success',
            canRetry: false
        });

        showToast('Connection restored', 'success', {
            title: 'Back Online',
            duration: 3000
        });

        // Hide banner after a brief delay
        setTimeout(() => {
            hide();
        }, 2000);
    }

    /**
     * Handle pending interactions event from DRXConnection
     * @param {Object} data - Event data with interactions array
     */
    function handlePendingInteractions(data) {
        console.log('[ReconnectUI] Pending interactions:', data);
        if (data.interactions && data.interactions.length > 0) {
            showPending(data.interactions);

            // Show notification about pending interactions
            showToast(
                `${data.interactions.length} research session(s) can be resumed`,
                'info',
                { duration: 8000 }
            );
        }
    }

    /**
     * Handle connection quality change event
     * @param {Object} data - Event data with quality info
     */
    function handleQualityChange(data) {
        console.log('[ReconnectUI] Quality change:', data);

        if (data.quality === 'poor' && !isVisible) {
            showToast('Connection quality is poor', 'warning', {
                duration: 4000
            });
        }
    }

    /**
     * Format timestamp as relative time (e.g., "2 minutes ago")
     * @param {number} timestamp - Unix timestamp in ms
     * @returns {string} - Formatted relative time
     */
    function formatTimeAgo(timestamp) {
        const seconds = Math.floor((Date.now() - timestamp) / 1000);

        if (seconds < 60) return 'just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }

    /**
     * Get current visibility state
     * @returns {boolean} - Whether banner is visible
     */
    function isShowing() {
        return isVisible;
    }

    /**
     * Update progress bar (for SSE reconnection progress)
     * @param {number} percent - Progress percentage (0-100)
     */
    function updateProgress(percent) {
        const progressBar = document.getElementById('connection-progress-bar');
        if (progressBar) {
            progressBar.classList.remove('indeterminate');
            progressBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
        }
    }

    // Initialize on DOM ready
    if (typeof document !== 'undefined') {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
    }

    // Public API
    return {
        init,
        show,
        hide,
        showPending,
        updateCountdown,
        showToast,
        dismissToast,
        resumeInteraction,
        updateProgress,
        isShowing,
    };
})();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DRXReconnectUI;
}
