/**
 * DRX Keyboard Navigation & Accessibility
 * Handles keyboard shortcuts, focus management, and ARIA updates
 */

const DRXKeyboard = (function() {
    'use strict';

    // State
    let initialized = false;
    let shortcuts = {};
    let focusTrapStack = [];
    let helpModalActive = false;

    // Platform detection
    const isMac = /Mac|iPhone|iPod|iPad/i.test(navigator.platform);
    const modKey = isMac ? 'meta' : 'ctrl';
    const modKeyDisplay = isMac ? '⌘' : 'Ctrl';

    // Shortcut definitions
    const SHORTCUTS = {
        'send': {
            keys: ['ctrl+enter', 'meta+enter'],
            description: 'Send message / Start research',
            handler: handleSendMessage,
            display: `${modKeyDisplay}+Enter`
        },
        'escape': {
            keys: ['escape'],
            description: 'Close modal / Cancel action',
            handler: handleEscape,
            display: 'Esc'
        },
        'toggleSettings': {
            keys: ['ctrl+,', 'meta+,'],
            description: 'Toggle research settings',
            handler: handleToggleSettings,
            display: `${modKeyDisplay}+,`
        },
        'focusSearch': {
            keys: ['ctrl+/', 'meta+/'],
            description: 'Focus search/input',
            handler: handleFocusSearch,
            display: `${modKeyDisplay}+/`
        },
        'copyResponse': {
            keys: ['ctrl+shift+c', 'meta+shift+c'],
            description: 'Copy last response',
            handler: handleCopyResponse,
            display: `${modKeyDisplay}+Shift+C`
        },
        'showHelp': {
            keys: ['?'],
            description: 'Show keyboard shortcuts',
            handler: handleShowHelp,
            display: '?'
        }
    };

    /**
     * Initialize keyboard navigation
     */
    function init() {
        if (initialized) {
            console.warn('[Keyboard] Already initialized');
            return;
        }

        console.log('[Keyboard] Initializing...');

        // Register all shortcuts
        Object.keys(SHORTCUTS).forEach(name => {
            const shortcut = SHORTCUTS[name];
            shortcut.keys.forEach(key => {
                shortcuts[key] = shortcut.handler;
            });
        });

        // Global keyboard event listener
        document.addEventListener('keydown', handleKeyDown, true);

        // Add ARIA labels to elements
        addARIALabels();

        // Initialize focus trap for modals
        initializeFocusTraps();

        initialized = true;
        console.log('[Keyboard] Ready');
    }

    /**
     * Handle global keydown events
     */
    function handleKeyDown(e) {
        // Build key combination string
        const key = buildKeyString(e);

        // Check if we have a handler for this combination
        const handler = shortcuts[key];

        if (handler) {
            // Don't prevent default if typing in input/textarea (except special cases)
            const activeEl = document.activeElement;
            const isInput = activeEl && (
                activeEl.tagName === 'INPUT' ||
                activeEl.tagName === 'TEXTAREA' ||
                activeEl.isContentEditable
            );

            // Allow Ctrl+Enter and Escape in input fields
            if (key === 'escape' || key.includes('enter')) {
                e.preventDefault();
                handler(e);
            } else if (!isInput) {
                // Only handle other shortcuts if not in input field
                e.preventDefault();
                handler(e);
            }
        }
    }

    /**
     * Build key string from keyboard event
     */
    function buildKeyString(e) {
        const parts = [];

        if (e.ctrlKey) parts.push('ctrl');
        if (e.metaKey) parts.push('meta');
        if (e.altKey) parts.push('alt');
        if (e.shiftKey && e.key.length > 1) parts.push('shift'); // Only add shift for special keys

        // Normalize key name
        let key = e.key.toLowerCase();

        // Handle special cases
        if (key === ' ') key = 'space';

        parts.push(key);

        return parts.join('+');
    }

    /**
     * Register a custom shortcut
     */
    function register(shortcut, handler, description = '') {
        if (typeof handler !== 'function') {
            console.error('[Keyboard] Handler must be a function');
            return;
        }

        shortcuts[shortcut.toLowerCase()] = handler;
        console.log('[Keyboard] Registered shortcut:', shortcut);
    }

    /**
     * Unregister a shortcut
     */
    function unregister(shortcut) {
        delete shortcuts[shortcut.toLowerCase()];
        console.log('[Keyboard] Unregistered shortcut:', shortcut);
    }

    /**
     * Handler: Send message
     */
    function handleSendMessage(e) {
        const sendBtn = document.getElementById('send-btn');
        const queryInput = document.getElementById('query-input');

        if (sendBtn && !sendBtn.disabled && queryInput && document.activeElement === queryInput) {
            // Trigger send
            if (typeof sendQuery === 'function') {
                sendQuery();
            }
        }
    }

    /**
     * Handler: Escape key
     */
    function handleEscape(e) {
        // Check if help modal is active
        if (helpModalActive) {
            hideHelp();
            return;
        }

        // Check if report modal is active
        const modal = document.getElementById('report-modal');
        if (modal && modal.classList.contains('active')) {
            if (typeof closeModal === 'function') {
                closeModal();
            }
            return;
        }

        // Blur active input
        if (document.activeElement &&
            (document.activeElement.tagName === 'INPUT' ||
             document.activeElement.tagName === 'TEXTAREA')) {
            document.activeElement.blur();
        }
    }

    /**
     * Handler: Toggle settings sidebar
     */
    function handleToggleSettings(e) {
        const sidebar = document.getElementById('sidebar-right');
        if (sidebar) {
            sidebar.classList.toggle('collapsed');
        }
    }

    /**
     * Handler: Focus search/input
     */
    function handleFocusSearch(e) {
        const queryInput = document.getElementById('query-input');
        if (queryInput) {
            queryInput.focus();
            // Move cursor to end
            queryInput.setSelectionRange(queryInput.value.length, queryInput.value.length);
        }
    }

    /**
     * Handler: Copy last response
     */
    function handleCopyResponse(e) {
        // Get the last assistant message
        const messages = document.querySelectorAll('.message.assistant');
        if (messages.length === 0) {
            if (typeof showToast === 'function') {
                showToast('No response to copy', 'warning');
            }
            return;
        }

        const lastMessage = messages[messages.length - 1];
        const content = lastMessage.querySelector('.message-content');

        if (content) {
            const text = content.innerText || content.textContent;

            navigator.clipboard.writeText(text).then(() => {
                if (typeof showToast === 'function') {
                    showToast('Response copied to clipboard', 'success');
                }
            }).catch(() => {
                if (typeof showToast === 'function') {
                    showToast('Failed to copy response', 'error');
                }
            });
        }
    }

    /**
     * Handler: Show help modal
     */
    function handleShowHelp(e) {
        // Don't show help if typing in input field (? is a common character)
        const activeEl = document.activeElement;
        if (activeEl && (
            activeEl.tagName === 'INPUT' ||
            activeEl.tagName === 'TEXTAREA' ||
            activeEl.isContentEditable
        )) {
            return;
        }

        showHelp();
    }

    /**
     * Show keyboard shortcuts help modal
     */
    function showHelp() {
        if (helpModalActive) return;

        const modal = createHelpModal();
        document.body.appendChild(modal);

        // Trap focus in modal
        trapFocus(modal);

        // Show modal with animation
        setTimeout(() => {
            modal.classList.add('active');
            helpModalActive = true;

            // Focus close button
            const closeBtn = modal.querySelector('.keyboard-help-close');
            if (closeBtn) closeBtn.focus();
        }, 10);
    }

    /**
     * Hide help modal
     */
    function hideHelp() {
        const modal = document.getElementById('keyboard-help-modal');
        if (modal) {
            modal.classList.remove('active');
            helpModalActive = false;

            // Release focus trap
            releaseFocusTrap();

            // Remove modal after animation
            setTimeout(() => {
                modal.remove();
            }, 300);
        }
    }

    /**
     * Create help modal element
     */
    function createHelpModal() {
        const modal = document.createElement('div');
        modal.id = 'keyboard-help-modal';
        modal.className = 'modal keyboard-help-modal';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.setAttribute('aria-labelledby', 'keyboard-help-title');

        const shortcuts = Object.values(SHORTCUTS);

        modal.innerHTML = `
            <div class="modal-backdrop" onclick="DRXKeyboard.hideHelp()"></div>
            <div class="modal-content keyboard-help-content">
                <div class="modal-header">
                    <h2 id="keyboard-help-title">Keyboard Shortcuts</h2>
                    <button class="btn btn-icon keyboard-help-close" onclick="DRXKeyboard.hideHelp()" aria-label="Close help">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M18 6L6 18M6 6l12 12"/>
                        </svg>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="keyboard-shortcuts-list">
                        ${shortcuts.map(sc => `
                            <div class="keyboard-shortcut-item">
                                <kbd class="keyboard-shortcut-key">${sc.display}</kbd>
                                <span class="keyboard-shortcut-desc">${sc.description}</span>
                            </div>
                        `).join('')}
                    </div>
                    <div class="keyboard-help-footer">
                        <p>Press <kbd>?</kbd> to show this help anytime</p>
                        <p>Press <kbd>Esc</kbd> to close</p>
                    </div>
                </div>
            </div>
        `;

        return modal;
    }

    /**
     * Add ARIA labels to interactive elements
     */
    function addARIALabels() {
        // Send button
        const sendBtn = document.getElementById('send-btn');
        if (sendBtn && !sendBtn.getAttribute('aria-label')) {
            sendBtn.setAttribute('aria-label', 'Send research query');
        }

        // Report modal
        const reportModal = document.getElementById('report-modal');
        if (reportModal) {
            if (!reportModal.getAttribute('role')) {
                reportModal.setAttribute('role', 'dialog');
            }
            if (!reportModal.getAttribute('aria-modal')) {
                reportModal.setAttribute('aria-modal', 'true');
            }
            if (!reportModal.querySelector('.modal-header h2')) {
                // Add aria-labelledby if header exists
                const heading = reportModal.querySelector('.modal-header h2');
                if (heading && !heading.id) {
                    heading.id = 'report-modal-title';
                    reportModal.setAttribute('aria-labelledby', 'report-modal-title');
                }
            }
        }

        // Progress container (status updates)
        const progressContainer = document.getElementById('progress-container');
        if (progressContainer && !progressContainer.getAttribute('aria-live')) {
            progressContainer.setAttribute('aria-live', 'polite');
            progressContainer.setAttribute('aria-atomic', 'false');
        }

        // Progress phase
        const progressPhase = document.getElementById('progress-phase');
        if (progressPhase && !progressPhase.getAttribute('role')) {
            progressPhase.setAttribute('role', 'status');
        }

        // Connection status
        const connectionStatus = document.getElementById('connection-status');
        if (connectionStatus && !connectionStatus.getAttribute('aria-live')) {
            connectionStatus.setAttribute('aria-live', 'polite');
            connectionStatus.setAttribute('role', 'status');
        }

        // Query input
        const queryInput = document.getElementById('query-input');
        if (queryInput && !queryInput.getAttribute('aria-label')) {
            queryInput.setAttribute('aria-label', 'Enter your research question');
        }

        // Icon buttons without labels
        addARIAToIconButtons();

        // DAG control buttons
        addARIAToDAGControls();

        // Toast container
        const toastContainer = document.getElementById('toast-container');
        if (toastContainer && !toastContainer.getAttribute('aria-live')) {
            toastContainer.setAttribute('aria-live', 'polite');
            toastContainer.setAttribute('aria-atomic', 'true');
        }

        console.log('[Keyboard] ARIA labels added');
    }

    /**
     * Add ARIA labels to icon buttons
     */
    function addARIAToIconButtons() {
        // Copy button in modal
        const modalButtons = document.querySelectorAll('.modal-actions .btn');
        modalButtons.forEach(btn => {
            if (!btn.getAttribute('aria-label')) {
                const text = btn.textContent.trim();
                if (text) {
                    btn.setAttribute('aria-label', text);
                }
            }
        });

        // Graph filter buttons
        const graphButtons = document.querySelectorAll('.graph-filter-btn');
        graphButtons.forEach(btn => {
            if (!btn.getAttribute('aria-label')) {
                const text = btn.textContent.trim();
                if (text) {
                    btn.setAttribute('aria-label', `Filter by ${text}`);
                }
            }
        });

        // Example query buttons
        const exampleButtons = document.querySelectorAll('.example-btn');
        exampleButtons.forEach(btn => {
            if (!btn.getAttribute('aria-label')) {
                btn.setAttribute('aria-label', 'Use this example query');
            }
        });
    }

    /**
     * Add ARIA labels to DAG control buttons
     */
    function addARIAToDAGControls() {
        const dagControls = document.querySelectorAll('.dag-control-btn');
        dagControls.forEach(btn => {
            if (!btn.getAttribute('aria-label') && btn.title) {
                btn.setAttribute('aria-label', btn.title);
            }
        });
    }

    /**
     * Initialize focus traps for modals
     */
    function initializeFocusTraps() {
        // Observe DOM for modal activation
        const observer = new MutationObserver(mutations => {
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1 && node.classList &&
                        node.classList.contains('modal') &&
                        node.classList.contains('active')) {
                        trapFocus(node);
                    }
                });
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['class']
        });
    }

    /**
     * Trap focus within an element (for modals)
     */
    function trapFocus(element) {
        // Get all focusable elements
        const focusableElements = element.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );

        if (focusableElements.length === 0) return;

        const firstFocusable = focusableElements[0];
        const lastFocusable = focusableElements[focusableElements.length - 1];

        function handleTrapFocus(e) {
            if (e.key !== 'Tab') return;

            if (e.shiftKey) {
                // Shift + Tab
                if (document.activeElement === firstFocusable) {
                    e.preventDefault();
                    lastFocusable.focus();
                }
            } else {
                // Tab
                if (document.activeElement === lastFocusable) {
                    e.preventDefault();
                    firstFocusable.focus();
                }
            }
        }

        element.addEventListener('keydown', handleTrapFocus);

        // Store for cleanup
        focusTrapStack.push({
            element,
            handler: handleTrapFocus
        });

        console.log('[Keyboard] Focus trapped in modal');
    }

    /**
     * Release the current focus trap
     */
    function releaseFocusTrap() {
        if (focusTrapStack.length === 0) return;

        const trap = focusTrapStack.pop();
        trap.element.removeEventListener('keydown', trap.handler);

        console.log('[Keyboard] Focus trap released');
    }

    /**
     * Cleanup
     */
    function destroy() {
        if (!initialized) return;

        document.removeEventListener('keydown', handleKeyDown, true);

        // Clear all focus traps
        focusTrapStack.forEach(trap => {
            trap.element.removeEventListener('keydown', trap.handler);
        });
        focusTrapStack = [];

        initialized = false;
        console.log('[Keyboard] Destroyed');
    }

    // Public API
    return {
        init,
        register,
        unregister,
        showHelp,
        hideHelp,
        destroy
    };
})();

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => DRXKeyboard.init());
} else {
    DRXKeyboard.init();
}
