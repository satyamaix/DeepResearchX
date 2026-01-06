/**
 * DRX Chat History Component
 * Handles message display, virtualization, and scroll behavior
 */

const DRXChat = (() => {
    // Configuration
    const CONFIG = {
        VIRTUALIZATION_THRESHOLD: 50,  // Enable virtualization above this many messages
        OVERSCAN: 5,                   // Extra messages to render above/below viewport
        ESTIMATED_MESSAGE_HEIGHT: 120, // Estimated height for virtualization
        SCROLL_THRESHOLD: 100,         // Distance from bottom to auto-scroll
        SCROLL_DEBOUNCE: 100,          // Debounce scroll events (ms)
    };

    // State
    let messages = [];
    let renderedMessages = new Map(); // id -> DOM element
    let streamRenderers = new Map();  // id -> MarkdownStream instance
    let isTyping = false;
    let isWaiting = false;
    let waitingPhase = '';
    let autoScroll = true;
    let scrollTimeout = null;
    let resizeObserver = null;

    // DOM Elements (cached)
    let container = null;
    let messagesContainer = null;
    let typingIndicator = null;
    let waitingIndicator = null;
    let scrollBottomBtn = null;

    /**
     * Initialize the chat component
     */
    function init() {
        container = document.getElementById('chat-container');
        messagesContainer = document.getElementById('chat-messages');

        if (!container || !messagesContainer) {
            console.error('[DRXChat] Required containers not found');
            return;
        }

        // Create UI elements
        createTypingIndicator();
        createWaitingIndicator();
        createScrollBottomButton();

        // Setup event listeners
        setupScrollListener();
        setupResizeObserver();

        // Load persisted messages
        loadPersistedMessages();

        console.log('[DRXChat] Initialized');
    }

    /**
     * Create typing indicator element
     */
    function createTypingIndicator() {
        typingIndicator = document.createElement('div');
        typingIndicator.className = 'chat-typing-indicator';
        typingIndicator.innerHTML = `
            <div class="typing-bubble">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span class="typing-label">Assistant is typing...</span>
            </div>
        `;
        messagesContainer.appendChild(typingIndicator);
    }

    /**
     * Create waiting phase indicator element
     */
    function createWaitingIndicator() {
        waitingIndicator = document.createElement('div');
        waitingIndicator.className = 'chat-waiting-indicator';
        waitingIndicator.innerHTML = `
            <div class="waiting-bubble" data-phase="">
                <div class="waiting-spinner"></div>
                <span class="waiting-phase"></span>
            </div>
        `;
        messagesContainer.appendChild(waitingIndicator);
    }

    /**
     * Create scroll to bottom button
     */
    function createScrollBottomButton() {
        scrollBottomBtn = document.createElement('button');
        scrollBottomBtn.className = 'chat-scroll-bottom';
        scrollBottomBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 5v14M19 12l-7 7-7-7"/>
            </svg>
            <span>Scroll to bottom</span>
        `;
        scrollBottomBtn.addEventListener('click', () => scrollToBottom(true));
        container.appendChild(scrollBottomBtn);
    }

    /**
     * Setup scroll event listener
     */
    function setupScrollListener() {
        messagesContainer.addEventListener('scroll', () => {
            if (scrollTimeout) clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(handleScroll, CONFIG.SCROLL_DEBOUNCE);
        });
    }

    /**
     * Handle scroll events
     */
    function handleScroll() {
        const { scrollTop, scrollHeight, clientHeight } = messagesContainer;
        const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

        // Update auto-scroll behavior
        autoScroll = distanceFromBottom < CONFIG.SCROLL_THRESHOLD;

        // Show/hide scroll button
        if (scrollBottomBtn) {
            scrollBottomBtn.classList.toggle('visible', !autoScroll && distanceFromBottom > 200);
        }

        // Virtualization check
        if (messages.length > CONFIG.VIRTUALIZATION_THRESHOLD) {
            updateVirtualization();
        }
    }

    /**
     * Setup resize observer for container
     */
    function setupResizeObserver() {
        resizeObserver = new ResizeObserver(() => {
            if (autoScroll) {
                scrollToBottom(false);
            }
        });
        resizeObserver.observe(messagesContainer);
    }

    /**
     * Load messages from persistence layer
     */
    function loadPersistedMessages() {
        if (typeof DRXPersistence === 'undefined') {
            console.warn('[DRXChat] Persistence layer not available');
            return;
        }

        const savedMessages = DRXPersistence.getMessages();
        if (savedMessages && savedMessages.length > 0) {
            // Filter out welcome message and render saved messages
            messages = savedMessages;
            renderAllMessages();
        }
    }

    /**
     * Add a new message to the chat
     * @param {ChatMessage} message - The message to add
     */
    function addMessage(message) {
        // Validate message
        if (!message || !message.id || !message.role) {
            console.error('[DRXChat] Invalid message:', message);
            return;
        }

        // Add timestamp if missing
        if (!message.timestamp) {
            message.timestamp = Date.now();
        }

        // Add status if missing
        if (!message.status) {
            message.status = 'complete';
        }

        // Add to messages array
        messages.push(message);

        // Persist the message
        if (typeof DRXPersistence !== 'undefined') {
            DRXPersistence.saveMessage(message);
        }

        // Render the message
        const element = renderMessage(message);

        // Insert before typing/waiting indicators
        if (typingIndicator && typingIndicator.parentNode) {
            messagesContainer.insertBefore(element, typingIndicator);
        } else {
            messagesContainer.appendChild(element);
        }

        renderedMessages.set(message.id, element);

        // Auto-scroll if enabled
        if (autoScroll) {
            scrollToBottom(false);
        }

        return element;
    }

    /**
     * Update an existing message
     * @param {string} id - Message ID
     * @param {Partial<ChatMessage>} updates - Updates to apply
     */
    function updateMessage(id, updates) {
        const index = messages.findIndex(m => m.id === id);
        if (index === -1) {
            console.warn('[DRXChat] Message not found:', id);
            return;
        }

        // Update message in array
        messages[index] = { ...messages[index], ...updates };

        // Persist update
        if (typeof DRXPersistence !== 'undefined') {
            DRXPersistence.updateMessage(id, updates);
        }

        // Re-render if element exists
        const element = renderedMessages.get(id);
        if (element) {
            // Update status indicator
            if (updates.status) {
                updateMessageStatus(element, updates.status);
            }

            // Update content if provided (for streaming)
            if (updates.content !== undefined) {
                updateMessageContent(element, messages[index]);
            }
        }
    }

    /**
     * Remove a message from the chat
     * @param {string} id - Message ID to remove
     */
    function removeMessage(id) {
        const index = messages.findIndex(m => m.id === id);
        if (index === -1) return;

        // Remove from array
        messages.splice(index, 1);

        // Remove from DOM
        const element = renderedMessages.get(id);
        if (element) {
            element.remove();
            renderedMessages.delete(id);
        }

        // Clean up stream renderer
        if (streamRenderers.has(id)) {
            streamRenderers.delete(id);
        }
    }

    /**
     * Render a single message
     * @param {ChatMessage} message - The message to render
     * @returns {HTMLElement} The rendered message element
     */
    function renderMessage(message) {
        const el = document.createElement('div');
        el.className = `chat-message ${message.role}`;
        el.dataset.messageId = message.id;
        if (message.status === 'error') {
            el.classList.add('error');
        }

        // Build message HTML
        el.innerHTML = `
            <div class="chat-message-header">
                <span class="chat-role-indicator">${getRoleLabel(message.role)}</span>
            </div>
            <div class="chat-bubble ${message.status === 'streaming' ? 'streaming' : ''}">
                <div class="chat-bubble-content ${message.role === 'assistant' ? 'markdown-content' : ''}">
                    ${renderMessageContent(message)}
                </div>
            </div>
            <div class="chat-message-footer">
                <span class="chat-timestamp">${formatTimestamp(message.timestamp)}</span>
                <div class="chat-status ${message.status}">
                    <span class="chat-status-dot"></span>
                </div>
                ${renderMessageActions(message)}
            </div>
        `;

        // Setup action handlers
        setupMessageActions(el, message);

        return el;
    }

    /**
     * Get label for message role
     */
    function getRoleLabel(role) {
        const labels = {
            user: 'You',
            assistant: 'DRX',
            system: 'System'
        };
        return labels[role] || role;
    }

    /**
     * Render message content
     */
    function renderMessageContent(message) {
        if (message.status === 'error') {
            return `
                <div class="chat-error-content">
                    <svg class="chat-error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M12 8v4M12 16h.01"/>
                    </svg>
                    <div>
                        <div class="chat-error-text">${escapeHtml(message.content || 'An error occurred')}</div>
                        ${message.errorDetails ? `<div class="chat-error-details">${escapeHtml(message.errorDetails)}</div>` : ''}
                    </div>
                </div>
            `;
        }

        // Use DRXRenderer for assistant markdown content
        if (message.role === 'assistant' && typeof DRXRenderer !== 'undefined') {
            return DRXRenderer.render(message.content || '');
        }

        return escapeHtml(message.content || '');
    }

    /**
     * Render message actions
     */
    function renderMessageActions(message) {
        const actions = [];

        // Copy action for all messages
        actions.push(`
            <button class="chat-action-btn copy" title="Copy to clipboard" data-action="copy">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                </svg>
            </button>
        `);

        // Retry action for error messages
        if (message.status === 'error' && message.role === 'user') {
            actions.push(`
                <button class="chat-action-btn retry" title="Retry" data-action="retry">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M1 4v6h6M23 20v-6h-6"/>
                        <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15"/>
                    </svg>
                </button>
            `);
        }

        return `<div class="chat-actions">${actions.join('')}</div>`;
    }

    /**
     * Setup action handlers for a message element
     */
    function setupMessageActions(element, message) {
        // Copy action
        const copyBtn = element.querySelector('[data-action="copy"]');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => copyMessageContent(message, copyBtn));
        }

        // Retry action
        const retryBtn = element.querySelector('[data-action="retry"]');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => retryMessage(message));
        }
    }

    /**
     * Copy message content to clipboard
     */
    async function copyMessageContent(message, button) {
        try {
            await navigator.clipboard.writeText(message.content || '');
            button.classList.add('copied');
            setTimeout(() => button.classList.remove('copied'), 2000);
        } catch (err) {
            console.error('[DRXChat] Failed to copy:', err);
        }
    }

    /**
     * Retry a failed message
     */
    function retryMessage(message) {
        // Dispatch custom event for retry handling
        const event = new CustomEvent('drx:retry-message', {
            detail: { message }
        });
        document.dispatchEvent(event);
    }

    /**
     * Update message status indicator
     */
    function updateMessageStatus(element, status) {
        const statusEl = element.querySelector('.chat-status');
        if (statusEl) {
            statusEl.className = `chat-status ${status}`;
        }

        const bubble = element.querySelector('.chat-bubble');
        if (bubble) {
            bubble.classList.toggle('streaming', status === 'streaming');
        }

        if (status === 'error') {
            element.classList.add('error');
        } else {
            element.classList.remove('error');
        }
    }

    /**
     * Update message content (for streaming)
     */
    function updateMessageContent(element, message) {
        const contentEl = element.querySelector('.chat-bubble-content');
        if (!contentEl) return;

        if (message.role === 'assistant' && typeof DRXRenderer !== 'undefined') {
            contentEl.innerHTML = DRXRenderer.render(message.content || '');
        } else {
            contentEl.textContent = message.content || '';
        }

        // Auto-scroll during streaming
        if (autoScroll) {
            scrollToBottom(false);
        }
    }

    /**
     * Start streaming content to a message
     * @param {string} id - Message ID
     * @returns {Object} Stream control object with append() and finalize() methods
     */
    function startStream(id) {
        const element = renderedMessages.get(id);
        if (!element) {
            console.error('[DRXChat] Cannot start stream, message not found:', id);
            return null;
        }

        const contentEl = element.querySelector('.chat-bubble-content');
        if (!contentEl) return null;

        // Clear content for streaming
        contentEl.innerHTML = '';

        // Create stream renderer
        let stream = null;
        if (typeof DRXRenderer !== 'undefined') {
            stream = DRXRenderer.createStream(contentEl);
            streamRenderers.set(id, stream);
        }

        // Update status
        updateMessage(id, { status: 'streaming' });

        return {
            append: (chunk) => {
                if (stream) {
                    stream.append(chunk);
                } else {
                    contentEl.textContent += chunk;
                }

                // Update stored content
                const msg = messages.find(m => m.id === id);
                if (msg) {
                    msg.content = (msg.content || '') + chunk;
                }

                // Auto-scroll during streaming
                if (autoScroll) {
                    scrollToBottom(false);
                }
            },
            finalize: () => {
                if (stream) {
                    stream.finalize();
                }
                streamRenderers.delete(id);
                updateMessage(id, { status: 'complete' });

                // Persist final content
                const msg = messages.find(m => m.id === id);
                if (msg && typeof DRXPersistence !== 'undefined') {
                    DRXPersistence.updateMessage(id, { content: msg.content, status: 'complete' });
                }
            }
        };
    }

    /**
     * Scroll to the bottom of the chat
     * @param {boolean} smooth - Use smooth scrolling
     */
    function scrollToBottom(smooth = true) {
        if (!messagesContainer) return;

        messagesContainer.scrollTo({
            top: messagesContainer.scrollHeight,
            behavior: smooth ? 'smooth' : 'auto'
        });

        autoScroll = true;
        if (scrollBottomBtn) {
            scrollBottomBtn.classList.remove('visible');
        }
    }

    /**
     * Scroll to a specific message
     * @param {string} id - Message ID
     */
    function scrollToMessage(id) {
        const element = renderedMessages.get(id);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    /**
     * Set typing indicator visibility
     * @param {boolean} typing - Whether assistant is typing
     */
    function setTyping(typing) {
        isTyping = typing;
        if (typingIndicator) {
            typingIndicator.classList.toggle('active', typing);
        }

        // Hide waiting indicator when typing starts
        if (typing && waitingIndicator) {
            waitingIndicator.classList.remove('active');
        }

        if (typing && autoScroll) {
            scrollToBottom(false);
        }
    }

    /**
     * Set waiting indicator visibility and phase
     * @param {boolean} waiting - Whether waiting for response
     * @param {string} phase - Current phase (Planning, Searching, etc.)
     */
    function setWaiting(waiting, phase = '') {
        isWaiting = waiting;
        waitingPhase = phase;

        if (waitingIndicator) {
            waitingIndicator.classList.toggle('active', waiting);

            const bubble = waitingIndicator.querySelector('.waiting-bubble');
            const phaseEl = waitingIndicator.querySelector('.waiting-phase');

            if (bubble && phase) {
                bubble.dataset.phase = phase.toLowerCase();
            }

            if (phaseEl) {
                phaseEl.textContent = phase ? `${getPhaseIcon(phase)} ${phase}...` : '';
            }
        }

        // Hide typing indicator when waiting starts
        if (waiting && typingIndicator) {
            typingIndicator.classList.remove('active');
        }

        if (waiting && autoScroll) {
            scrollToBottom(false);
        }
    }

    /**
     * Get icon for phase
     */
    function getPhaseIcon(phase) {
        const icons = {
            'Planning': '\u{1F4DD}',      // Memo
            'Searching': '\u{1F50D}',     // Magnifying glass
            'Reading': '\u{1F4D6}',       // Book
            'Synthesizing': '\u{2699}',   // Gear
            'Reviewing': '\u{1F50E}',     // Right magnifying glass
            'Reporting': '\u{1F4C4}'      // Document
        };
        return icons[phase] || '\u{23F3}'; // Hourglass as default
    }

    /**
     * Render all messages (used for initial load and virtualization)
     */
    function renderAllMessages() {
        // Clear existing rendered messages
        renderedMessages.forEach(el => el.remove());
        renderedMessages.clear();

        // Don't render if no messages
        if (messages.length === 0) return;

        // Check if virtualization is needed
        if (messages.length > CONFIG.VIRTUALIZATION_THRESHOLD) {
            setupVirtualization();
        } else {
            // Render all messages directly
            const fragment = document.createDocumentFragment();
            messages.forEach(msg => {
                const element = renderMessage(msg);
                renderedMessages.set(msg.id, element);
                fragment.appendChild(element);
            });

            // Insert before indicators
            if (typingIndicator) {
                messagesContainer.insertBefore(fragment, typingIndicator);
            } else {
                messagesContainer.appendChild(fragment);
            }
        }

        scrollToBottom(false);
    }

    /**
     * Setup virtualization for large message lists
     */
    function setupVirtualization() {
        // For now, render the last N messages
        // Full virtualization would require more complex DOM management
        const visibleCount = Math.min(messages.length, CONFIG.VIRTUALIZATION_THRESHOLD);
        const startIndex = messages.length - visibleCount;

        const fragment = document.createDocumentFragment();
        for (let i = startIndex; i < messages.length; i++) {
            const msg = messages[i];
            const element = renderMessage(msg);
            renderedMessages.set(msg.id, element);
            fragment.appendChild(element);
        }

        if (typingIndicator) {
            messagesContainer.insertBefore(fragment, typingIndicator);
        } else {
            messagesContainer.appendChild(fragment);
        }
    }

    /**
     * Update virtualization on scroll
     */
    function updateVirtualization() {
        // Simplified virtualization - show recent messages
        // A full implementation would calculate visible range and render accordingly
        // This is a placeholder for the virtualization logic
    }

    /**
     * Format timestamp for display
     */
    function formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;

        // Within last hour - show relative time
        if (diff < 3600000) {
            const minutes = Math.floor(diff / 60000);
            if (minutes < 1) return 'Just now';
            return `${minutes}m ago`;
        }

        // Today - show time only
        if (date.toDateString() === now.toDateString()) {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }

        // Yesterday
        const yesterday = new Date(now);
        yesterday.setDate(yesterday.getDate() - 1);
        if (date.toDateString() === yesterday.toDateString()) {
            return 'Yesterday ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }

        // Older - show date and time
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' }) +
               ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    /**
     * Escape HTML to prevent XSS
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Clear all messages
     */
    function clearMessages() {
        messages = [];
        renderedMessages.forEach(el => el.remove());
        renderedMessages.clear();
        streamRenderers.clear();

        if (typeof DRXPersistence !== 'undefined') {
            DRXPersistence.clearMessages();
        }
    }

    /**
     * Get all messages
     */
    function getMessages() {
        return [...messages];
    }

    /**
     * Get message by ID
     */
    function getMessage(id) {
        return messages.find(m => m.id === id);
    }

    /**
     * Generate unique message ID
     */
    function generateId() {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Public API
    return {
        // Messages
        addMessage,
        updateMessage,
        removeMessage,
        getMessage,
        getMessages,
        clearMessages,

        // Streaming
        startStream,

        // Scroll
        scrollToBottom,
        scrollToMessage,

        // Status
        setTyping,
        setWaiting,

        // Utils
        generateId,
        init,
    };
})();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DRXChat;
}
