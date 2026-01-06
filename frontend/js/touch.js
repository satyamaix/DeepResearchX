/**
 * DRX Touch Gestures Module
 * Handles touch gestures for mobile interactions:
 * - Edge swipes to open/close sidebars
 * - Long-press for context menus
 */

const DRXTouch = (() => {
    // Configuration
    const config = {
        edgeThreshold: 30,      // px from edge to detect edge swipe
        minSwipeDistance: 50,   // minimum distance for swipe detection
        minSwipeVelocity: 0.3,  // minimum velocity (px/ms)
        longPressDuration: 500, // ms for long press
        longPressThreshold: 10  // max movement allowed during long press
    };

    // State
    let touchState = {
        startX: 0,
        startY: 0,
        startTime: 0,
        currentX: 0,
        currentY: 0,
        isTracking: false,
        isSwiping: false,
        longPressTimer: null,
        longPressTarget: null,
        swipeHandlers: {
            left: [],
            right: []
        }
    };

    /**
     * Initialize touch gesture handling
     */
    function init() {
        console.log('[DRXTouch] Initializing touch gestures...');

        // Only enable on touch devices
        if (!('ontouchstart' in window)) {
            console.log('[DRXTouch] Touch not supported, skipping initialization');
            return;
        }

        // Set up global touch listeners
        document.addEventListener('touchstart', handleTouchStart, { passive: false });
        document.addEventListener('touchmove', handleTouchMove, { passive: false });
        document.addEventListener('touchend', handleTouchEnd, { passive: false });
        document.addEventListener('touchcancel', handleTouchCancel, { passive: false });

        // Set up sidebar-specific listeners
        setupSidebarGestures();

        console.log('[DRXTouch] Touch gestures initialized');
    }

    /**
     * Handle touch start
     */
    function handleTouchStart(e) {
        // Only handle single touch
        if (e.touches.length !== 1) {
            resetTouchState();
            return;
        }

        const touch = e.touches[0];
        touchState.startX = touch.clientX;
        touchState.startY = touch.clientY;
        touchState.currentX = touch.clientX;
        touchState.currentY = touch.clientY;
        touchState.startTime = Date.now();
        touchState.isTracking = true;
        touchState.isSwiping = false;

        // Check if this is an edge swipe
        const isLeftEdge = touch.clientX < config.edgeThreshold;
        const isRightEdge = touch.clientX > window.innerWidth - config.edgeThreshold;

        if (isLeftEdge || isRightEdge) {
            // Potential edge swipe, prevent default to avoid conflicts
            touchState.isSwiping = true;
        }

        // Start long press timer if touching a message
        const target = e.target.closest('.message');
        if (target && target.classList.contains('assistant')) {
            touchState.longPressTarget = target;
            touchState.longPressTimer = setTimeout(() => {
                handleLongPress(target);
            }, config.longPressDuration);
        }
    }

    /**
     * Handle touch move
     */
    function handleTouchMove(e) {
        if (!touchState.isTracking) return;

        const touch = e.touches[0];
        touchState.currentX = touch.clientX;
        touchState.currentY = touch.clientY;

        const deltaX = touchState.currentX - touchState.startX;
        const deltaY = touchState.currentY - touchState.startY;
        const absDeltaX = Math.abs(deltaX);
        const absDeltaY = Math.abs(deltaY);

        // Cancel long press if moved too much
        if (touchState.longPressTimer &&
            (absDeltaX > config.longPressThreshold || absDeltaY > config.longPressThreshold)) {
            clearTimeout(touchState.longPressTimer);
            touchState.longPressTimer = null;
            touchState.longPressTarget = null;
        }

        // Check if this is a horizontal swipe
        if (absDeltaX > absDeltaY && absDeltaX > 10) {
            touchState.isSwiping = true;

            // Prevent default scroll during horizontal swipe
            if (e.cancelable) {
                e.preventDefault();
            }
        }
    }

    /**
     * Handle touch end
     */
    function handleTouchEnd(e) {
        if (!touchState.isTracking) return;

        // Clear long press timer
        if (touchState.longPressTimer) {
            clearTimeout(touchState.longPressTimer);
            touchState.longPressTimer = null;
            touchState.longPressTarget = null;
        }

        // Calculate swipe properties
        const deltaX = touchState.currentX - touchState.startX;
        const deltaY = touchState.currentY - touchState.startY;
        const absDeltaX = Math.abs(deltaX);
        const absDeltaY = Math.abs(deltaY);
        const duration = Date.now() - touchState.startTime;
        const velocity = absDeltaX / duration;

        // Check if this is a valid swipe
        if (touchState.isSwiping &&
            absDeltaX > config.minSwipeDistance &&
            absDeltaX > absDeltaY &&
            velocity > config.minSwipeVelocity) {

            const direction = deltaX > 0 ? 'right' : 'left';
            handleSwipe(direction, touchState.startX, touchState.startY);
        }

        resetTouchState();
    }

    /**
     * Handle touch cancel
     */
    function handleTouchCancel(e) {
        resetTouchState();
    }

    /**
     * Reset touch state
     */
    function resetTouchState() {
        if (touchState.longPressTimer) {
            clearTimeout(touchState.longPressTimer);
        }
        touchState = {
            ...touchState,
            isTracking: false,
            isSwiping: false,
            longPressTimer: null,
            longPressTarget: null
        };
    }

    /**
     * Handle swipe gesture
     */
    function handleSwipe(direction, startX, startY) {
        console.log('[DRXTouch] Swipe detected:', direction, 'from', startX);

        // Handle edge swipes to open sidebars
        if (direction === 'right' && startX < config.edgeThreshold) {
            // Swipe right from left edge - open left sidebar
            const leftSidebar = document.getElementById('sidebar-left');
            if (leftSidebar && leftSidebar.classList.contains('collapsed')) {
                toggleSidebar('left');
                return;
            }
        }

        if (direction === 'left' && startX > window.innerWidth - config.edgeThreshold) {
            // Swipe left from right edge - open right sidebar
            const rightSidebar = document.getElementById('sidebar-right');
            if (rightSidebar && rightSidebar.classList.contains('collapsed')) {
                toggleSidebar('right');
                return;
            }
        }

        // Trigger custom swipe handlers
        const handlers = touchState.swipeHandlers[direction] || [];
        handlers.forEach(handler => {
            try {
                handler(startX, startY);
            } catch (error) {
                console.error('[DRXTouch] Swipe handler error:', error);
            }
        });
    }

    /**
     * Set up sidebar-specific gestures
     */
    function setupSidebarGestures() {
        // Left sidebar - swipe left to close
        const leftSidebar = document.getElementById('sidebar-left');
        if (leftSidebar) {
            leftSidebar.addEventListener('touchstart', handleSidebarTouchStart('left'), { passive: false });
        }

        // Right sidebar - swipe right to close
        const rightSidebar = document.getElementById('sidebar-right');
        if (rightSidebar) {
            rightSidebar.addEventListener('touchstart', handleSidebarTouchStart('right'), { passive: false });
        }
    }

    /**
     * Create sidebar-specific touch handler
     */
    function handleSidebarTouchStart(side) {
        return function(e) {
            const sidebar = e.currentTarget;

            // Only handle if sidebar is open
            if (sidebar.classList.contains('collapsed')) return;

            const touch = e.touches[0];
            const startX = touch.clientX;
            const startY = touch.clientY;
            let currentX = startX;
            let currentY = startY;
            let hasMoved = false;

            const handleMove = (e) => {
                const touch = e.touches[0];
                currentX = touch.clientX;
                currentY = touch.clientY;

                const deltaX = currentX - startX;
                const deltaY = currentY - startY;
                const absDeltaX = Math.abs(deltaX);
                const absDeltaY = Math.abs(deltaY);

                // Check if moving horizontally
                if (absDeltaX > absDeltaY && absDeltaX > 10) {
                    hasMoved = true;
                    if (e.cancelable) {
                        e.preventDefault();
                    }
                }
            };

            const handleEnd = (e) => {
                sidebar.removeEventListener('touchmove', handleMove);
                sidebar.removeEventListener('touchend', handleEnd);
                sidebar.removeEventListener('touchcancel', handleEnd);

                if (!hasMoved) return;

                const deltaX = currentX - startX;
                const absDeltaX = Math.abs(deltaX);

                // Check if swipe is valid
                if (absDeltaX > config.minSwipeDistance) {
                    if (side === 'left' && deltaX < 0) {
                        // Swipe left on left sidebar - close it
                        toggleSidebar('left');
                    } else if (side === 'right' && deltaX > 0) {
                        // Swipe right on right sidebar - close it
                        toggleSidebar('right');
                    }
                }
            };

            sidebar.addEventListener('touchmove', handleMove, { passive: false });
            sidebar.addEventListener('touchend', handleEnd);
            sidebar.addEventListener('touchcancel', handleEnd);
        };
    }

    /**
     * Handle long press
     */
    function handleLongPress(target) {
        console.log('[DRXTouch] Long press detected on message');

        // Add haptic feedback if available
        if (navigator.vibrate) {
            navigator.vibrate(50);
        }

        // Show context menu
        showMessageContextMenu(target);
    }

    /**
     * Show context menu for message
     */
    function showMessageContextMenu(messageElement) {
        // Remove any existing context menu
        const existingMenu = document.querySelector('.message-context-menu');
        if (existingMenu) {
            existingMenu.remove();
        }

        // Create context menu
        const menu = document.createElement('div');
        menu.className = 'message-context-menu';
        menu.innerHTML = `
            <button class="context-menu-item" data-action="copy">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                </svg>
                Copy
            </button>
            <button class="context-menu-item" data-action="retry">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M1 4v6h6M23 20v-6h-6"/>
                    <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15"/>
                </svg>
                Retry
            </button>
        `;

        // Position menu relative to message
        messageElement.style.position = 'relative';
        messageElement.appendChild(menu);

        // Add event listeners
        menu.querySelectorAll('.context-menu-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const action = item.dataset.action;
                handleContextMenuAction(action, messageElement);
                menu.remove();
            });
        });

        // Close menu when clicking outside
        setTimeout(() => {
            const closeMenu = (e) => {
                if (!menu.contains(e.target)) {
                    menu.remove();
                    document.removeEventListener('click', closeMenu);
                    document.removeEventListener('touchstart', closeMenu);
                }
            };
            document.addEventListener('click', closeMenu);
            document.addEventListener('touchstart', closeMenu);
        }, 100);
    }

    /**
     * Handle context menu action
     */
    function handleContextMenuAction(action, messageElement) {
        const messageContent = messageElement.querySelector('.message-content');
        if (!messageContent) return;

        switch (action) {
            case 'copy':
                copyMessageContent(messageContent);
                break;
            case 'retry':
                retryQuery(messageElement);
                break;
        }
    }

    /**
     * Copy message content to clipboard
     */
    async function copyMessageContent(contentElement) {
        try {
            const text = contentElement.textContent || contentElement.innerText;
            await navigator.clipboard.writeText(text);

            // Show toast notification if available
            if (typeof showToast === 'function') {
                showToast('Message copied to clipboard', 'success');
            } else {
                console.log('[DRXTouch] Message copied to clipboard');
            }
        } catch (error) {
            console.error('[DRXTouch] Failed to copy message:', error);
            if (typeof showToast === 'function') {
                showToast('Failed to copy message', 'error');
            }
        }
    }

    /**
     * Retry query from message
     */
    function retryQuery(messageElement) {
        // Find the original user message that triggered this response
        let userMessage = messageElement.previousElementSibling;
        while (userMessage && !userMessage.classList.contains('user')) {
            userMessage = userMessage.previousElementSibling;
        }

        if (userMessage) {
            const query = userMessage.querySelector('.message-content')?.textContent;
            if (query && typeof setQuery === 'function') {
                setQuery(query);
                if (typeof showToast === 'function') {
                    showToast('Query loaded - click send to retry', 'info');
                }
            }
        } else {
            if (typeof showToast === 'function') {
                showToast('Could not find original query', 'warning');
            }
        }
    }

    /**
     * Register custom swipe handler
     */
    function onSwipe(direction, handler) {
        if (!touchState.swipeHandlers[direction]) {
            touchState.swipeHandlers[direction] = [];
        }
        touchState.swipeHandlers[direction].push(handler);
    }

    /**
     * Register custom long press handler
     */
    function onLongPress(element, handler) {
        if (!element) return;

        element.addEventListener('touchstart', (e) => {
            const touch = e.touches[0];
            const startX = touch.clientX;
            const startY = touch.clientY;
            let moved = false;

            const timer = setTimeout(() => {
                if (!moved) {
                    handler(element, e);
                }
            }, config.longPressDuration);

            const moveHandler = (e) => {
                const touch = e.touches[0];
                const deltaX = Math.abs(touch.clientX - startX);
                const deltaY = Math.abs(touch.clientY - startY);

                if (deltaX > config.longPressThreshold || deltaY > config.longPressThreshold) {
                    moved = true;
                    clearTimeout(timer);
                }
            };

            const endHandler = () => {
                clearTimeout(timer);
                element.removeEventListener('touchmove', moveHandler);
                element.removeEventListener('touchend', endHandler);
                element.removeEventListener('touchcancel', endHandler);
            };

            element.addEventListener('touchmove', moveHandler);
            element.addEventListener('touchend', endHandler);
            element.addEventListener('touchcancel', endHandler);
        });
    }

    // Public API
    return {
        init,
        onSwipe,
        onLongPress
    };
})();

// Auto-initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => DRXTouch.init());
} else {
    DRXTouch.init();
}
