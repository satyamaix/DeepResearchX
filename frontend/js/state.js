/**
 * DRX State Management
 * Handles application state with localStorage persistence
 */

const DRXState = (() => {
    const STORAGE_KEY = 'drx_state';
    const HISTORY_KEY = 'drx_history';
    const MAX_HISTORY = 50;

    // Default state
    const defaultState = {
        // Model settings
        model: 'google/gemini-2.5-flash-preview',
        searchModel: 'google/gemini-3-flash-preview:online',

        // Thinking modes
        thinkingEnabled: true,
        extendedThinking: false,

        // Agent toggles
        agents: {
            planner: true,
            searcher: true,
            reader: true,
            synthesizer: true,
            critic: true,
            reporter: true
        },

        // Theme
        theme: 'dark',

        // Steerability
        steerability: {
            maxIterations: 5,
            searchDepth: 'balanced',
            citationStyle: 'inline',
            outputFormat: 'markdown',
            language: 'en'
        },

        // Research parameters
        researchParams: {
            qualityThreshold: 0.7,
            maxSources: 20,
            includeAcademic: true,
            includeNews: true,
            includeTechnical: true,
            focusAreas: []
        },

        // Connection state
        connection: {
            status: 'disconnected',
            lastConnected: null,
            reconnectAttempts: 0
        },

        // Current interaction
        currentInteraction: null,

        // UI state
        ui: {
            leftSidebarOpen: true,
            rightSidebarOpen: true,
            activeTab: 'models'
        }
    };

    let state = { ...defaultState };
    let listeners = new Map();
    let listenerIdCounter = 0;

    /**
     * Load state from localStorage
     */
    function load() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                const parsed = JSON.parse(saved);
                state = deepMerge(defaultState, parsed);
            }
        } catch (e) {
            console.warn('Failed to load state from localStorage:', e);
            state = { ...defaultState };
        }
        return state;
    }

    /**
     * Save state to localStorage
     */
    function save() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        } catch (e) {
            console.warn('Failed to save state to localStorage:', e);
        }
    }

    /**
     * Deep merge two objects
     */
    function deepMerge(target, source) {
        const result = { ...target };
        for (const key in source) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                result[key] = deepMerge(target[key] || {}, source[key]);
            } else {
                result[key] = source[key];
            }
        }
        return result;
    }

    /**
     * Get current state or a specific path
     */
    function get(path = null) {
        if (!path) return { ...state };

        const parts = path.split('.');
        let value = state;
        for (const part of parts) {
            if (value === undefined) return undefined;
            value = value[part];
        }
        return value;
    }

    /**
     * Set state value at path
     */
    function set(path, value) {
        const parts = path.split('.');
        let current = state;

        for (let i = 0; i < parts.length - 1; i++) {
            if (current[parts[i]] === undefined) {
                current[parts[i]] = {};
            }
            current = current[parts[i]];
        }

        const lastPart = parts[parts.length - 1];
        const oldValue = current[lastPart];
        current[lastPart] = value;

        save();
        notifyListeners(path, value, oldValue);

        return value;
    }

    /**
     * Update multiple values at once
     */
    function update(updates) {
        for (const [path, value] of Object.entries(updates)) {
            set(path, value);
        }
    }

    /**
     * Subscribe to state changes
     */
    function subscribe(pathOrCallback, callback = null) {
        const id = ++listenerIdCounter;

        if (typeof pathOrCallback === 'function') {
            // Global listener
            listeners.set(id, { path: '*', callback: pathOrCallback });
        } else {
            // Path-specific listener
            listeners.set(id, { path: pathOrCallback, callback });
        }

        // Return unsubscribe function
        return () => listeners.delete(id);
    }

    /**
     * Notify listeners of state change
     */
    function notifyListeners(path, newValue, oldValue) {
        for (const [, listener] of listeners) {
            if (listener.path === '*' || path.startsWith(listener.path)) {
                try {
                    listener.callback(path, newValue, oldValue);
                } catch (e) {
                    console.error('State listener error:', e);
                }
            }
        }
    }

    /**
     * Reset state to defaults
     */
    function reset() {
        state = { ...defaultState };
        save();
        notifyListeners('*', state, null);
    }

    // History management

    /**
     * Get research history
     */
    function getHistory() {
        try {
            const saved = localStorage.getItem(HISTORY_KEY);
            return saved ? JSON.parse(saved) : [];
        } catch (e) {
            console.warn('Failed to load history:', e);
            return [];
        }
    }

    /**
     * Add item to history
     */
    function addToHistory(item) {
        try {
            const history = getHistory();

            // Add timestamp if not present
            if (!item.timestamp) {
                item.timestamp = new Date().toISOString();
            }

            // Add to beginning
            history.unshift(item);

            // Limit history size
            if (history.length > MAX_HISTORY) {
                history.pop();
            }

            localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
            notifyListeners('history', history, null);

            return history;
        } catch (e) {
            console.warn('Failed to save to history:', e);
            return [];
        }
    }

    /**
     * Clear history
     */
    function clearHistory() {
        try {
            localStorage.removeItem(HISTORY_KEY);
            notifyListeners('history', [], null);
        } catch (e) {
            console.warn('Failed to clear history:', e);
        }
    }

    /**
     * Remove item from history
     */
    function removeFromHistory(id) {
        try {
            const history = getHistory();
            const filtered = history.filter(item => item.id !== id);
            localStorage.setItem(HISTORY_KEY, JSON.stringify(filtered));
            notifyListeners('history', filtered, history);
            return filtered;
        } catch (e) {
            console.warn('Failed to remove from history:', e);
            return [];
        }
    }

    // Initialize on load
    load();

    return {
        get,
        set,
        update,
        subscribe,
        reset,
        load,
        save,
        getHistory,
        addToHistory,
        clearHistory,
        removeFromHistory,
        defaults: defaultState
    };
})();

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DRXState;
}
