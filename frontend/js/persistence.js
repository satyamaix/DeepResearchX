/**
 * DRX Persistence Layer
 * Handles localStorage persistence with versioning and migration
 */

const DRXPersistence = (() => {
    const STORAGE_VERSION = 1;
    const KEYS = {
        VERSION: 'drx_version',
        MESSAGES: 'drx_messages',
        CONFIG: 'drx_config',
        SESSION: 'drx_session',
        INTERACTIONS: 'drx_interactions',
    };
    const MAX_MESSAGES = 50;

    // Initialize storage with version check
    function init() {
        const version = getItem(KEYS.VERSION);
        if (version !== STORAGE_VERSION) {
            migrate(version);
            setItem(KEYS.VERSION, STORAGE_VERSION);
        }
    }

    function migrate(fromVersion) {
        // Handle migrations between versions
        if (!fromVersion) {
            // Fresh install, initialize defaults
            setItem(KEYS.MESSAGES, []);
            setItem(KEYS.CONFIG, getDefaultConfig());
            setItem(KEYS.SESSION, null);
            setItem(KEYS.INTERACTIONS, {});
        }
    }

    function getDefaultConfig() {
        return {
            defaultModel: 'google/gemini-3-flash-preview',
            reasoningModel: 'deepseek/deepseek-r1',
            thinkingSummaries: true,
            showAgentTransitions: true,
            tone: 'technical',
            format: 'markdown',
            language: 'en',
            maxIterations: 5,
            maxSources: 20,
            tokenBudget: 500000,
            timeout: 600,
            enableCitations: true,
            enableQualityChecks: true,
            focusAreas: [],
            excludeTopics: [],
            preferredDomains: [],
            darkMode: true,
        };
    }

    // Generic storage helpers
    function setItem(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (e) {
            console.error('[Persistence] Failed to save:', key, e);
            return false;
        }
    }

    function getItem(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.error('[Persistence] Failed to load:', key, e);
            return defaultValue;
        }
    }

    // Chat history methods
    function saveMessage(message) {
        const messages = getMessages();
        messages.push({
            ...message,
            timestamp: message.timestamp || Date.now()
        });
        // Keep only last MAX_MESSAGES
        const trimmed = messages.slice(-MAX_MESSAGES);
        setItem(KEYS.MESSAGES, trimmed);
        return message;
    }

    function getMessages(limit = MAX_MESSAGES) {
        const messages = getItem(KEYS.MESSAGES, []);
        return messages.slice(-limit);
    }

    function updateMessage(id, updates) {
        const messages = getMessages();
        const index = messages.findIndex(m => m.id === id);
        if (index >= 0) {
            messages[index] = { ...messages[index], ...updates };
            setItem(KEYS.MESSAGES, messages);
            return messages[index];
        }
        return null;
    }

    function clearMessages() {
        setItem(KEYS.MESSAGES, []);
    }

    // Configuration methods
    function saveConfig(config) {
        const current = getConfig();
        const merged = { ...current, ...config };
        setItem(KEYS.CONFIG, merged);
        return merged;
    }

    function getConfig() {
        return getItem(KEYS.CONFIG, getDefaultConfig());
    }

    function resetConfig() {
        setItem(KEYS.CONFIG, getDefaultConfig());
        return getDefaultConfig();
    }

    // Session state methods
    function saveSession(session) {
        setItem(KEYS.SESSION, {
            ...session,
            savedAt: Date.now()
        });
    }

    function getSession() {
        return getItem(KEYS.SESSION, null);
    }

    function clearSession() {
        setItem(KEYS.SESSION, null);
    }

    // Interaction recovery methods
    function saveInteraction(id, lastEventId, status = 'active') {
        const interactions = getItem(KEYS.INTERACTIONS, {});
        interactions[id] = {
            lastEventId,
            status,
            updatedAt: Date.now()
        };
        setItem(KEYS.INTERACTIONS, interactions);
    }

    function getInteraction(id) {
        const interactions = getItem(KEYS.INTERACTIONS, {});
        return interactions[id] || null;
    }

    function getPendingInteractions() {
        const interactions = getItem(KEYS.INTERACTIONS, {});
        const pending = [];
        const now = Date.now();
        const MAX_AGE = 24 * 60 * 60 * 1000; // 24 hours

        for (const [id, data] of Object.entries(interactions)) {
            if (data.status === 'active' && (now - data.updatedAt) < MAX_AGE) {
                pending.push({ id, ...data });
            }
        }
        return pending;
    }

    function completeInteraction(id) {
        const interactions = getItem(KEYS.INTERACTIONS, {});
        if (interactions[id]) {
            interactions[id].status = 'completed';
            interactions[id].updatedAt = Date.now();
            setItem(KEYS.INTERACTIONS, interactions);
        }
    }

    function clearOldInteractions() {
        const interactions = getItem(KEYS.INTERACTIONS, {});
        const now = Date.now();
        const MAX_AGE = 24 * 60 * 60 * 1000;

        for (const [id, data] of Object.entries(interactions)) {
            if ((now - data.updatedAt) > MAX_AGE) {
                delete interactions[id];
            }
        }
        setItem(KEYS.INTERACTIONS, interactions);
    }

    // Export/Import configuration
    function exportConfig() {
        return JSON.stringify({
            version: STORAGE_VERSION,
            config: getConfig(),
            exportedAt: new Date().toISOString()
        }, null, 2);
    }

    function importConfig(jsonString) {
        try {
            const data = JSON.parse(jsonString);
            if (data.config) {
                saveConfig(data.config);
                return { success: true, config: data.config };
            }
            return { success: false, error: 'Invalid config format' };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }

    // Initialize on load
    init();

    return {
        // Messages
        saveMessage,
        getMessages,
        updateMessage,
        clearMessages,

        // Config
        saveConfig,
        getConfig,
        resetConfig,
        getDefaultConfig,

        // Session
        saveSession,
        getSession,
        clearSession,

        // Interactions
        saveInteraction,
        getInteraction,
        getPendingInteractions,
        completeInteraction,
        clearOldInteractions,

        // Import/Export
        exportConfig,
        importConfig,

        // Utils
        init,
    };
})();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = DRXPersistence;
}
