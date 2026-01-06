/**
 * DRX API Client
 * Handles API communication with retry logic and error handling
 */

const DRXApi = (() => {
    const DEFAULT_BASE_URL = '/api/v1';
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 1000;
    const RETRY_BACKOFF = 2;

    let baseUrl = DEFAULT_BASE_URL;
    let requestInterceptors = [];
    let responseInterceptors = [];

    /**
     * Configure the API client
     */
    function configure(options = {}) {
        if (options.baseUrl) {
            baseUrl = options.baseUrl;
        }
    }

    /**
     * Add request interceptor
     */
    function addRequestInterceptor(fn) {
        requestInterceptors.push(fn);
        return () => {
            requestInterceptors = requestInterceptors.filter(f => f !== fn);
        };
    }

    /**
     * Add response interceptor
     */
    function addResponseInterceptor(fn) {
        responseInterceptors.push(fn);
        return () => {
            responseInterceptors = responseInterceptors.filter(f => f !== fn);
        };
    }

    /**
     * Sleep for specified milliseconds
     */
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Check if error is retryable
     */
    function isRetryable(error, status) {
        // Network errors
        if (error instanceof TypeError && error.message.includes('fetch')) {
            return true;
        }

        // Server errors (5xx) and rate limiting (429)
        if (status && (status >= 500 || status === 429)) {
            return true;
        }

        return false;
    }

    /**
     * Make HTTP request with retry logic
     */
    async function request(endpoint, options = {}) {
        const url = `${baseUrl}${endpoint}`;
        let lastError = null;
        let retries = options.retries ?? MAX_RETRIES;

        // Apply request interceptors
        let finalOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        for (const interceptor of requestInterceptors) {
            finalOptions = await interceptor(finalOptions) || finalOptions;
        }

        for (let attempt = 0; attempt <= retries; attempt++) {
            try {
                const response = await fetch(url, finalOptions);

                // Apply response interceptors
                let processedResponse = response;
                for (const interceptor of responseInterceptors) {
                    processedResponse = await interceptor(processedResponse) || processedResponse;
                }

                if (!processedResponse.ok) {
                    const error = new Error(`HTTP ${processedResponse.status}: ${processedResponse.statusText}`);
                    error.status = processedResponse.status;
                    error.response = processedResponse;

                    // Try to parse error body
                    try {
                        error.body = await processedResponse.json();
                    } catch {
                        // Ignore parse errors
                    }

                    // Retry if appropriate
                    if (attempt < retries && isRetryable(error, processedResponse.status)) {
                        const delay = RETRY_DELAY * Math.pow(RETRY_BACKOFF, attempt);
                        console.warn(`Request failed, retrying in ${delay}ms (attempt ${attempt + 1}/${retries})`);
                        await sleep(delay);
                        lastError = error;
                        continue;
                    }

                    throw error;
                }

                // Parse response
                const contentType = processedResponse.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    return await processedResponse.json();
                }

                return processedResponse;

            } catch (error) {
                lastError = error;

                // Retry on network errors
                if (attempt < retries && isRetryable(error, null)) {
                    const delay = RETRY_DELAY * Math.pow(RETRY_BACKOFF, attempt);
                    console.warn(`Request failed, retrying in ${delay}ms (attempt ${attempt + 1}/${retries})`);
                    await sleep(delay);
                    continue;
                }

                throw error;
            }
        }

        throw lastError;
    }

    /**
     * GET request
     */
    function get(endpoint, options = {}) {
        return request(endpoint, { method: 'GET', ...options });
    }

    /**
     * POST request
     */
    function post(endpoint, data, options = {}) {
        return request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data),
            ...options
        });
    }

    /**
     * PUT request
     */
    function put(endpoint, data, options = {}) {
        return request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data),
            ...options
        });
    }

    /**
     * DELETE request
     */
    function del(endpoint, options = {}) {
        return request(endpoint, { method: 'DELETE', ...options });
    }

    // DRX-specific API methods

    /**
     * Create new research interaction
     */
    async function createInteraction(input, config = {}) {
        const state = typeof DRXState !== 'undefined' ? DRXState.get() : {};

        const payload = {
            input,
            config: {
                model: config.model || state.model,
                search_model: config.searchModel || state.searchModel,
                max_iterations: config.maxIterations || state.steerability?.maxIterations || 5,
                thinking_enabled: config.thinkingEnabled ?? state.thinkingEnabled ?? true,
                extended_thinking: config.extendedThinking ?? state.extendedThinking ?? false,
                quality_threshold: config.qualityThreshold || state.researchParams?.qualityThreshold || 0.7,
                max_sources: config.maxSources || state.researchParams?.maxSources || 20,
                agents: config.agents || state.agents,
                focus_areas: config.focusAreas || state.researchParams?.focusAreas || [],
                output_format: config.outputFormat || state.steerability?.outputFormat || 'markdown',
                citation_style: config.citationStyle || state.steerability?.citationStyle || 'inline',
                ...config
            }
        };

        return post('/interactions', payload);
    }

    /**
     * Get interaction status
     */
    function getInteraction(interactionId) {
        return get(`/interactions/${interactionId}`);
    }

    /**
     * Get interaction events
     */
    function getInteractionEvents(interactionId, options = {}) {
        const params = new URLSearchParams();
        if (options.after) params.set('after', options.after);
        if (options.limit) params.set('limit', options.limit);

        const query = params.toString();
        return get(`/interactions/${interactionId}/events${query ? '?' + query : ''}`);
    }

    /**
     * Cancel interaction
     */
    function cancelInteraction(interactionId) {
        return post(`/interactions/${interactionId}/cancel`);
    }

    /**
     * Get system health
     */
    function getHealth() {
        return get('/health');
    }

    /**
     * Get system info
     */
    function getInfo() {
        return get('/info');
    }

    /**
     * Get available models
     */
    function getModels() {
        return get('/models').catch(() => {
            // Return defaults if endpoint doesn't exist
            return {
                models: [
                    { id: 'google/gemini-2.5-flash-preview', name: 'Gemini 2.5 Flash' },
                    { id: 'google/gemini-2.5-pro-preview', name: 'Gemini 2.5 Pro' },
                    { id: 'anthropic/claude-sonnet-4', name: 'Claude Sonnet 4' },
                    { id: 'anthropic/claude-opus-4', name: 'Claude Opus 4' },
                    { id: 'openai/gpt-4.1', name: 'GPT-4.1' },
                    { id: 'openai/o3-mini', name: 'OpenAI o3-mini' }
                ],
                searchModels: [
                    { id: 'google/gemini-3-flash-preview:online', name: 'Gemini 3 Flash Online' },
                    { id: 'perplexity/sonar-pro', name: 'Perplexity Sonar Pro' },
                    { id: 'perplexity/sonar', name: 'Perplexity Sonar' }
                ]
            };
        });
    }

    return {
        configure,
        addRequestInterceptor,
        addResponseInterceptor,
        request,
        get,
        post,
        put,
        delete: del,
        // DRX methods
        createInteraction,
        getInteraction,
        getInteractionEvents,
        cancelInteraction,
        getHealth,
        getInfo,
        getModels
    };
})();

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DRXApi;
}
