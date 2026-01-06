/**
 * DRX Configuration UI
 * Handles import/export and preset configurations
 */

const DRXConfigUI = (() => {
    // Preset configurations
    const PRESETS = {
        quick: {
            maxIterations: 2,
            maxSources: 5,
            tokenBudget: 100000
        },
        deep: {
            maxIterations: 5,
            maxSources: 20,
            tokenBudget: 500000
        },
        comprehensive: {
            maxIterations: 8,
            maxSources: 50,
            tokenBudget: 1000000
        },
        costEfficient: {
            maxIterations: 3,
            maxSources: 10,
            tokenBudget: 200000,
            defaultModel: 'google/gemini-2.5-flash-preview-05-20'
        }
    };

    /**
     * Initialize config UI handlers
     */
    function init() {
        // Export button
        const exportBtn = document.getElementById('export-config-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', handleExport);
        }

        // Import button and file input
        const importBtn = document.getElementById('import-config-btn');
        const fileInput = document.getElementById('import-config-file');

        if (importBtn && fileInput) {
            importBtn.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', handleImport);
        }

        // Preset selector
        const presetSelect = document.getElementById('config-preset');
        if (presetSelect) {
            presetSelect.addEventListener('change', handlePresetChange);
        }
    }

    /**
     * Handle config export
     */
    function handleExport() {
        try {
            const configJSON = DRXPersistence.exportConfig();
            const blob = new Blob([configJSON], { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `drx-config-${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showToast('Configuration exported successfully', 'success');
        } catch (error) {
            console.error('Export failed:', error);
            showToast('Failed to export configuration', 'error');
        }
    }

    /**
     * Handle config import
     */
    function handleImport(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const result = DRXPersistence.importConfig(e.target.result);

                if (result.success) {
                    // Apply the imported configuration to UI
                    applyConfigToUI(result.config);
                    showToast('Configuration imported successfully', 'success');
                } else {
                    showToast(`Import failed: ${result.error}`, 'error');
                }
            } catch (error) {
                console.error('Import failed:', error);
                showToast('Failed to import configuration', 'error');
            }
        };

        reader.onerror = () => {
            showToast('Failed to read file', 'error');
        };

        reader.readAsText(file);

        // Reset file input
        event.target.value = '';
    }

    /**
     * Handle preset selection
     */
    function handlePresetChange(event) {
        const presetKey = event.target.value;
        if (!presetKey || !PRESETS[presetKey]) {
            return;
        }

        const preset = PRESETS[presetKey];

        // Save preset to persistence
        DRXPersistence.saveConfig(preset);

        // Apply to UI
        applyConfigToUI(preset);

        showToast(`Applied ${presetKey} preset`, 'success');

        // Reset select to placeholder
        setTimeout(() => {
            event.target.value = '';
        }, 100);
    }

    /**
     * Apply configuration values to UI elements
     */
    function applyConfigToUI(config) {
        // Model selections
        if (config.defaultModel) {
            const modelSelect = document.getElementById('default-model');
            if (modelSelect) modelSelect.value = config.defaultModel;
        }

        if (config.reasoningModel) {
            const reasoningSelect = document.getElementById('reasoning-model');
            if (reasoningSelect) reasoningSelect.value = config.reasoningModel;
        }

        // Research depth parameters
        if (config.maxIterations !== undefined) {
            const iterationsSlider = document.getElementById('max-iterations');
            const iterationsValue = document.getElementById('max-iterations-value');
            if (iterationsSlider && iterationsValue) {
                iterationsSlider.value = config.maxIterations;
                iterationsValue.textContent = config.maxIterations;
            }
        }

        if (config.maxSources !== undefined) {
            const sourcesSlider = document.getElementById('max-sources');
            const sourcesValue = document.getElementById('max-sources-value');
            if (sourcesSlider && sourcesValue) {
                sourcesSlider.value = config.maxSources;
                sourcesValue.textContent = config.maxSources;
            }
        }

        if (config.tokenBudget !== undefined) {
            const budgetSlider = document.getElementById('token-budget');
            const budgetValue = document.getElementById('token-budget-value');
            if (budgetSlider && budgetValue) {
                budgetSlider.value = config.tokenBudget;
                budgetValue.textContent = formatTokenBudget(config.tokenBudget);
            }
        }

        if (config.timeout !== undefined) {
            const timeoutSlider = document.getElementById('timeout');
            const timeoutValue = document.getElementById('timeout-value');
            if (timeoutSlider && timeoutValue) {
                timeoutSlider.value = config.timeout;
                timeoutValue.textContent = formatTimeout(config.timeout);
            }
        }

        // Output style
        if (config.tone) {
            const toneSelect = document.getElementById('tone');
            if (toneSelect) toneSelect.value = config.tone;
        }

        if (config.format) {
            const formatSelect = document.getElementById('format');
            if (formatSelect) formatSelect.value = config.format;
        }

        if (config.language) {
            const languageSelect = document.getElementById('language');
            if (languageSelect) languageSelect.value = config.language;
        }

        // Checkboxes
        if (config.thinkingSummaries !== undefined) {
            const checkbox = document.getElementById('thinking-summaries');
            if (checkbox) checkbox.checked = config.thinkingSummaries;
        }

        if (config.showAgentTransitions !== undefined) {
            const checkbox = document.getElementById('show-agent-transitions');
            if (checkbox) checkbox.checked = config.showAgentTransitions;
        }

        if (config.enableCitations !== undefined) {
            const checkbox = document.getElementById('enable-citations');
            if (checkbox) checkbox.checked = config.enableCitations;
        }

        if (config.enableQualityChecks !== undefined) {
            const checkbox = document.getElementById('enable-quality-checks');
            if (checkbox) checkbox.checked = config.enableQualityChecks;
        }

        if (config.darkMode !== undefined) {
            const checkbox = document.getElementById('dark-mode');
            if (checkbox) checkbox.checked = config.darkMode;
        }

        // Trigger change events to ensure any dependent logic runs
        document.querySelectorAll('.config-select, .config-slider, input[type="checkbox"]').forEach(el => {
            el.dispatchEvent(new Event('change'));
        });
    }

    /**
     * Format token budget for display
     */
    function formatTokenBudget(value) {
        if (value >= 1000000) {
            return (value / 1000000).toFixed(1) + 'M';
        } else if (value >= 1000) {
            return (value / 1000).toFixed(0) + 'K';
        }
        return value.toString();
    }

    /**
     * Format timeout for display
     */
    function formatTimeout(seconds) {
        const minutes = Math.floor(seconds / 60);
        return `${minutes} min`;
    }

    return {
        init,
        applyConfigToUI,
        PRESETS
    };
})();

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', DRXConfigUI.init);
} else {
    DRXConfigUI.init();
}
