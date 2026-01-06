/**
 * DRX Report Display Optimization - WP-2C
 * Enhanced report modal with TOC, progress tracking, and navigation
 */

const DRXReport = (() => {
    // State
    let isOpen = false;
    let currentContent = '';
    let currentCitations = [];
    let tocEntries = [];
    let activeSection = null;
    let scrollProgress = 0;
    let keyboardNavEnabled = true;
    let lazyLoadObserver = null;

    // DOM References (cached on init)
    let modal = null;
    let modalContent = null;
    let tocSidebar = null;
    let tocContent = null;
    let reportBody = null;
    let progressBar = null;
    let progressFill = null;
    let jumpToTopBtn = null;
    let mobileBackdrop = null;

    // Configuration
    const CONFIG = {
        progressUpdateThrottle: 50,
        scrollThreshold: 200,
        sectionObserverThreshold: 0.3,
        lazyLoadThreshold: 100,
        animationDuration: 300,
    };

    /**
     * Initialize the report module
     */
    function init() {
        modal = document.getElementById('report-modal');
        if (!modal) {
            console.warn('DRXReport: report-modal not found');
            return;
        }

        cacheElements();
        setupEventListeners();
        setupIntersectionObserver();

        console.log('DRXReport: Initialized');
    }

    /**
     * Cache DOM element references
     */
    function cacheElements() {
        modalContent = modal.querySelector('.modal-content');
        tocSidebar = modal.querySelector('.report-toc-sidebar');
        tocContent = modal.querySelector('.report-toc-content');
        reportBody = modal.querySelector('.report-body');
        progressBar = modal.querySelector('.report-progress-bar');
        progressFill = modal.querySelector('.report-progress-fill');
        jumpToTopBtn = modal.querySelector('.jump-to-top');
        mobileBackdrop = modal.querySelector('.toc-mobile-backdrop');
    }

    /**
     * Setup event listeners
     */
    function setupEventListeners() {
        // Scroll progress tracking
        if (reportBody) {
            reportBody.addEventListener('scroll', throttle(handleScroll, CONFIG.progressUpdateThrottle));
        }

        // Jump to top button
        if (jumpToTopBtn) {
            jumpToTopBtn.addEventListener('click', scrollToTop);
        }

        // Keyboard navigation
        document.addEventListener('keydown', handleKeydown);

        // Mobile TOC backdrop
        if (mobileBackdrop) {
            mobileBackdrop.addEventListener('click', closeMobileToc);
        }

        // Download dropdown handling
        document.addEventListener('click', (e) => {
            const dropdown = modal?.querySelector('.download-dropdown');
            if (dropdown && !dropdown.contains(e.target)) {
                dropdown.classList.remove('open');
            }
        });
    }

    /**
     * Setup Intersection Observer for TOC highlighting and lazy loading
     */
    function setupIntersectionObserver() {
        if (!('IntersectionObserver' in window)) return;

        lazyLoadObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const section = entry.target;

                    // Lazy load content if needed
                    if (section.dataset.lazy === 'true') {
                        loadSectionContent(section);
                    }

                    // Update active TOC item
                    const sectionId = section.id;
                    if (sectionId) {
                        setActiveTocItem(sectionId);
                    }
                }
            });
        }, {
            root: reportBody,
            threshold: CONFIG.sectionObserverThreshold,
            rootMargin: '-10% 0px -70% 0px'
        });
    }

    /**
     * Show the report modal with content
     * @param {string} content - Markdown content
     * @param {Array} citations - Optional citations array
     */
    function show(content, citations = []) {
        if (!modal) init();
        if (!modal) return;

        currentContent = content;
        currentCitations = citations;
        isOpen = true;

        // Render content
        renderReport(content, citations);

        // Generate TOC
        const hasHeadings = generateTOC(reportBody);
        modal.classList.toggle('has-toc', hasHeadings);

        // Show modal
        modal.classList.add('active', 'open');
        document.body.style.overflow = 'hidden';

        // Reset scroll and progress
        if (reportBody) {
            reportBody.scrollTop = 0;
        }
        updateProgress();

        // Setup section observers
        observeSections();

        // Focus management for accessibility
        modal.setAttribute('aria-hidden', 'false');
        const firstFocusable = modal.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (firstFocusable) firstFocusable.focus();
    }

    /**
     * Hide the report modal
     */
    function hide() {
        if (!modal) return;

        isOpen = false;
        modal.classList.remove('active', 'open');
        document.body.style.overflow = '';
        modal.setAttribute('aria-hidden', 'true');

        // Cleanup observers
        if (lazyLoadObserver) {
            lazyLoadObserver.disconnect();
        }

        // Reset state
        currentContent = '';
        currentCitations = [];
        tocEntries = [];
        activeSection = null;
    }

    /**
     * Render report content with sections
     * @param {string} content - Markdown content
     * @param {Array} citations - Citations array
     */
    function renderReport(content, citations) {
        if (!reportBody) return;

        // Use DRXMarkdown if available, otherwise basic rendering
        let html;
        if (typeof DRXMarkdown !== 'undefined') {
            html = DRXMarkdown.render(content);
        } else if (typeof DRXRenderer !== 'undefined') {
            html = DRXRenderer.render(content);
        } else {
            html = basicMarkdownRender(content);
        }

        // Wrap sections for collapsibility
        html = wrapSectionsForCollapse(html);

        // Add citations panel if citations exist
        if (citations && citations.length > 0) {
            html += renderCitationsPanel(citations);
        }

        reportBody.innerHTML = `<div class="report-content markdown-content">${html}</div>`;

        // Setup section collapse handlers
        setupSectionCollapseHandlers();

        // Setup citation click handlers
        if (typeof DRXRenderer !== 'undefined' && citations.length > 0) {
            DRXRenderer.setupCitations(reportBody, citations);
        }
    }

    /**
     * Wrap h1/h2 sections for collapse functionality
     * @param {string} html - HTML content
     * @returns {string} - Wrapped HTML
     */
    function wrapSectionsForCollapse(html) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(`<div>${html}</div>`, 'text/html');
        const container = doc.body.firstChild;

        const result = [];
        let currentSection = null;
        let sectionContent = [];

        const flushSection = () => {
            if (currentSection) {
                const wordCount = countWords(sectionContent.join(' '));
                result.push(createSectionHtml(currentSection, sectionContent.join(''), wordCount));
                currentSection = null;
                sectionContent = [];
            }
        };

        Array.from(container.childNodes).forEach(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
                const tagName = node.tagName.toLowerCase();

                if (tagName === 'h1' || tagName === 'h2') {
                    flushSection();
                    currentSection = {
                        id: node.id || generateId(node.textContent),
                        title: node.textContent,
                        level: tagName === 'h1' ? 1 : 2
                    };
                } else if (currentSection) {
                    sectionContent.push(node.outerHTML);
                } else {
                    result.push(node.outerHTML);
                }
            } else if (node.nodeType === Node.TEXT_NODE && node.textContent.trim()) {
                if (currentSection) {
                    sectionContent.push(node.textContent);
                } else {
                    result.push(node.textContent);
                }
            }
        });

        flushSection();
        return result.join('');
    }

    /**
     * Create HTML for a collapsible section
     */
    function createSectionHtml(section, content, wordCount) {
        return `
            <div class="report-section" id="section-${section.id}" data-section-id="${section.id}">
                <div class="report-section-header" onclick="DRXReport.toggleSection('${section.id}')">
                    <span class="section-collapse-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                    </span>
                    <h${section.level + 1} class="report-section-title">${escapeHtml(section.title)}</h${section.level + 1}>
                    <span class="section-word-count">${wordCount} words</span>
                </div>
                <div class="report-section-content">
                    ${content}
                </div>
            </div>
        `;
    }

    /**
     * Setup section collapse handlers
     */
    function setupSectionCollapseHandlers() {
        // Handlers are set via onclick in the HTML
        // Additional setup can be done here if needed
    }

    /**
     * Toggle section collapse state
     * @param {string} sectionId - Section ID
     */
    function toggleSection(sectionId) {
        const section = reportBody?.querySelector(`[data-section-id="${sectionId}"]`);
        if (!section) return;

        section.classList.toggle('collapsed');

        // Update aria attributes
        const content = section.querySelector('.report-section-content');
        const isCollapsed = section.classList.contains('collapsed');
        content?.setAttribute('aria-hidden', isCollapsed.toString());
    }

    /**
     * Expand all sections
     */
    function expandAll() {
        const sections = reportBody?.querySelectorAll('.report-section.collapsed');
        sections?.forEach(section => section.classList.remove('collapsed'));
    }

    /**
     * Collapse all sections
     */
    function collapseAll() {
        const sections = reportBody?.querySelectorAll('.report-section:not(.collapsed)');
        sections?.forEach(section => section.classList.add('collapsed'));
    }

    /**
     * Generate table of contents from headings
     * @param {HTMLElement} container - Container to scan for headings
     * @returns {boolean} - Whether TOC was generated
     */
    function generateTOC(container) {
        if (!container || !tocContent) return false;

        tocEntries = [];
        const headings = container.querySelectorAll('h1, h2, h3, .report-section-title');

        if (headings.length === 0) return false;

        const tocList = document.createElement('ul');
        tocList.className = 'toc-list';

        headings.forEach((heading, index) => {
            // Skip if inside collapsed content initially
            const section = heading.closest('.report-section');
            const sectionId = section?.dataset.sectionId || heading.id || `heading-${index}`;

            // Ensure heading has an ID
            if (!heading.id && !section) {
                heading.id = sectionId;
            }

            const level = getHeadingLevel(heading);
            const text = heading.textContent.trim();

            tocEntries.push({
                id: sectionId,
                text: text,
                level: level,
                element: section || heading
            });

            const li = document.createElement('li');
            li.className = 'toc-item';
            li.dataset.level = level;

            const link = document.createElement('a');
            link.className = 'toc-link';
            link.href = `#${sectionId}`;
            link.textContent = text;
            link.addEventListener('click', (e) => {
                e.preventDefault();
                scrollToSection(sectionId);
                closeMobileToc();
            });

            li.appendChild(link);
            tocList.appendChild(li);
        });

        tocContent.innerHTML = '';
        tocContent.appendChild(tocList);

        return tocEntries.length > 0;
    }

    /**
     * Get heading level from element
     */
    function getHeadingLevel(element) {
        if (element.classList.contains('report-section-title')) {
            return element.closest('.report-section')?.dataset.level || 1;
        }
        const match = element.tagName.match(/H(\d)/);
        return match ? parseInt(match[1], 10) : 1;
    }

    /**
     * Scroll to a specific section
     * @param {string} id - Section ID
     */
    function scrollToSection(id) {
        if (!reportBody) return;

        // Try to find section wrapper first, then heading
        let target = reportBody.querySelector(`[data-section-id="${id}"]`) ||
                     reportBody.querySelector(`#${id}`) ||
                     reportBody.querySelector(`#section-${id}`);

        if (!target) return;

        // Expand section if collapsed
        const section = target.closest('.report-section');
        if (section?.classList.contains('collapsed')) {
            section.classList.remove('collapsed');
        }

        // Smooth scroll
        const targetTop = target.offsetTop - 20;
        reportBody.scrollTo({
            top: targetTop,
            behavior: 'smooth'
        });

        // Highlight briefly
        target.classList.add('highlighted');
        setTimeout(() => target.classList.remove('highlighted'), 1000);

        // Update active TOC
        setActiveTocItem(id);
    }

    /**
     * Set active TOC item
     * @param {string} id - Section ID
     */
    function setActiveTocItem(id) {
        if (!tocContent) return;

        activeSection = id;

        // Remove active from all
        tocContent.querySelectorAll('.toc-link.active').forEach(link => {
            link.classList.remove('active');
        });

        // Add active to current
        const activeLink = tocContent.querySelector(`a[href="#${id}"]`);
        if (activeLink) {
            activeLink.classList.add('active');

            // Scroll TOC to show active item
            const linkRect = activeLink.getBoundingClientRect();
            const tocRect = tocContent.getBoundingClientRect();

            if (linkRect.top < tocRect.top || linkRect.bottom > tocRect.bottom) {
                activeLink.scrollIntoView({ block: 'center', behavior: 'smooth' });
            }
        }
    }

    /**
     * Observe sections for intersection
     */
    function observeSections() {
        if (!lazyLoadObserver || !reportBody) return;

        const sections = reportBody.querySelectorAll('.report-section, h1, h2, h3');
        sections.forEach(section => {
            lazyLoadObserver.observe(section);
        });
    }

    /**
     * Load lazy section content
     */
    function loadSectionContent(section) {
        const placeholder = section.querySelector('.section-lazy-placeholder');
        if (placeholder) {
            // Content already loaded or loading
            placeholder.remove();
        }
        section.dataset.lazy = 'false';
    }

    /**
     * Update reading progress
     */
    function updateProgress() {
        if (!reportBody || !progressFill) return;

        const scrollTop = reportBody.scrollTop;
        const scrollHeight = reportBody.scrollHeight - reportBody.clientHeight;

        if (scrollHeight <= 0) {
            scrollProgress = 100;
        } else {
            scrollProgress = Math.round((scrollTop / scrollHeight) * 100);
        }

        progressFill.style.width = `${scrollProgress}%`;

        // Update jump to top visibility
        if (jumpToTopBtn) {
            jumpToTopBtn.classList.toggle('visible', scrollTop > CONFIG.scrollThreshold);
        }
    }

    /**
     * Handle scroll event
     */
    function handleScroll() {
        updateProgress();
        updateActiveSectionFromScroll();
    }

    /**
     * Update active section based on scroll position
     */
    function updateActiveSectionFromScroll() {
        if (!reportBody || tocEntries.length === 0) return;

        const scrollTop = reportBody.scrollTop;
        const offset = reportBody.clientHeight * 0.3;

        for (let i = tocEntries.length - 1; i >= 0; i--) {
            const entry = tocEntries[i];
            const element = entry.element;

            if (element && element.offsetTop <= scrollTop + offset) {
                if (activeSection !== entry.id) {
                    setActiveTocItem(entry.id);
                }
                break;
            }
        }
    }

    /**
     * Scroll to top of report
     */
    function scrollToTop() {
        if (!reportBody) return;
        reportBody.scrollTo({ top: 0, behavior: 'smooth' });
    }

    /**
     * Handle keyboard navigation
     */
    function handleKeydown(e) {
        if (!isOpen || !keyboardNavEnabled) return;

        switch (e.key) {
            case 'Escape':
                e.preventDefault();
                hide();
                break;

            case 'ArrowUp':
                if (e.altKey) {
                    e.preventDefault();
                    navigateToPreviousSection();
                }
                break;

            case 'ArrowDown':
                if (e.altKey) {
                    e.preventDefault();
                    navigateToNextSection();
                }
                break;

            case 'Home':
                if (e.ctrlKey) {
                    e.preventDefault();
                    scrollToTop();
                }
                break;

            case 'End':
                if (e.ctrlKey && reportBody) {
                    e.preventDefault();
                    reportBody.scrollTo({
                        top: reportBody.scrollHeight,
                        behavior: 'smooth'
                    });
                }
                break;
        }
    }

    /**
     * Navigate to previous section
     */
    function navigateToPreviousSection() {
        if (tocEntries.length === 0) return;

        const currentIndex = tocEntries.findIndex(e => e.id === activeSection);
        const prevIndex = Math.max(0, currentIndex - 1);

        if (tocEntries[prevIndex]) {
            scrollToSection(tocEntries[prevIndex].id);
        }
    }

    /**
     * Navigate to next section
     */
    function navigateToNextSection() {
        if (tocEntries.length === 0) return;

        const currentIndex = tocEntries.findIndex(e => e.id === activeSection);
        const nextIndex = Math.min(tocEntries.length - 1, currentIndex + 1);

        if (tocEntries[nextIndex]) {
            scrollToSection(tocEntries[nextIndex].id);
        }
    }

    /**
     * Toggle mobile TOC
     */
    function toggleMobileToc() {
        if (!tocSidebar) return;

        const isOpen = tocSidebar.classList.toggle('mobile-open');
        if (mobileBackdrop) {
            mobileBackdrop.classList.toggle('visible', isOpen);
        }
    }

    /**
     * Close mobile TOC
     */
    function closeMobileToc() {
        if (!tocSidebar) return;

        tocSidebar.classList.remove('mobile-open');
        if (mobileBackdrop) {
            mobileBackdrop.classList.remove('visible');
        }
    }

    /**
     * Toggle download dropdown
     */
    function toggleDownloadDropdown() {
        const dropdown = modal?.querySelector('.download-dropdown');
        if (dropdown) {
            dropdown.classList.toggle('open');
        }
    }

    /**
     * Copy report to clipboard
     * @returns {Promise<void>}
     */
    async function copy() {
        if (!currentContent) return;

        try {
            await navigator.clipboard.writeText(currentContent);
            showToast('Report copied to clipboard', 'success');
        } catch (err) {
            console.error('Failed to copy:', err);
            showToast('Failed to copy report', 'error');
        }
    }

    /**
     * Download report in specified format
     * @param {string} format - 'md', 'html', or 'pdf'
     */
    function download(format = 'md') {
        if (!currentContent) return;

        const timestamp = new Date().toISOString().split('T')[0];
        const filename = `drx-report-${timestamp}`;

        switch (format) {
            case 'md':
                downloadAsMarkdown(filename);
                break;
            case 'html':
                downloadAsHtml(filename);
                break;
            case 'pdf':
                downloadAsPdf();
                break;
            default:
                downloadAsMarkdown(filename);
        }

        // Close dropdown
        const dropdown = modal?.querySelector('.download-dropdown');
        if (dropdown) dropdown.classList.remove('open');
    }

    /**
     * Download as Markdown
     */
    function downloadAsMarkdown(filename) {
        const blob = new Blob([currentContent], { type: 'text/markdown' });
        triggerDownload(blob, `${filename}.md`);
    }

    /**
     * Download as HTML
     */
    function downloadAsHtml(filename) {
        const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DRX Research Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 800px; margin: 0 auto; padding: 40px 20px; line-height: 1.6; }
        h1, h2, h3 { margin-top: 2em; margin-bottom: 0.5em; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
        pre { background: #f4f4f4; padding: 16px; border-radius: 8px; overflow-x: auto; }
        pre code { background: none; padding: 0; }
        blockquote { border-left: 4px solid #ddd; margin: 0; padding-left: 16px; color: #666; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
        th { background: #f4f4f4; }
        a { color: #0066cc; }
        .citation { color: #0066cc; cursor: pointer; }
    </style>
</head>
<body>
${typeof DRXMarkdown !== 'undefined' ? DRXMarkdown.render(currentContent) : currentContent}
</body>
</html>`;

        const blob = new Blob([html], { type: 'text/html' });
        triggerDownload(blob, `${filename}.html`);
    }

    /**
     * Download as PDF (uses browser print)
     */
    function downloadAsPdf() {
        // Create a new window with print-optimized styles
        const printWindow = window.open('', '_blank');
        if (!printWindow) {
            showToast('Please allow popups to download PDF', 'warning');
            return;
        }

        const content = reportBody?.innerHTML || '';

        printWindow.document.write(`<!DOCTYPE html>
<html>
<head>
    <title>DRX Research Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; line-height: 1.6; }
        h1, h2, h3 { margin-top: 1.5em; margin-bottom: 0.5em; }
        .report-section { border: none !important; margin: 0 !important; }
        .report-section-header { display: none; }
        .report-section-content { max-height: none !important; opacity: 1 !important; padding: 0 !important; }
        @media print { body { padding: 0; } }
    </style>
</head>
<body>${content}</body>
</html>`);

        printWindow.document.close();

        setTimeout(() => {
            printWindow.print();
            printWindow.close();
        }, 250);
    }

    /**
     * Trigger file download
     */
    function triggerDownload(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast(`Downloaded ${filename}`, 'success');
    }

    /**
     * Render citations panel
     */
    function renderCitationsPanel(citations) {
        if (!citations || citations.length === 0) return '';

        const items = citations.map((citation, index) => `
            <div class="citation-item">
                <span class="citation-number">${index + 1}</span>
                <div class="citation-details">
                    <div class="citation-title">${escapeHtml(citation.title || 'Untitled')}</div>
                    <a class="citation-url" href="${escapeHtml(citation.url)}" target="_blank" rel="noopener">${escapeHtml(citation.url)}</a>
                </div>
            </div>
        `).join('');

        return `
            <div class="report-citations-panel">
                <h4 class="report-citations-title">References</h4>
                <div class="citation-list">${items}</div>
            </div>
        `;
    }

    // ========================================================================
    // Utility Functions
    // ========================================================================

    /**
     * Throttle function
     */
    function throttle(fn, delay) {
        let lastCall = 0;
        return function(...args) {
            const now = Date.now();
            if (now - lastCall >= delay) {
                lastCall = now;
                fn.apply(this, args);
            }
        };
    }

    /**
     * Escape HTML entities
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Generate ID from text
     */
    function generateId(text) {
        return text
            .toLowerCase()
            .replace(/[^\w\s-]/g, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-')
            .substring(0, 50);
    }

    /**
     * Count words in text
     */
    function countWords(text) {
        return text
            .replace(/<[^>]*>/g, '')
            .split(/\s+/)
            .filter(word => word.length > 0)
            .length;
    }

    /**
     * Basic markdown render fallback
     */
    function basicMarkdownRender(markdown) {
        return markdown
            .replace(/^### (.+)$/gm, '<h3>$1</h3>')
            .replace(/^## (.+)$/gm, '<h2>$1</h2>')
            .replace(/^# (.+)$/gm, '<h1>$1</h1>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
            .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/^(.+)$/gm, (match) => {
                if (match.startsWith('<')) return match;
                return `<p>${match}</p>`;
            });
    }

    /**
     * Show toast notification
     */
    function showToast(message, type = 'info') {
        // Use existing toast system if available
        if (typeof window.showToast === 'function') {
            window.showToast(message, type);
            return;
        }

        // Fallback toast implementation
        const container = document.getElementById('toast-container') || createToastContainer();
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-message">${escapeHtml(message)}</span>
            <button class="toast-close" onclick="this.parentElement.remove()">&times;</button>
        `;
        container.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    /**
     * Create toast container if not exists
     */
    function createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
        return container;
    }

    /**
     * Get current reading stats
     */
    function getReadingStats() {
        const wordCount = countWords(currentContent);
        const readingTime = Math.ceil(wordCount / 200); // 200 wpm average
        const sectionCount = tocEntries.length;

        return {
            wordCount,
            readingTime,
            sectionCount,
            progress: scrollProgress
        };
    }

    // ========================================================================
    // Public API
    // ========================================================================

    return {
        // Lifecycle
        init,

        // Display
        show,
        hide,

        // Navigation
        generateTOC,
        scrollToSection,
        scrollToTop,
        navigateToPreviousSection,
        navigateToNextSection,

        // Progress
        updateProgress,
        getReadingStats,

        // Sections
        toggleSection,
        expandAll,
        collapseAll,

        // Actions
        copy,
        download,
        toggleDownloadDropdown,

        // Mobile
        toggleMobileToc,
        closeMobileToc,

        // State getters
        get isOpen() { return isOpen; },
        get progress() { return scrollProgress; },
        get tocEntries() { return [...tocEntries]; },
    };
})();

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', DRXReport.init);
} else {
    DRXReport.init();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DRXReport;
}
