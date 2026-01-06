/**
 * DRX Streaming Markdown Renderer
 * Incrementally renders markdown without full re-renders
 */

const DRXRenderer = (() => {
    // Simple markdown patterns
    const PATTERNS = {
        heading: /^(#{1,6})\s+(.+)$/gm,
        bold: /\*\*(.+?)\*\*/g,
        italic: /\*(.+?)\*/g,
        code: /`([^`]+)`/g,
        codeBlock: /```(\w*)\n([\s\S]*?)```/g,
        link: /\[([^\]]+)\]\(([^)]+)\)/g,
        list: /^[-*]\s+(.+)$/gm,
        orderedList: /^(\d+)\.\s+(.+)$/gm,
        blockquote: /^>\s+(.+)$/gm,
        hr: /^---$/gm,
        citation: /\[\^(\d+)\]/g,
    };

    // Code highlighting (basic)
    const KEYWORDS = {
        js: ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'export', 'async', 'await', 'try', 'catch'],
        python: ['def', 'class', 'return', 'if', 'else', 'elif', 'for', 'while', 'import', 'from', 'try', 'except', 'with', 'as', 'async', 'await'],
    };

    class MarkdownStream {
        constructor(container) {
            this.container = container;
            this.buffer = '';
            this.currentBlock = null;
            this.inCodeBlock = false;
            this.codeLanguage = '';
            this.codeBuffer = '';
        }

        append(chunk) {
            this.buffer += chunk;
            this.processBuffer();
        }

        processBuffer() {
            // Check for code block start/end
            if (!this.inCodeBlock) {
                const codeStart = this.buffer.match(/```(\w*)\n/);
                if (codeStart) {
                    // Render everything before code block
                    const beforeCode = this.buffer.substring(0, codeStart.index);
                    if (beforeCode) {
                        this.renderMarkdown(beforeCode);
                    }
                    this.inCodeBlock = true;
                    this.codeLanguage = codeStart[1] || 'text';
                    this.codeBuffer = '';
                    this.buffer = this.buffer.substring(codeStart.index + codeStart[0].length);

                    // Create code block element
                    this.currentBlock = document.createElement('div');
                    this.currentBlock.className = 'code-block';
                    this.currentBlock.innerHTML = `
                        <div class="code-header">
                            <span class="code-language">${this.codeLanguage}</span>
                            <button class="code-copy-btn" onclick="DRXRenderer.copyCode(this)">Copy</button>
                        </div>
                        <pre><code class="language-${this.codeLanguage}"></code></pre>
                    `;
                    this.container.appendChild(this.currentBlock);
                }
            }

            if (this.inCodeBlock) {
                const codeEnd = this.buffer.indexOf('```');
                if (codeEnd !== -1) {
                    // Code block ended
                    this.codeBuffer += this.buffer.substring(0, codeEnd);
                    const codeEl = this.currentBlock.querySelector('code');
                    codeEl.textContent = this.codeBuffer;
                    this.highlightCode(codeEl, this.codeLanguage);

                    this.inCodeBlock = false;
                    this.currentBlock = null;
                    this.buffer = this.buffer.substring(codeEnd + 3);

                    // Continue processing
                    this.processBuffer();
                } else {
                    // Still in code block, append to code
                    this.codeBuffer += this.buffer;
                    const codeEl = this.currentBlock.querySelector('code');
                    codeEl.textContent = this.codeBuffer;
                    this.buffer = '';
                }
            } else {
                // Look for complete lines to render
                const lines = this.buffer.split('\n');
                if (lines.length > 1) {
                    // Render all complete lines
                    const completeLines = lines.slice(0, -1).join('\n');
                    this.renderMarkdown(completeLines + '\n');
                    this.buffer = lines[lines.length - 1];
                }
            }
        }

        renderMarkdown(text) {
            // Process block-level elements
            const blocks = this.parseBlocks(text);

            for (const block of blocks) {
                const el = this.createBlockElement(block);
                if (el) {
                    this.container.appendChild(el);
                }
            }

            // Scroll to bottom
            this.container.scrollTop = this.container.scrollHeight;
        }

        parseBlocks(text) {
            const blocks = [];
            const lines = text.split('\n');
            let currentParagraph = [];

            for (const line of lines) {
                const trimmed = line.trim();

                if (!trimmed) {
                    if (currentParagraph.length > 0) {
                        blocks.push({ type: 'paragraph', content: currentParagraph.join(' ') });
                        currentParagraph = [];
                    }
                    continue;
                }

                // Check for special blocks
                if (trimmed.match(/^#{1,6}\s/)) {
                    if (currentParagraph.length > 0) {
                        blocks.push({ type: 'paragraph', content: currentParagraph.join(' ') });
                        currentParagraph = [];
                    }
                    const match = trimmed.match(/^(#{1,6})\s+(.+)$/);
                    blocks.push({ type: 'heading', level: match[1].length, content: match[2] });
                } else if (trimmed.match(/^[-*]\s/)) {
                    if (currentParagraph.length > 0) {
                        blocks.push({ type: 'paragraph', content: currentParagraph.join(' ') });
                        currentParagraph = [];
                    }
                    blocks.push({ type: 'listItem', content: trimmed.replace(/^[-*]\s+/, '') });
                } else if (trimmed.match(/^\d+\.\s/)) {
                    if (currentParagraph.length > 0) {
                        blocks.push({ type: 'paragraph', content: currentParagraph.join(' ') });
                        currentParagraph = [];
                    }
                    blocks.push({ type: 'orderedListItem', content: trimmed.replace(/^\d+\.\s+/, '') });
                } else if (trimmed.match(/^>\s/)) {
                    if (currentParagraph.length > 0) {
                        blocks.push({ type: 'paragraph', content: currentParagraph.join(' ') });
                        currentParagraph = [];
                    }
                    blocks.push({ type: 'blockquote', content: trimmed.replace(/^>\s+/, '') });
                } else if (trimmed === '---') {
                    if (currentParagraph.length > 0) {
                        blocks.push({ type: 'paragraph', content: currentParagraph.join(' ') });
                        currentParagraph = [];
                    }
                    blocks.push({ type: 'hr' });
                } else {
                    currentParagraph.push(trimmed);
                }
            }

            if (currentParagraph.length > 0) {
                blocks.push({ type: 'paragraph', content: currentParagraph.join(' ') });
            }

            return blocks;
        }

        createBlockElement(block) {
            let el;

            switch (block.type) {
                case 'heading':
                    el = document.createElement(`h${block.level}`);
                    el.innerHTML = this.renderInline(block.content);
                    break;
                case 'paragraph':
                    el = document.createElement('p');
                    el.innerHTML = this.renderInline(block.content);
                    break;
                case 'listItem':
                    el = document.createElement('li');
                    el.innerHTML = this.renderInline(block.content);
                    // Wrap in ul if needed
                    const lastChild = this.container.lastElementChild;
                    if (lastChild && lastChild.tagName === 'UL') {
                        lastChild.appendChild(el);
                        return null;
                    } else {
                        const ul = document.createElement('ul');
                        ul.appendChild(el);
                        return ul;
                    }
                case 'orderedListItem':
                    el = document.createElement('li');
                    el.innerHTML = this.renderInline(block.content);
                    const lastOl = this.container.lastElementChild;
                    if (lastOl && lastOl.tagName === 'OL') {
                        lastOl.appendChild(el);
                        return null;
                    } else {
                        const ol = document.createElement('ol');
                        ol.appendChild(el);
                        return ol;
                    }
                case 'blockquote':
                    el = document.createElement('blockquote');
                    el.innerHTML = this.renderInline(block.content);
                    break;
                case 'hr':
                    el = document.createElement('hr');
                    break;
                default:
                    return null;
            }

            return el;
        }

        renderInline(text) {
            return text
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.+?)\*/g, '<em>$1</em>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
                .replace(/\[\^(\d+)\]/g, '<sup class="citation" data-citation="$1">[$1]</sup>');
        }

        highlightCode(element, language) {
            const keywords = KEYWORDS[language] || KEYWORDS.js;
            let html = element.textContent;

            // Escape HTML
            html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

            // Highlight strings
            html = html.replace(/(["'`])(?:(?!\1)[^\\]|\\.)*\1/g, '<span class="code-string">$&</span>');

            // Highlight comments
            html = html.replace(/(\/\/.*$|\/\*[\s\S]*?\*\/|#.*$)/gm, '<span class="code-comment">$1</span>');

            // Highlight keywords
            const keywordPattern = new RegExp(`\\b(${keywords.join('|')})\\b`, 'g');
            html = html.replace(keywordPattern, '<span class="code-keyword">$1</span>');

            // Highlight numbers
            html = html.replace(/\b(\d+)\b/g, '<span class="code-number">$1</span>');

            element.innerHTML = html;
        }

        finalize() {
            // Render any remaining buffer
            if (this.buffer) {
                this.renderMarkdown(this.buffer);
                this.buffer = '';
            }
            if (this.inCodeBlock && this.codeBuffer) {
                const codeEl = this.currentBlock.querySelector('code');
                codeEl.textContent = this.codeBuffer;
                this.highlightCode(codeEl, this.codeLanguage);
            }
        }
    }

    // Static render for non-streaming
    function render(markdown) {
        const container = document.createElement('div');
        const stream = new MarkdownStream(container);
        stream.append(markdown);
        stream.finalize();
        return container.innerHTML;
    }

    // Create streaming renderer
    function createStream(container) {
        return new MarkdownStream(container);
    }

    // Copy code to clipboard
    function copyCode(button) {
        const codeBlock = button.closest('.code-block');
        const code = codeBlock.querySelector('code').textContent;

        navigator.clipboard.writeText(code).then(() => {
            const originalText = button.textContent;
            button.textContent = 'Copied!';
            button.classList.add('copied');
            setTimeout(() => {
                button.textContent = originalText;
                button.classList.remove('copied');
            }, 2000);
        });
    }

    // Setup citation hover cards
    function setupCitations(container, citations) {
        const citationEls = container.querySelectorAll('.citation');
        citationEls.forEach(el => {
            const num = el.dataset.citation;
            const citation = citations[num - 1];
            if (citation) {
                el.title = `${citation.title}\n${citation.url}`;
                el.addEventListener('click', () => {
                    window.open(citation.url, '_blank');
                });
            }
        });
    }

    return {
        render,
        createStream,
        copyCode,
        setupCitations,
        MarkdownStream,
    };
})();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = DRXRenderer;
}
