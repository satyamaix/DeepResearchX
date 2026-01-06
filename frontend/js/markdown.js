/**
 * DRX Markdown Renderer
 * Lightweight markdown parser and renderer with syntax highlighting
 */

const DRXMarkdown = (() => {
    /**
     * Escape HTML entities
     */
    function escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, char => map[char]);
    }

    /**
     * Parse inline elements (bold, italic, code, links, etc.)
     */
    function parseInline(text) {
        if (!text) return '';

        let result = escapeHtml(text);

        // Code (must be first to prevent other patterns matching inside)
        result = result.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold + Italic
        result = result.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
        result = result.replace(/___(.+?)___/g, '<strong><em>$1</em></strong>');

        // Bold
        result = result.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        result = result.replace(/__(.+?)__/g, '<strong>$1</strong>');

        // Italic
        result = result.replace(/\*(.+?)\*/g, '<em>$1</em>');
        result = result.replace(/_(.+?)_/g, '<em>$1</em>');

        // Strikethrough
        result = result.replace(/~~(.+?)~~/g, '<del>$1</del>');

        // Links
        result = result.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

        // Auto-link URLs
        result = result.replace(
            /(?<!href="|">)(https?:\/\/[^\s<]+)/g,
            '<a href="$1" target="_blank" rel="noopener">$1</a>'
        );

        // Superscript citations [^1]
        result = result.replace(/\[\^(\d+)\]/g, '<sup class="citation">[$1]</sup>');

        return result;
    }

    /**
     * Simple syntax highlighting for code blocks
     */
    function highlightCode(code, language) {
        if (!language) return escapeHtml(code);

        let highlighted = escapeHtml(code);

        // Common patterns for various languages
        const patterns = {
            // Keywords (language-agnostic common ones)
            keywords: /\b(function|const|let|var|if|else|for|while|return|class|import|export|from|async|await|try|catch|throw|new|this|super|extends|implements|interface|type|enum|public|private|protected|static|final|void|int|string|boolean|true|false|null|undefined|None|True|False|def|self|lambda|yield|with|as|raise|except|finally|pass|break|continue|in|is|not|and|or)\b/g,
            // Strings
            strings: /(["'`])(?:(?!\1)[^\\]|\\.)*\1/g,
            // Comments
            comments: /(\/\/.*$|\/\*[\s\S]*?\*\/|#.*$)/gm,
            // Numbers
            numbers: /\b(\d+\.?\d*)\b/g,
            // Functions
            functions: /\b([a-zA-Z_]\w*)\s*\(/g
        };

        // Apply highlighting in order (comments last to not break strings)
        highlighted = highlighted.replace(patterns.strings, '<span class="hljs-string">$&</span>');
        highlighted = highlighted.replace(patterns.keywords, '<span class="hljs-keyword">$1</span>');
        highlighted = highlighted.replace(patterns.numbers, '<span class="hljs-number">$1</span>');
        highlighted = highlighted.replace(patterns.functions, '<span class="hljs-function">$1</span>(');
        highlighted = highlighted.replace(patterns.comments, '<span class="hljs-comment">$1</span>');

        return highlighted;
    }

    /**
     * Parse markdown to HTML
     */
    function render(markdown, options = {}) {
        if (!markdown) return '';

        const lines = markdown.split('\n');
        const result = [];
        let inCodeBlock = false;
        let codeBlockLang = '';
        let codeBlockContent = [];
        let inList = false;
        let listType = null;
        let listItems = [];
        let inBlockquote = false;
        let blockquoteContent = [];
        let inTable = false;
        let tableRows = [];

        function flushList() {
            if (listItems.length > 0) {
                const tag = listType === 'ol' ? 'ol' : 'ul';
                result.push(`<${tag}>`);
                listItems.forEach(item => {
                    result.push(`<li>${parseInline(item)}</li>`);
                });
                result.push(`</${tag}>`);
                listItems = [];
                inList = false;
                listType = null;
            }
        }

        function flushBlockquote() {
            if (blockquoteContent.length > 0) {
                result.push('<blockquote>');
                result.push(render(blockquoteContent.join('\n')));
                result.push('</blockquote>');
                blockquoteContent = [];
                inBlockquote = false;
            }
        }

        function flushTable() {
            if (tableRows.length > 0) {
                result.push('<table>');
                tableRows.forEach((row, i) => {
                    const cells = row.split('|').filter(c => c.trim());
                    const tag = i === 0 ? 'th' : 'td';
                    const rowTag = i === 0 ? 'thead' : (i === 1 ? 'tbody' : '');

                    if (rowTag === 'thead') result.push('<thead>');
                    if (rowTag === 'tbody') result.push('<tbody>');

                    // Skip separator row
                    if (!/^[\s\-:|]+$/.test(row)) {
                        result.push('<tr>');
                        cells.forEach(cell => {
                            result.push(`<${tag}>${parseInline(cell.trim())}</${tag}>`);
                        });
                        result.push('</tr>');
                    }

                    if (i === 0) result.push('</thead>');
                });
                result.push('</tbody></table>');
                tableRows = [];
                inTable = false;
            }
        }

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            // Code blocks
            if (line.startsWith('```')) {
                if (inCodeBlock) {
                    const highlighted = highlightCode(codeBlockContent.join('\n'), codeBlockLang);
                    result.push(`<pre><code class="language-${codeBlockLang || 'text'}">${highlighted}</code></pre>`);
                    codeBlockContent = [];
                    codeBlockLang = '';
                    inCodeBlock = false;
                } else {
                    flushList();
                    flushBlockquote();
                    flushTable();
                    codeBlockLang = line.slice(3).trim();
                    inCodeBlock = true;
                }
                continue;
            }

            if (inCodeBlock) {
                codeBlockContent.push(line);
                continue;
            }

            // Blockquotes
            if (line.startsWith('>')) {
                flushList();
                flushTable();
                inBlockquote = true;
                blockquoteContent.push(line.slice(1).trim());
                continue;
            } else if (inBlockquote && line.trim() === '') {
                flushBlockquote();
                continue;
            } else if (inBlockquote) {
                blockquoteContent.push(line);
                continue;
            }

            // Tables
            if (line.includes('|') && line.trim().startsWith('|')) {
                flushList();
                flushBlockquote();
                inTable = true;
                tableRows.push(line);
                continue;
            } else if (inTable) {
                flushTable();
            }

            // Headings
            const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
            if (headingMatch) {
                flushList();
                flushBlockquote();
                const level = headingMatch[1].length;
                const text = headingMatch[2];
                const id = text.toLowerCase().replace(/[^\w]+/g, '-');
                result.push(`<h${level} id="${id}">${parseInline(text)}</h${level}>`);
                continue;
            }

            // Horizontal rule
            if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
                flushList();
                flushBlockquote();
                result.push('<hr>');
                continue;
            }

            // Unordered list
            const ulMatch = line.match(/^(\s*)[-*+]\s+(.+)$/);
            if (ulMatch) {
                flushBlockquote();
                flushTable();
                if (!inList || listType !== 'ul') {
                    flushList();
                    inList = true;
                    listType = 'ul';
                }
                listItems.push(ulMatch[2]);
                continue;
            }

            // Ordered list
            const olMatch = line.match(/^(\s*)\d+\.\s+(.+)$/);
            if (olMatch) {
                flushBlockquote();
                flushTable();
                if (!inList || listType !== 'ol') {
                    flushList();
                    inList = true;
                    listType = 'ol';
                }
                listItems.push(olMatch[2]);
                continue;
            }

            // Empty line
            if (line.trim() === '') {
                flushList();
                flushBlockquote();
                flushTable();
                continue;
            }

            // Paragraph
            flushList();
            flushBlockquote();
            flushTable();
            result.push(`<p>${parseInline(line)}</p>`);
        }

        // Flush remaining content
        flushList();
        flushBlockquote();
        flushTable();

        if (inCodeBlock) {
            const highlighted = highlightCode(codeBlockContent.join('\n'), codeBlockLang);
            result.push(`<pre><code class="language-${codeBlockLang || 'text'}">${highlighted}</code></pre>`);
        }

        return result.join('\n');
    }

    /**
     * Render markdown with streaming support (for partial content)
     */
    function renderStreaming(markdown, container) {
        const html = render(markdown);
        container.innerHTML = html;

        // Scroll to bottom if near bottom
        const parent = container.parentElement;
        if (parent) {
            const isNearBottom = parent.scrollHeight - parent.scrollTop - parent.clientHeight < 100;
            if (isNearBottom) {
                parent.scrollTop = parent.scrollHeight;
            }
        }
    }

    /**
     * Extract table of contents from markdown
     */
    function extractToc(markdown) {
        const toc = [];
        const lines = markdown.split('\n');

        for (const line of lines) {
            const match = line.match(/^(#{1,6})\s+(.+)$/);
            if (match) {
                const level = match[1].length;
                const text = match[2];
                const id = text.toLowerCase().replace(/[^\w]+/g, '-');
                toc.push({ level, text, id });
            }
        }

        return toc;
    }

    /**
     * Render table of contents
     */
    function renderToc(markdown) {
        const toc = extractToc(markdown);
        if (toc.length === 0) return '';

        const result = ['<nav class="toc"><h4>Contents</h4><ul>'];
        let prevLevel = 1;

        for (const item of toc) {
            if (item.level > prevLevel) {
                result.push('<ul>'.repeat(item.level - prevLevel));
            } else if (item.level < prevLevel) {
                result.push('</ul>'.repeat(prevLevel - item.level));
            }
            result.push(`<li><a href="#${item.id}">${escapeHtml(item.text)}</a></li>`);
            prevLevel = item.level;
        }

        result.push('</ul>'.repeat(prevLevel));
        result.push('</nav>');

        return result.join('');
    }

    /**
     * Strip markdown formatting
     */
    function stripMarkdown(markdown) {
        if (!markdown) return '';

        return markdown
            // Remove code blocks
            .replace(/```[\s\S]*?```/g, '')
            // Remove inline code
            .replace(/`[^`]+`/g, '')
            // Remove headings
            .replace(/^#{1,6}\s+/gm, '')
            // Remove bold/italic
            .replace(/\*\*\*(.+?)\*\*\*/g, '$1')
            .replace(/\*\*(.+?)\*\*/g, '$1')
            .replace(/\*(.+?)\*/g, '$1')
            .replace(/___(.+?)___/g, '$1')
            .replace(/__(.+?)__/g, '$1')
            .replace(/_(.+?)_/g, '$1')
            // Remove links
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
            // Remove images
            .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
            // Remove blockquotes
            .replace(/^>\s+/gm, '')
            // Remove horizontal rules
            .replace(/^(-{3,}|\*{3,}|_{3,})$/gm, '')
            // Clean up whitespace
            .replace(/\n{3,}/g, '\n\n')
            .trim();
    }

    return {
        render,
        renderStreaming,
        renderToc,
        extractToc,
        stripMarkdown,
        escapeHtml,
        parseInline
    };
})();

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DRXMarkdown;
}
