# DRX Frontend Usage Guide

## Quick Start

```bash
# Start the backend services
cd deployment && docker compose up -d

# Access the UI
open http://localhost:8000
```

The frontend is served directly by the FastAPI backend on port 8000.

---

## Interface Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  LEFT SIDEBAR          MAIN CONTENT            RIGHT SIDEBAR        │
│  ┌─────────────┐      ┌─────────────────┐     ┌─────────────────┐  │
│  │ Model       │      │                 │     │ Output Style    │  │
│  │ Selection   │      │  Chat Messages  │     │ - Tone          │  │
│  │             │      │                 │     │ - Format        │  │
│  │ Thinking    │      │  + DAG Viz      │     │ - Language      │  │
│  │ Mode        │      │                 │     │                 │  │
│  │             │      │                 │     │ Research Depth  │  │
│  │ Active      │      │                 │     │ - Iterations    │  │
│  │ Agents      │      │                 │     │ - Sources       │  │
│  │             │      │                 │     │ - Token Budget  │  │
│  │ Appearance  │      ├─────────────────┤     │                 │  │
│  └─────────────┘      │  Query Input    │     │ Focus Areas     │  │
│                       └─────────────────┘     │ Exclude Topics  │  │
│                                               │ Preferred Doms  │  │
│                                               └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Left Sidebar: Configuration

### Model Selection

| Setting | Options | Default |
|---------|---------|---------|
| **Primary Model** | Gemini 3 Flash, Gemini 3 Pro, Claude Sonnet 4, GPT-4o, DeepSeek Chat | Gemini 3 Flash |
| **Reasoning Model** | DeepSeek R1, Claude Sonnet 4, OpenAI o1 | DeepSeek R1 |

### Thinking Mode

- **Show Thinking Steps**: Display agent reasoning in real-time
- **Show Agent Transitions**: Show when control passes between agents

### Active Agents

Toggle individual agents on/off (Planner and Reporter are always required):

| Agent | Role | Toggleable |
|-------|------|------------|
| Planner | Creates research plan | No |
| Searcher | Finds sources | Yes |
| Reader | Extracts content | Yes |
| Synthesizer | Merges findings | Yes |
| Critic | Quality assessment | Yes |
| Reporter | Generates report | No |

### Appearance

- **Dark Mode**: Toggle between dark/light themes

---

## Right Sidebar: Research Settings

### Output Style

| Setting | Options |
|---------|---------|
| Tone | Technical, Executive Summary, Casual |
| Format | Markdown, Markdown with Tables, JSON |
| Language | English, Spanish, French, German, Chinese, Japanese |

### Research Depth

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Max Iterations | 1-10 | 5 | Maximum critique/refinement cycles |
| Max Sources | 5-50 | 20 | Maximum sources to analyze |
| Token Budget | 50K-1M | 500K | Maximum tokens per session |
| Timeout | 1-30 min | 10 min | Maximum research duration |

### Quality Settings

- **Enable Citations**: Include source citations in report
- **Quality Checks**: Run critic agent for quality assessment

### Focus Areas

Add keywords to focus research on specific topics. Press Enter to add.

### Exclude Topics

Add keywords to exclude certain topics from research.

### Preferred Domains

Add domain patterns to prioritize sources:
- `arxiv.org` - Specific domain
- `*.edu` - All educational domains
- `*.gov` - All government domains

### Configuration Management

- **Quick Presets**: Pre-configured settings
  - Quick: 2 iterations, 5 sources
  - Deep: 5 iterations, 20 sources
  - Comprehensive: 8 iterations, 50 sources
  - Cost-Efficient: Free models only
- **Export Config**: Save settings to JSON file
- **Import Config**: Load settings from JSON file

---

## Main Content Area

### Chat Interface

1. **Type your research question** in the input area (minimum 10 characters)
2. **Press Enter** or click Send button to start research
3. **View progress** in the progress indicator:
   - Current phase (Planning, Searching, Reading, etc.)
   - Token usage
   - Elapsed time
   - Active agent status

### DAG Visualization

During research, a directed acyclic graph shows the research workflow:

| Node Color | Status |
|------------|--------|
| Gray | Pending |
| Blue (pulsing) | Running |
| Green | Completed |
| Red | Failed |

Controls:
- **Fit**: Fit graph to container
- **Reset**: Reset zoom level
- **Export**: Download as PNG

### Research Report

When complete, the report modal opens with:

- **Table of Contents**: Navigate sections (left sidebar)
- **Reading Progress**: Track scroll position
- **Collapsible Sections**: Expand/collapse content
- **Knowledge Graph**: Visualize entities and relationships
- **Export Options**: Download as Markdown, HTML, or PDF

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` / `Cmd+Enter` | Send message |
| `Escape` | Close modal / Cancel |
| `Ctrl+,` / `Cmd+,` | Toggle settings sidebar |
| `Ctrl+/` / `Cmd+/` | Focus search/input |
| `Ctrl+Shift+C` / `Cmd+Shift+C` | Copy report |
| `?` | Show keyboard shortcuts |
| `Alt+Up/Down` | Navigate report sections |

---

## Mobile Experience

### Touch Gestures

| Gesture | Action |
|---------|--------|
| Swipe from left edge | Open left sidebar |
| Swipe from right edge | Open right sidebar |
| Swipe sidebar away | Close sidebar |
| Long-press on message | Open context menu |

### Responsive Layout

- Sidebars collapse to hamburger menus
- Bottom sheet for settings on mobile
- Touch-friendly controls

---

## Connection Status

The connection indicator shows real-time status:

| Status | Indicator |
|--------|-----------|
| Connected | Green dot |
| Connecting | Yellow dot (pulsing) |
| Disconnected | Red dot |
| Reconnecting | Blue dot with spinner |

### Automatic Reconnection

If connection is lost:
1. Banner appears with countdown timer
2. Automatic retry with exponential backoff
3. Manual retry button available
4. Session recovery on reconnect

---

## Data Persistence

The frontend automatically saves:

- **Chat History**: Last 50 messages (localStorage)
- **Configuration**: All settings
- **Session State**: Active research sessions
- **Pending Interactions**: For recovery after disconnect

Clear data: Open browser DevTools → Application → Local Storage → Clear

---

## Troubleshooting

### API Connection Failed

```bash
# Check if backend is running
curl http://localhost:8000/api/v1/health

# Check Docker services
cd deployment && docker compose ps
```

### Research Not Starting

1. Verify query is at least 10 characters
2. Check connection status indicator
3. Open browser DevTools console for errors

### Report Not Loading

1. Check for JavaScript errors in console
2. Verify SSE connection is active
3. Try refreshing the page

### Performance Issues

1. Clear chat history (localStorage)
2. Reduce max sources setting
3. Use Quick preset for faster results

---

## Browser Support

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 90+ | Full support |
| Firefox | 88+ | Full support |
| Safari | 14+ | Full support |
| Edge | 90+ | Full support |

Required features: ES2020, CSS Grid, CSS Custom Properties, EventSource (SSE)
