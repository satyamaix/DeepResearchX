"""
Report Exporter Service for DRX Deep Research System.

Provides multi-format export of research reports including HTML and PDF.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from src.models.knowledge import KnowledgeGraph
    from src.orchestrator.state import AgentState

logger = logging.getLogger(__name__)

# Template directory
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


class ReportContext(TypedDict):
    """Context data for report template rendering."""

    title: str
    query: str
    synthesis: str
    findings: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    quality_metrics: dict[str, Any] | None
    knowledge_graph_svg: str | None
    generated_at: str
    language: str


class ReportExporter:
    """
    Export research reports to multiple formats.

    Supports:
    - HTML with embedded CSS
    - PDF via WeasyPrint
    - JSON structured data

    Example:
        ```python
        exporter = ReportExporter()

        html = exporter.to_html(state)
        pdf = exporter.to_pdf(state)
        ```
    """

    def __init__(
        self,
        template_dir: Path | None = None,
        default_template: str = "report_default.html.j2",
    ):
        """
        Initialize the report exporter.

        Args:
            template_dir: Directory containing Jinja2 templates
            default_template: Default template name
        """
        self._template_dir = template_dir or TEMPLATE_DIR
        self._default_template = default_template

        # Initialize Jinja2 environment
        self._env = Environment(
            loader=FileSystemLoader(str(self._template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def to_html(
        self,
        state: AgentState,
        template: str | None = None,
        include_graph: bool = True,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> str:
        """
        Export report as HTML.

        Args:
            state: Agent state with research results
            template: Template name (default: report_default.html.j2)
            include_graph: Whether to include knowledge graph visualization
            knowledge_graph: Optional knowledge graph for visualization

        Returns:
            HTML string
        """
        template_name = template or self._default_template

        # Build context
        context = self._build_context(state, knowledge_graph if include_graph else None)

        # Render template
        try:
            tmpl = self._env.get_template(template_name)
            return tmpl.render(**context)
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            # Fallback to simple HTML
            return self._fallback_html(context)

    def to_pdf(
        self,
        state: AgentState,
        template: str | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> bytes:
        """
        Export report as PDF.

        Args:
            state: Agent state with research results
            template: Template name
            knowledge_graph: Optional knowledge graph

        Returns:
            PDF bytes
        """
        try:
            from weasyprint import HTML
        except ImportError:
            logger.error("weasyprint not installed. Install with: pip install weasyprint")
            raise ImportError("weasyprint required for PDF export")

        # First generate HTML
        html_content = self.to_html(
            state, template, include_graph=True, knowledge_graph=knowledge_graph
        )

        # Convert to PDF
        html_doc = HTML(string=html_content)
        pdf_bytes = html_doc.write_pdf()

        return pdf_bytes

    def to_json(self, state: AgentState) -> dict[str, Any]:
        """
        Export report as structured JSON.

        Args:
            state: Agent state

        Returns:
            JSON-serializable dict
        """
        return {
            "title": "Research Report",
            "query": state.get("user_query", ""),
            "session_id": state.get("session_id", ""),
            "synthesis": state.get("synthesis", ""),
            "final_report": state.get("final_report"),
            "findings": state.get("findings", []),
            "citations": state.get("citations", []),
            "quality_metrics": state.get("quality_metrics"),
            "gaps": state.get("gaps", []),
            "tokens_used": state.get("tokens_used", 0),
            "iteration_count": state.get("iteration_count", 0),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    def render_knowledge_graph_svg(
        self,
        knowledge_graph: KnowledgeGraph,
        width: int = 800,
        height: int = 400,
    ) -> str:
        """
        Render knowledge graph as SVG.

        Uses a simple force-directed layout approximation.

        Args:
            knowledge_graph: Knowledge graph to render
            width: SVG width
            height: SVG height

        Returns:
            SVG string
        """
        cytoscape_data = knowledge_graph.export_cytoscape()
        nodes = cytoscape_data.get("nodes", [])
        edges = cytoscape_data.get("edges", [])

        if not nodes:
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">'
                '<text x="400" y="200" text-anchor="middle" fill="#666">No graph data</text>'
                "</svg>"
            )

        # Simple grid layout for nodes
        node_positions: dict[str, tuple[float, float]] = {}
        cols = max(3, int(len(nodes) ** 0.5) + 1)

        for i, node in enumerate(nodes):
            node_id = node["data"]["id"]
            col = i % cols
            row = i // cols
            x = 100 + col * ((width - 200) / max(cols - 1, 1))
            y = 80 + row * 100
            node_positions[node_id] = (x, min(y, height - 80))

        # Build SVG
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'style="background: #f9fafb; border-radius: 8px;">',
            "<defs>",
            '<marker id="arrowhead" markerWidth="10" markerHeight="7" '
            'refX="10" refY="3.5" orient="auto">',
            '<polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8"/>',
            "</marker>",
            "</defs>",
        ]

        # Draw edges
        for edge in edges:
            source_id = edge["data"]["source"]
            target_id = edge["data"]["target"]

            if source_id in node_positions and target_id in node_positions:
                x1, y1 = node_positions[source_id]
                x2, y2 = node_positions[target_id]

                # Shorten line to not overlap with nodes
                dx, dy = x2 - x1, y2 - y1
                length = (dx**2 + dy**2) ** 0.5
                if length > 0:
                    x2 = x1 + dx * (1 - 35 / length)
                    y2 = y1 + dy * (1 - 35 / length)

                svg_parts.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrowhead)"/>'
                )

        # Draw nodes
        type_colors = {
            "person": "#3b82f6",
            "organization": "#10b981",
            "concept": "#8b5cf6",
            "event": "#f59e0b",
            "location": "#ef4444",
            "document": "#6366f1",
            "claim": "#ec4899",
        }

        for node in nodes:
            node_id = node["data"]["id"]
            label = node["data"].get("label", node_id)[:20]
            node_type = node["data"].get("type", "concept")
            color = type_colors.get(node_type, "#6b7280")

            if node_id in node_positions:
                x, y = node_positions[node_id]

                svg_parts.append(
                    f'<circle cx="{x}" cy="{y}" r="30" fill="{color}" opacity="0.9"/>'
                )
                svg_parts.append(
                    f'<text x="{x}" y="{y + 4}" text-anchor="middle" fill="white" '
                    f'font-size="10" font-family="sans-serif">{label}</text>'
                )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _build_context(
        self,
        state: AgentState,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> ReportContext:
        """Build template context from agent state."""
        # Generate graph SVG if available
        graph_svg = None
        if knowledge_graph and knowledge_graph.entity_count > 0:
            graph_svg = self.render_knowledge_graph_svg(knowledge_graph)

        # Get synthesis or final report
        synthesis = state.get("final_report") or state.get("synthesis", "")

        # Convert markdown-ish to simple HTML
        synthesis_html = self._markdown_to_html(synthesis)

        return ReportContext(
            title="Research Report",
            query=state.get("user_query", ""),
            synthesis=synthesis_html,
            findings=state.get("findings", []),
            citations=state.get("citations", []),
            quality_metrics=state.get("quality_metrics"),
            knowledge_graph_svg=graph_svg,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            language=state.get("steerability", {}).get("language", "en"),
        )

    def _markdown_to_html(self, text: str) -> str:
        """Simple markdown to HTML conversion."""
        if not text:
            return ""

        # Headers
        text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
        text = re.sub(r"^## (.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
        text = re.sub(r"^# (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)

        # Bold and italic
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)

        # Lists
        text = re.sub(r"^- (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
        text = re.sub(r"(<li>.+</li>\n)+", r"<ul>\g<0></ul>", text)

        # Paragraphs
        paragraphs = text.split("\n\n")
        processed = []
        for p in paragraphs:
            p = p.strip()
            if p and not p.startswith("<"):
                p = f"<p>{p}</p>"
            processed.append(p)

        return "\n".join(processed)

    def _fallback_html(self, context: ReportContext) -> str:
        """Generate simple fallback HTML if template fails."""
        citations_list = "".join(
            f"<li>{c.get('title', c.get('url', 'Unknown'))}</li>"
            for c in context["citations"][:10]
        )
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{context['title']}</title>
    <style>body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; }}</style>
</head>
<body>
    <h1>{context['title']}</h1>
    <p><strong>Query:</strong> {context['query']}</p>
    <h2>Summary</h2>
    <div>{context['synthesis']}</div>
    <h2>Sources</h2>
    <ul>
    {citations_list}
    </ul>
    <footer>Generated: {context['generated_at']}</footer>
</body>
</html>"""


def create_report_exporter(**kwargs: Any) -> ReportExporter:
    """Factory function to create a ReportExporter."""
    return ReportExporter(**kwargs)


__all__ = [
    "ReportExporter",
    "ReportContext",
    "create_report_exporter",
]
