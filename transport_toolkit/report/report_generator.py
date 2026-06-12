"""Simple report generation."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from transport_toolkit.utils.results import result_to_dict


def _plain_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    try:
        return result_to_dict(value)
    except TypeError:
        return value


def _render_mapping(mapping: dict[str, Any]) -> str:
    rows = []
    for key, value in mapping.items():
        value = _plain_value(value)
        if isinstance(value, dict):
            rendered = _render_mapping(value)
        else:
            rendered = f"<pre>{html.escape(str(value))}</pre>"
        rows.append(f"<section><h2>{html.escape(str(key))}</h2>{rendered}</section>")
    return "\n".join(rows)


def _minimal_pdf(path: Path, lines: list[str]) -> None:
    text = "\\n".join(lines).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 11 Tf 72 740 Td 14 TL ({text}) Tj ET".encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode("ascii"))
        content.extend(obj)
        content.extend(b"\nendobj\n")
    xref = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(content)


def generate_report(
    results: dict[str, Any],
    *,
    output: str | Path = "transport_report.html",
    sample_info: dict[str, Any] | None = None,
    figures: list[str | Path] | None = None,
) -> Path:
    """Generate an HTML report, or a compact PDF text summary."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sample_info": sample_info or {}, "results": results}
    if figures:
        payload["figures"] = [str(figure) for figure in figures]

    if output_path.suffix.lower() == ".pdf":
        lines = ["Transport Analysis Report", ""]
        for key, value in payload.items():
            lines.append(str(key))
            lines.append(str(_plain_value(value)))
            lines.append("")
        _minimal_pdf(output_path, lines)
        return output_path

    body = _render_mapping(payload)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Transport Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; line-height: 1.45; color: #1f2933; }}
    h1 {{ margin-bottom: 0.25rem; }}
    h2 {{ margin-top: 1.5rem; color: #12355b; }}
    section {{ border-top: 1px solid #d8dee6; padding-top: 0.75rem; }}
    pre {{ background: #f6f8fa; padding: 0.75rem; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Transport Analysis Report</h1>
  {body}
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    return output_path
