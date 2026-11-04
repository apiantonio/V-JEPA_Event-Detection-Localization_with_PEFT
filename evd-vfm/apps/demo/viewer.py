"""``index.html``: the viewer, third artifact of the demonstrator.

A single file without dependencies, server or build: the payload JSON travels inside
the document, the video sits next to it, and it opens with a double click even offline.
Compared with ``annotated.mp4`` it can be paused, rewound and queried, and it shows the
causal windows as discrete overlapping cells (8 s with a 4 s hop).

The template is ``assets/viewer.html``, a real HTML file rather than a string inside a
Python module.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from apps.demo.payload import DemoPayload

#: The template, next to this module.
TEMPLATE = Path(__file__).resolve().parent / "assets" / "viewer.html"

#: The two placeholders declared by the template, searched exactly: if the template
#: renames one, writing fails instead of producing a page without data.
_PAYLOAD_PLACEHOLDER = "__PAYLOAD__"
_TITLE_PLACEHOLDER = "__TITLE__"


def scrivi(payload: DemoPayload, out_path: str | Path) -> Path:
    """Write the page with the payload embedded.

    The JSON goes inside a ``<script type="application/json">`` (not a JavaScript variable),
    so it is not interpreted as code; ``</`` is split because it is the only sequence that
    would close the script early.

    :param payload: the demonstrator data.
    :param out_path: where to write ``index.html``; parents are created.
    :return: the written path.
    :raises FileNotFoundError: if the template is not next to the module.
    :raises ValueError: if the template lacks the expected placeholders.
    """
    if not TEMPLATE.exists():
        raise FileNotFoundError(
            f"{TEMPLATE} does not exist: the page template ships with the module "
            f"and is not regenerated"
        )
    html = TEMPLATE.read_text(encoding="utf-8")
    for placeholder in (_PAYLOAD_PLACEHOLDER, _TITLE_PLACEHOLDER):
        if placeholder not in html:
            raise ValueError(f"{TEMPLATE} does not contain the placeholder {placeholder}")
    html = html.replace(
        "__METRICS_JS__", (TEMPLATE.parent / "metrics.js").read_text(encoding="utf-8")
    )
    data = json.dumps(payload.as_dict(), separators=(",", ":")).replace("</", "<\\/")
    title = escape(f"{payload.video_id} — EVD Studio")
    html = html.replace(_TITLE_PLACEHOLDER, title).replace(_PAYLOAD_PLACEHOLDER, data)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p
