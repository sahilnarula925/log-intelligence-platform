"""Source-aware template extraction for deduplication."""

from __future__ import annotations

import re

WIKIMEDIA_PATTERN = re.compile(r"^(.+?) edited (.+?): \+(\d+) bytes$")
MASTODON_PATTERN = re.compile(r"^@(.+?) posted on Mastodon:")
WEBHOOK_PATTERN = re.compile(r"^\[(INFO|WARN|ERROR|DEBUG)\] (.+?): (.+)$")


def extract_template(message: str, source: str = "unknown") -> str:
    if source == "wikimedia":
        if WIKIMEDIA_PATTERN.match(message):
            return "<user> edited <page>: <delta> bytes"
    elif source == "mastodon":
        if MASTODON_PATTERN.match(message):
            return "@<user> posted on Mastodon: <content>"
    elif source == "webhook":
        match = WEBHOOK_PATTERN.match(message)
        if match:
            return f"[{match.group(1)}] <service>: <message>"

    # Fallback: collapse long numeric tokens to reduce embedding churn.
    collapsed = re.sub(r"\b\d+\b", "<num>", message)
    if len(collapsed) > 120:
        collapsed = collapsed[:120] + "…"
    return collapsed
