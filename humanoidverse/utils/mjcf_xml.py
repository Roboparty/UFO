"""Tolerant MJCF loading for the Python XML parsers used alongside MuJoCo.

MuJoCo's own parser (tinyxml2) accepts XML comments that contain a double
hyphen (``<!-- ... -- ... -->``), but the XML specification forbids it and both
``xml.etree.ElementTree`` and ``lxml`` reject such files with
``not well-formed (invalid token)``.  Robot MJCFs exported by external tools
(for example the NEURA 4NE-1 files, whose header says ``-- do not edit by
hand``) hit exactly this.  MotionLib and the inference relabel utilities parse
the same MJCF with Python parsers, so they must be able to read anything
MuJoCo can load.

The helpers below strip every XML comment from the document text before
handing it to the parser.  Comments carry no model information, so the parsed
tree is identical to what a strictly well-formed file would produce.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def read_mjcf_xml_text(path: str | Path) -> str:
    """Return the MJCF text with all ``<!-- ... -->`` comments removed."""

    text = Path(path).expanduser().read_text(encoding="utf-8")
    return _COMMENT_RE.sub("", text)


def read_mjcf_xml_bytes(path: str | Path) -> bytes:
    """Bytes variant of :func:`read_mjcf_xml_text` for parsers that want a binary stream."""

    return read_mjcf_xml_text(path).encode("utf-8")


def parse_mjcf_etree(path: str | Path) -> ET.ElementTree:
    """Parse an MJCF with ``xml.etree`` after stripping comments; a drop-in for ``ET.parse(path)``."""

    return ET.ElementTree(ET.fromstring(read_mjcf_xml_text(path)))
