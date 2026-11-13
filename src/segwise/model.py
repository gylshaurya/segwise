"""Plain data structures shared by the analysis, the exporters and the UI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class Issue:
    """Something a person should know about the file.

    `detail` is the technical statement; `explain` says why it matters in plain
    words; `fix` says what segwise does (or what the user should do).
    """

    code: str
    severity: str  # "error" | "warning" | "info"
    title: str
    detail: str
    fix: str = ""
    explain: str = ""
    auto_fixed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    """One decision about the file (e.g. 'inline numbers are in bytes 9-12'), with its proof."""

    key: str
    label: str
    value: Any
    display: str
    confidence: str  # "high" | "medium" | "low" | "none"
    source: str  # "data" | "text header" | "binary header" | "user" | "default"
    evidence: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FieldRef:
    """A location of a value inside the 240-byte trace header."""

    byte: int  # 1-based
    type: str  # "int32" | "int16" | "float32"

    @property
    def size(self) -> int:
        return 2 if self.type == "int16" else 4

    def __str__(self) -> str:
        return f"bytes {self.byte}-{self.byte + self.size - 1} ({self.type})"

    def to_dict(self) -> dict[str, Any]:
        return {"byte": self.byte, "type": self.type}

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "FieldRef | None":
        if not d:
            return None
        return FieldRef(int(d["byte"]), str(d.get("type", "int32")))


def sort_issues(issues: list[Issue]) -> list[Issue]:
    return sorted(issues, key=lambda i: (SEVERITY_ORDER.get(i.severity, 9), i.auto_fixed, i.code))
