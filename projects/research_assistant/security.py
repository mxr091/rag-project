"""Small, explicit security checks for untrusted RAG evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Match imperative attempts, not ordinary educational text such as "模型调用工具".
_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"忽略(以上|之前|前面).{0,12}(指令|规则|要求)"),
    re.compile(r"(显示|输出|泄露|告诉我).{0,10}(system\s*prompt|系统提示词)", re.IGNORECASE),
    re.compile(r"(请|必须|立即|现在).{0,8}(调用|执行).{0,8}(工具|命令|代码)"),
)


@dataclass(frozen=True)
class SecurityFinding:
    rule: str
    matched_text: str


def find_prompt_injection(text: str) -> tuple[SecurityFinding, ...]:
    findings: list[SecurityFinding] = []
    for index, pattern in enumerate(_INJECTION_PATTERNS, start=1):
        match = pattern.search(text)
        if match:
            findings.append(SecurityFinding(f"prompt_injection_{index}", match.group(0)))
    return tuple(findings)


def is_safe_evidence(text: str) -> bool:
    return not find_prompt_injection(text)
