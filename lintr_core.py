from __future__ import annotations

from dataclasses import dataclass, field
import json
import random
import re
from typing import Any


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class Intervention:
    kind: str
    reason: str
    raw: str = ""
    repaired: str = ""
    sampler_patch: dict[str, Any] | None = None


@dataclass
class SessionState:
    is_thinking: bool = False
    think_tail: str = ""
    visible_window: str = ""
    recent_sentences: list[str] = field(default_factory=list)
    next_sampler_patch: dict[str, Any] | None = None
    detections_in_response: int = 0
    interventions: list[Intervention] = field(default_factory=list)

    def reset_response(self) -> None:
        self.is_thinking = False
        self.think_tail = ""
        self.visible_window = ""
        self.recent_sentences.clear()
        self.detections_in_response = 0

    def record(self, intervention: Intervention) -> None:
        self.interventions.append(intervention)
        if len(self.interventions) > 50:
            del self.interventions[:-50]


class ThinkSplitter:
    """Splits a chunk into visible and think spans while preserving raw output."""

    def __init__(self, state: SessionState) -> None:
        self.state = state

    def visible_parts(self, chunk: str) -> list[str]:
        text = self.state.think_tail + chunk
        self.state.think_tail = ""
        visible: list[str] = []
        i = 0
        while i < len(text):
            if self.state.is_thinking:
                close_i = text.find(THINK_CLOSE, i)
                if close_i < 0:
                    self.state.think_tail = partial_tag_tail(text[i:], THINK_CLOSE)
                    return visible
                i = close_i + len(THINK_CLOSE)
                self.state.is_thinking = False
                continue

            open_i = text.find(THINK_OPEN, i)
            if open_i < 0:
                tail = partial_tag_tail(text[i:], THINK_OPEN)
                safe_end = len(text) - len(tail)
                if safe_end > i:
                    visible.append(text[i:safe_end])
                self.state.think_tail = tail
                return visible
            if open_i > i:
                visible.append(text[i:open_i])
            i = open_i + len(THINK_OPEN)
            self.state.is_thinking = True
        return visible


def partial_tag_tail(text: str, tag: str) -> str:
    max_len = min(len(text), len(tag) - 1)
    for size in range(max_len, 0, -1):
        suffix = text[-size:]
        if tag.startswith(suffix):
            return suffix
    return ""


class DegradationDetector:
    sentence_re = re.compile(r"[^.!?\n]+[.!?\n]")
    word_re = re.compile(r"[A-Za-z][A-Za-z'_-]*")
    url_re = re.compile(r"https?://\S+")
    code_fence_re = re.compile(r"```.*?```", re.DOTALL)

    def inspect(self, state: SessionState, visible_text: str) -> list[str]:
        if not visible_text:
            return []
        state.visible_window = (state.visible_window + visible_text)[-4000:]
        clean = self._strip_ignored(state.visible_window)
        reasons: list[str] = []

        if re.search(r"\S{97,}", clean):
            reasons.append("long_no_space_run")

        words = [w.lower().strip("'_-") for w in self.word_re.findall(clean)][-80:]
        counts: dict[str, int] = {}
        for word in words:
            if not word:
                continue
            counts[word] = counts.get(word, 0) + 1
        for word, count in counts.items():
            if len(word) >= 5 and count >= 5:
                reasons.append(f"word_repeat:{word}")
                break
            if len(word) < 5 and count >= 18:
                reasons.append(f"small_word_repeat:{word}")
                break

        for match in self.sentence_re.finditer(visible_text):
            normalized = self._normalize_sentence(match.group(0))
            if len(normalized) < 24:
                continue
            if normalized in state.recent_sentences:
                reasons.append("sentence_repeat")
                break
            state.recent_sentences.append(normalized)
            if len(state.recent_sentences) > 32:
                del state.recent_sentences[:-32]

        if self._unicode_burst(clean):
            reasons.append("unicode_burst")

        return reasons

    def _strip_ignored(self, text: str) -> str:
        text = self.code_fence_re.sub(" ", text)
        text = self.url_re.sub(" ", text)
        return text

    def _normalize_sentence(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _unicode_burst(self, text: str) -> bool:
        unexpected = 0
        latinish = 0
        for ch in text[-600:]:
            code = ord(ch)
            if ch.isascii() or ch.isspace():
                if ch.isalpha():
                    latinish += 1
                continue
            if 0x1F300 <= code <= 0x1FAFF:
                continue
            if 0x00A0 <= code <= 0x024F:
                continue
            if 0x2000 <= code <= 0x206F:
                continue
            unexpected += 1
        return latinish >= 24 and unexpected >= 4


class SamplerScrambler:
    ranges = {
        "off": None,
        "mild": {"temperature": (0.65, 1.05), "top_k": (35, 80)},
        "medium": {"temperature": (0.75, 1.25), "top_k": (30, 110)},
        "high": {"temperature": (0.90, 1.45), "top_k": (25, 150)},
    }

    def __init__(self, intensity: str = "mild") -> None:
        self.intensity = intensity if intensity in self.ranges else "mild"

    def patch_for_detection_count(self, count: int) -> dict[str, Any] | None:
        if self.intensity == "off":
            return None
        tier = self.intensity
        if count >= 2 and tier == "mild":
            tier = "medium"
        spec = self.ranges[tier]
        if spec is None:
            return None
        temp_min, temp_max = spec["temperature"]
        k_min, k_max = spec["top_k"]
        return {
            "temperature": round(random.uniform(temp_min, temp_max), 2),
            "top_k": random.randint(k_min, k_max),
        }


class ToolJsonLinter:
    tool_markers = (
        '"tool_calls"',
        '"function"',
        '"arguments"',
        '"method"',
        '"params"',
    )

    def lint_text(self, text: str) -> tuple[str, Intervention | None]:
        if not self._looks_like_tool_json(text):
            return text, None
        candidate = self._extract_candidate(text)
        if not candidate:
            return text, None
        repaired = self._repair(candidate)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError as exc:
            return text, Intervention(
                kind="tool_lint_error",
                reason=f"unrepairable_json:{exc.msg}",
                raw=candidate,
            )
        normalized = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return normalized, Intervention(
            kind="tool_lint_repair",
            reason="json_repaired",
            raw=candidate,
            repaired=normalized,
        )

    def _looks_like_tool_json(self, text: str) -> bool:
        if not any(marker in text for marker in self.tool_markers):
            return False
        return "{" in text or "[" in text

    def _extract_candidate(self, text: str) -> str | None:
        stripped = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
        if fence:
            stripped = fence.group(1).strip()
        starts = [i for i in (stripped.find("{"), stripped.find("[")) if i >= 0]
        if not starts:
            return None
        start = min(starts)
        return self._balanced_prefix(stripped[start:])

    def _balanced_prefix(self, text: str) -> str:
        stack: list[str] = []
        in_string = False
        escaped = False
        end = len(text)
        pairs = {"{": "}", "[": "]"}
        for i, ch in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in pairs:
                stack.append(pairs[ch])
            elif ch in "]}":
                if stack and ch == stack[-1]:
                    stack.pop()
                    if not stack:
                        end = i + 1
                        break
        return text[:end].strip()

    def _repair(self, text: str) -> str:
        repaired = text.strip()
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        repaired = self._close_string(repaired)
        repaired = self._close_brackets(repaired)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        return repaired

    def _close_string(self, text: str) -> str:
        escaped = False
        in_string = False
        for ch in text:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
        return text + ('"' if in_string else "")

    def _close_brackets(self, text: str) -> str:
        stack: list[str] = []
        out: list[str] = []
        in_string = False
        escaped = False
        pairs = {"{": "}", "[": "]"}
        for ch in text:
            out.append(ch)
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in pairs:
                stack.append(pairs[ch])
            elif ch in "]}":
                while stack and ch != stack[-1]:
                    out.insert(len(out) - 1, stack.pop())
                if stack and ch == stack[-1]:
                    stack.pop()
        return "".join(out) + "".join(reversed(stack))


class LintrEngine:
    def __init__(self, intensity: str = "mild") -> None:
        self.sessions: dict[str, SessionState] = {}
        self.detector = DegradationDetector()
        self.scrambler = SamplerScrambler(intensity)
        self.tool_linter = ToolJsonLinter()
        self.tool_repair_enabled: bool = True

    def state_for(self, session_id: str) -> SessionState:
        return self.sessions.setdefault(session_id, SessionState())

    def begin_request(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.state_for(session_id)
        patched = dict(payload)
        if state.next_sampler_patch:
            patched.update(state.next_sampler_patch)
            state.record(Intervention(
                kind="sampler_patch_applied",
                reason="previous_degradation",
                sampler_patch=state.next_sampler_patch,
            ))
            state.next_sampler_patch = None
        state.reset_response()
        return patched

    def observe_delta(self, session_id: str, text: str) -> list[Intervention]:
        state = self.state_for(session_id)
        splitter = ThinkSplitter(state)
        interventions: list[Intervention] = []
        for visible in splitter.visible_parts(text):
            reasons = self.detector.inspect(state, visible)
            for reason in reasons:
                state.detections_in_response += 1
                patch = self.scrambler.patch_for_detection_count(state.detections_in_response)
                if patch:
                    state.next_sampler_patch = patch
                intervention = Intervention(
                    kind="degradation_detected",
                    reason=reason,
                    raw=visible[-240:],
                    sampler_patch=patch,
                )
                state.record(intervention)
                interventions.append(intervention)
        return interventions

    def lint_final_text(self, session_id: str, text: str) -> str:
        self.observe_delta(session_id, text)
        if not self.tool_repair_enabled:
            return text
        repaired, intervention = self.tool_linter.lint_text(text)
        if intervention:
            self.state_for(session_id).record(intervention)
        return repaired

    def debug_state(self, session_id: str | None = None) -> dict[str, Any]:
        ids = [session_id] if session_id else sorted(self.sessions)
        out: dict[str, Any] = {}
        for sid in ids:
            state = self.sessions.get(sid)
            if not state:
                continue
            out[sid] = {
                "is_thinking": state.is_thinking,
                "next_sampler_patch": state.next_sampler_patch,
                "detections_in_response": state.detections_in_response,
                "interventions": [i.__dict__ for i in state.interventions[-20:]],
            }
        return out


def conversation_id(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("conversation_id", "conversationId", "session_id", "sessionId"):
            value = metadata.get(key)
            if value:
                return str(value)
    for key in ("conversation_id", "conversationId", "session_id", "sessionId", "user"):
        value = payload.get(key)
        if value:
            return str(value)
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict):
            content = str(first.get("content", ""))
            if content:
                return "hash:" + str(abs(hash(content)) % 1_000_000_000)
    return "default"
