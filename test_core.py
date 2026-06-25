import json

from lintr_core import LintrEngine, ToolJsonLinter


def test_think_bypass_and_degradation():
    engine = LintrEngine(intensity="mild")
    engine.begin_request("s", {"messages": []})
    engine.observe_delta("s", "<think>repeat repeat repeat repeat repeat</think>")
    assert engine.debug_state("s")["s"]["next_sampler_patch"] is None
    engine.observe_delta("s", " alpha alpha alpha alpha alpha.")
    assert engine.debug_state("s")["s"]["next_sampler_patch"] is not None


def test_tool_json_repair():
    linter = ToolJsonLinter()
    text = '```json\n{"tool_calls":[{"function":{"name":"web_search","arguments":{"query":"hi",},},]}\n```'
    repaired, intervention = linter.lint_text(text)
    assert intervention is not None
    assert '"web_search"' in repaired
    assert ",}" not in repaired
    assert ",]" not in repaired


def test_tool_json_repair_unclosed_brackets():
    # Model stops generating mid-object — missing closing }]}
    linter = ToolJsonLinter()
    text = '{"tool_calls":[{"function":{"name":"foo","arguments":{"q":"hi"}'
    repaired, intervention = linter.lint_text(text)
    assert intervention is not None
    parsed = json.loads(repaired)  # must be valid JSON after repair
    assert parsed["tool_calls"][0]["function"]["name"] == "foo"


def test_tool_json_repair_mismatched_brackets():
    # Closing brace where a bracket was expected — {[}
    linter = ToolJsonLinter()
    text = '{"arguments":[}'
    repaired, intervention = linter.lint_text(text)
    assert intervention is not None
    json.loads(repaired)  # must produce valid JSON


def test_tool_json_repair_no_false_positive():
    # Normal prose with no tool markers should pass through unchanged
    linter = ToolJsonLinter()
    text = "Here is the answer to your question."
    repaired, intervention = linter.lint_text(text)
    assert intervention is None
    assert repaired == text


def test_sampler_escalation():
    # count >= 4 with mild base should reach high tier
    from lintr_core import SamplerScrambler
    s = SamplerScrambler("mild")
    p1 = s.patch_for_detection_count(1)
    p4 = s.patch_for_detection_count(4)
    assert p1 is not None
    assert p4 is not None
    # high tier has wider top_k range (25-150) vs mild (35-80)
    # run enough samples to be confident we saw a high-tier value
    hits_above_mild_max = sum(
        1 for _ in range(50) if s.patch_for_detection_count(4)["top_k"] > 80
    )
    assert hits_above_mild_max > 0, "count=4 should sometimes produce top_k > mild ceiling"


def test_conversation_id_stable():
    # conversation_id fallback must be deterministic across calls (no random hash)
    from lintr_core import conversation_id
    payload = {"messages": [{"role": "system", "content": "you are helpful"}]}
    assert conversation_id(payload) == conversation_id(payload)


if __name__ == "__main__":
    test_think_bypass_and_degradation()
    test_tool_json_repair()
    test_tool_json_repair_unclosed_brackets()
    test_tool_json_repair_mismatched_brackets()
    test_tool_json_repair_no_false_positive()
    test_sampler_escalation()
    test_conversation_id_stable()
    print("ok")
