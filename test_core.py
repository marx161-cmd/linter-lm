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


if __name__ == "__main__":
    test_think_bypass_and_degradation()
    test_tool_json_repair()
    print("ok")
