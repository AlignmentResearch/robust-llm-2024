from robust_llm.attacks.trl.utils import prepare_prompts


def test_prepare_prompts():
    text_chunked = [["A", "B", "C"], ["D", "E", "F"]]

    # Test replacement with a single response
    response_text = "X"
    prompts = prepare_prompts(
        text_chunked=text_chunked,
        modifiable_chunks_spec=(False, True, False),
        response_text=response_text,
    )
    assert prompts == ["AXC", "DXF"]

    # Test replacement with a sequence of responses
    response_text_list = ["X", "Y"]
    prompts = prepare_prompts(
        text_chunked=text_chunked,
        modifiable_chunks_spec=(False, False, True),
        response_text=response_text_list,
    )
    assert prompts == ["ABX", "DEY"]
