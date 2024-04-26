from robust_llm.attacks.trl.utils import prepare_prompts
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)


def test_prepare_prompts():
    text_chunked = [["A", "B", "C"], ["D", "E", "F"]]

    # Test replacement with a single response
    response_text = "X"
    prompts = prepare_prompts(
        text_chunked=text_chunked,
        modifiable_chunk_spec=ModifiableChunkSpec(
            ChunkType.IMMUTABLE, ChunkType.OVERWRITABLE, ChunkType.IMMUTABLE
        ),
        response_text=response_text,
    )
    assert prompts == ["AXC", "DXF"]

    # Test replacement with a sequence of responses
    response_text_list = ["X", "Y"]
    prompts = prepare_prompts(
        text_chunked=text_chunked,
        modifiable_chunk_spec=ModifiableChunkSpec(
            ChunkType.IMMUTABLE, ChunkType.IMMUTABLE, ChunkType.OVERWRITABLE
        ),
        response_text=response_text_list,
    )
    assert prompts == ["ABX", "DEY"]
