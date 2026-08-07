from app import build_context_text


def test_build_context_text_formats_documents():
    documents = [
        type(
            "Doc",
            (),
            {
                "metadata": {"title": "Paris"},
                "page_content": "Paris is the capital of France.",
            },
        )()
    ]

    result = build_context_text(documents)

    assert "[1] Source: Paris" in result
    assert "Paris is the capital of France." in result
