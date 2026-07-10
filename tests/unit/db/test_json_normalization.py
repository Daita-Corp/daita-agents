import pytest

from daita.db.json_normalization import strip_json_fence


@pytest.mark.parametrize(
    ("content", "expected"),
    (
        ('  {"value": 1}\n', '{"value": 1}'),
        ('```\n  {"value": 1}\n```', '{"value": 1}'),
        ('```json\n  {"value": 1}\n```', '{"value": 1}'),
        ("```json\n```", ""),
        ('```JSON\n{"value": 1}\n```', 'JSON\n{"value": 1}'),
        ('```python\n{"value": 1}\n```', 'python\n{"value": 1}'),
        ('```json\n{"value": 1}', '```json\n{"value": 1}'),
        (
            'prefix ```json\n{"value": 1}\n``` suffix',
            'prefix ```json\n{"value": 1}\n``` suffix',
        ),
        ("   ", ""),
    ),
)
def test_strip_json_fence_preserves_model_response_contract(content, expected):
    assert strip_json_fence(content) == expected
