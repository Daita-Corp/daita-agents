"""
Tests for memory content preprocessor.

Verifies that preprocess_content() correctly strips code blocks, markdown
formatting, and template noise while preserving the factual signal.
"""

from daita.plugins.memory.preprocessor import preprocess_content


class TestFencedCodeStripping:
    def test_strips_sql_code_block(self):
        content = (
            "Fix the revenue table:\n"
            "```sql\n"
            "UPDATE orders SET total = total_cents;\n"
            "DELETE FROM revenue_daily;\n"
            "```\n"
            "This restores original precision."
        )
        _, index = preprocess_content(content)
        assert "UPDATE orders" not in index
        assert "DELETE FROM" not in index
        assert "restores original precision" in index

    def test_strips_yaml_code_block(self):
        content = (
            "Pipeline config:\n"
            "```yaml\n"
            "schema_check: {source_columns: {total: DECIMAL}}\n"
            "```\n"
            "Add this to prevent recurrence."
        )
        _, index = preprocess_content(content)
        assert "schema_check" not in index
        assert "prevent recurrence" in index

    def test_strips_multiple_code_blocks(self):
        content = (
            "Step 1:\n```sql\nSELECT 1;\n```\n"
            "Step 2:\n```json\n{\"key\": \"val\"}\n```\n"
            "Done."
        )
        _, index = preprocess_content(content)
        assert "SELECT" not in index
        assert '{"key"' not in index
        assert "Done" in index

    def test_purely_code_content_falls_back(self):
        content = "```python\nprint('hello')\n```"
        _, index = preprocess_content(content)
        # Should not be empty — falls back to original
        assert len(index) > 0


class TestInlineCodeStripping:
    def test_preserves_identifier_text(self):
        content = "The `orders.total` column changed from `DECIMAL` to `INTEGER`."
        _, index = preprocess_content(content)
        assert "orders.total" in index
        assert "DECIMAL" in index
        assert "`" not in index

    def test_strips_backticks_only(self):
        content = "Check `customer_segments` table."
        _, index = preprocess_content(content)
        assert "customer_segments" in index
        assert "`" not in index


class TestMarkdownStripping:
    def test_strips_bold(self):
        content = "**Remediation Plan** for the incident."
        _, index = preprocess_content(content)
        assert "Remediation Plan" in index
        assert "**" not in index

    def test_strips_headers(self):
        content = "## SQL Fix Steps\nRestore values."
        _, index = preprocess_content(content)
        assert "SQL Fix Steps" in index
        assert "##" not in index

    def test_strips_bullet_prefixes(self):
        content = "- Step 1: restore\n- Step 2: validate\n* Step 3: deploy"
        _, index = preprocess_content(content)
        assert "Step 1: restore" in index
        # Bullet chars should be stripped
        lines = [l.strip() for l in index.strip().split("\n") if l.strip()]
        for line in lines:
            assert not line.startswith("- ")
            assert not line.startswith("* ")

    def test_strips_numbered_list_prefixes(self):
        content = "1. First\n2. Second\n3. Third"
        _, index = preprocess_content(content)
        assert "First" in index
        assert "Second" in index


class TestWhitespaceCollapse:
    def test_collapses_multiple_blanks(self):
        content = "Line one.\n\n\n\n\nLine two."
        _, index = preprocess_content(content)
        assert "\n\n\n" not in index
        assert "Line one" in index
        assert "Line two" in index


class TestStoragePreserved:
    def test_storage_is_original(self):
        content = "**Bold** with ```sql\nSELECT 1;\n```"
        storage, index = preprocess_content(content)
        assert storage == content  # Original preserved
        assert "**Bold**" in storage
        assert "SELECT" in storage
        # Index is cleaned
        assert "**" not in index
        assert "SELECT" not in index

    def test_empty_content(self):
        storage, index = preprocess_content("")
        assert storage == ""
        assert index == ""

    def test_none_content(self):
        storage, index = preprocess_content(None)
        assert storage is None
        assert index is None

    def test_plain_text_unchanged(self):
        content = "Pipeline etl_orders_agg reads from orders and writes to revenue_daily."
        storage, index = preprocess_content(content)
        assert storage == content
        assert index == content


class TestRealisticContent:
    def test_remediation_plan(self):
        """The Remediator's markdown+SQL output should produce clean index text."""
        content = (
            "**Remediation Plan for Data Quality Incident**\n\n"
            "**SQL Fix Steps:**\n"
            "1. Restore original values:\n"
            "```sql\n"
            "UPDATE orders SET total = total_cents;\n"
            "```\n"
            "2. Verify restoration:\n"
            "```sql\n"
            "SELECT total, total_cents FROM orders;\n"
            "```\n\n"
            "**Prevention Measures:**\n"
            "- Add schema drift detection to CI/CD\n"
            "- Require approval for column type changes\n"
        )
        _, index = preprocess_content(content)
        # SQL should be gone
        assert "UPDATE orders" not in index
        assert "SELECT total" not in index
        # Prose should remain
        assert "Remediation Plan" in index
        assert "schema drift detection" in index
        assert "column type changes" in index
        # Markdown formatting should be gone
        assert "**" not in index
        assert "```" not in index

    def test_schema_memory(self):
        """Schema memories are plain text — should pass through mostly unchanged."""
        content = (
            "Table: orders\n"
            "Columns:\n"
            "- id: INTEGER\n"
            "- customer_id: INTEGER NOT NULL\n"
            "- total: DECIMAL(10,2) NOT NULL\n"
        )
        _, index = preprocess_content(content)
        # Content preserved (no code blocks or markdown to strip)
        assert "orders" in index
        assert "DECIMAL(10,2)" in index
        # Bullet prefix stripped
        assert "id: INTEGER" in index
