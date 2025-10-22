from click.testing import CliRunner
from datetime import datetime
import llm
from llm.cli import cli
import nest_asyncio
import json
import os
import pytest
import pydantic
from llm_gemini_code_assist import cleanup_schema, get_oauth_token, OAuthError, _SharedGemini
from unittest.mock import patch, MagicMock
from pathlib import Path
import textwrap as tw

def _assert_recursive_contains(actual, expected, path=""):
    """
    Internal helper to recursively assert 'actual' contains 'expected'.
    Ignores extra keys in 'actual' dicts.
    Requires exact matches for lists and simple values.
    """

    # Case 1: Expected is a dictionary
    if isinstance(expected, dict):
        # Check that 'actual' is also a dict
        if not isinstance(actual, dict):
            pytest.fail(f"Type mismatch at path '{path}': Expected dict, got {type(actual).__name__}")

        # Check all keys from 'expected' exist in 'actual'
        for key, expected_value in expected.items():
            current_path = f"{path}.{key}" if path else key
            if key not in actual:
                pytest.fail(f"Missing key '{key}' in actual response at path: {path}")

            # Recurse to check the value
            _assert_recursive_contains(actual[key], expected_value, path=current_path)

    # Case 2: Expected is a list
    elif isinstance(expected, list):
        # Check that 'actual' is also a list
        if not isinstance(actual, list):
            pytest.fail(f"Type mismatch at path '{path}': Expected list, got {type(actual).__name__}")

        # Check for exact length match (adjust if partial lists are OK)
        if len(actual) != len(expected):
            pytest.fail(f"List length mismatch for key: '{path}'. Expected {len(expected)}, got {len(actual)}")

        # Recurse for each item in the list
        for i, expected_item in enumerate(expected):
            item_path = f"{path}[{i}]"
            _assert_recursive_contains(actual[i], expected_item, path=item_path)

    # Case 3: Simple value (str, int, bool, etc.)
    else:
        # Simple value comparison
        if actual != expected:
            pytest.fail(f"Value mismatch for key: '{path}'. Expected '{expected}', got '{actual}'")

def assert_dict_contains(actual, expected, path=""):
    """
    Recursively asserts that the 'actual' dict contains all keys and values
    from the 'expected' dict, ignoring extra keys in 'actual'.

    This is a wrapper for the fully recursive helper.
    """
    # This entry-point function expects the top level to be a dict.
    if not isinstance(actual, dict):
        pytest.fail(f"Expected 'actual' to be a dict at path: {path or 'root'}")
    if not isinstance(expected, dict):
        pytest.fail("Expected 'expected' to be a dict.")

    # Call the helper that can handle any nested type
    _assert_recursive_contains(actual, expected, path)

def assert_structure_matches(actual, schema, path=""):
    """
    Recursively asserts that the 'actual' data matches the 'schema' structure.

    - 'actual' is the JSON data you received.
    - 'schema' is a dict/list/type defining the expected structure.

    - Ignores extra keys in 'actual' dictionaries.
    - Validates types, not specific values.
    """

    # Case 1: Schema is a dictionary
    if isinstance(schema, dict):
        if not isinstance(actual, dict):
            pytest.fail(f"Type mismatch at path '{path}': Expected dict, got {type(actual).__name__}")

        # Iterate through the keys defined in the SCHEMA
        for key, sub_schema in schema.items():
            current_path = f"{path}.{key}" if path else key

            # Check that the key exists in the actual response
            if key not in actual:
                pytest.fail(f"Missing key at path '{path}': Expected key '{key}'")

            # Recurse to check the structure of the sub-element
            assert_structure_matches(actual[key], sub_schema, path=current_path)

    # Case 2: Schema is a list (defines the structure for all items)
    elif isinstance(schema, list):
        if not isinstance(actual, list):
            pytest.fail(f"Type mismatch at path '{path}': Expected list, got {type(actual).__name__}")

        # If the schema list is empty (schema=[]), it matches any list (including empty).
        if not schema:
            return

        # Use the first element of the schema as the template for ALL items
        item_schema = schema[0]
        for index, actual_item in enumerate(actual):
            current_path = f"{path}[{index}]"
            assert_structure_matches(actual_item, item_schema, path=current_path)

    # Case 3: Schema is a type (str, int, bool, float, etc.)
    elif isinstance(schema, type):
        if not isinstance(actual, schema):
            pytest.fail(f"Type mismatch at path '{path}': Expected type {schema.__name__}, got {type(actual).__name__}")

    # Case 4: Invalid schema definition
    else:
        raise TypeError(f"Invalid schema at path '{path}'. Schema must be a dict, list, or type (like str). Got {type(schema)}")

def assert_gemini_2_5_flash_lite_response(response):
    expected_response = {
        "candidates": [
            {
                "finishReason": "STOP",
            }
        ],
        "modelVersion": "gemini-2.5-flash-lite",
    }
    assert_dict_contains(response.response_json, expected_response)

def assert_gemini_2_5_flash_response(response):
    expected_response = {
        "candidates": [
            {
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                    },
                ],
            }
        ],
        "modelVersion": "gemini-2.5-flash",
    }
    assert_dict_contains(response.response_json, expected_response)

nest_asyncio.apply()

@pytest.mark.vcr
def test_prompt_sync():
    model = llm.get_model("gemini-ca/gemini-2.5-flash-lite")
    response = model.prompt("Most popular search engine, just the name", key=None)
    assert "Google" in str(response)
    assert_gemini_2_5_flash_lite_response(response)
    assert_dict_contains(response.token_details, {
        "candidatesTokenCount": 1,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 8}],
        "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 1}],
    })
    assert response.input_tokens == 8
    assert response.output_tokens == 1


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt_async():
    # And try it async too
    async_model = llm.get_async_model("gemini-ca/gemini-2.5-flash-lite")
    response = await async_model.prompt(
        "Most popular search engine, just the name", key=None
    )
    text = await response.text()
    assert "Google" in str(text)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt_with_pydantic_schema():
    class Dog(pydantic.BaseModel):
        name: str
        age: int
        bio: str

    model = llm.get_model("gemini-2.5-flash-ca")
    response = model.prompt(
        "Invent a cool dog", key=None, schema=Dog, stream=False
    )
    print(response)
    # assert_structure_matches(json.loads(response.text()), {
    #     "age": int,
    #     "bio": str,
    #     "name": str,
    # })
    # assert_gemini_2_5_flash_response(response)
    assert response.input_tokens == 17


@pytest.mark.parametrize(
    "schema,expected",
    [
        # Test 1: Top-level keys removal
        (
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "Example Schema",
                "additionalProperties": False,
                "type": "object",
            },
            {"type": "object"},
        ),
        # Test 2: Preserve keys within a "properties" block
        (
            {
                "type": "object",
                "properties": {
                    "authors": {"type": "string"},
                    "title": {"type": "string"},
                    "reference": {"type": "string"},
                    "year": {"type": "string"},
                },
                "title": "This should be removed from the top-level",
            },
            {
                "type": "object",
                "properties": {
                    "authors": {"type": "string"},
                    "title": {"type": "string"},
                    "reference": {"type": "string"},
                    "year": {"type": "string"},
                },
            },
        ),
        # Test 3: Nested keys outside and inside properties block
        (
            {
                "definitions": {
                    "info": {
                        "title": "Info title",  # should be removed because it's not inside a "properties" block
                        "description": "A description",
                        "properties": {
                            "name": {
                                "title": "Name Title",
                                "type": "string",
                            },  # title here should be preserved
                            "$schema": {
                                "type": "string"
                            },  # should be preserved as it's within properties
                        },
                    }
                },
                "$schema": "http://example.com/schema",
            },
            {
                "definitions": {
                    "info": {
                        "description": "A description",
                        "properties": {
                            "name": {"title": "Name Title", "type": "string"},
                            "$schema": {"type": "string"},
                        },
                    }
                }
            },
        ),
        # Test 4: List of schemas
        (
            [
                {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                },
                {"title": "Should be removed", "type": "array"},
            ],
            [{"type": "object"}, {"type": "array"}],
        ),
    ],
)
def test_cleanup_schema(schema, expected):
    # Use a deep copy so the original test data remains unchanged.
    result = cleanup_schema(schema)
    assert result == expected


@pytest.mark.vcr
def test_cli_gemini_models(tmpdir, monkeypatch):
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))
    # Mock home directory to prevent finding real OAuth creds
    monkeypatch.setattr(Path, "home", lambda: tmpdir)
    # With no OAuth creds should error nicely
    runner = CliRunner()
    result = runner.invoke(cli, ["gemini-ca", "models"])
    assert result.output == tw.dedent("""
    [
      "gemini-2.5-flash",
      "gemini-2.5-flash-lite",
      "gemini-2.5-pro"
    ]
    """).lstrip()


@pytest.mark.vcr
def test_tools():
    model = llm.get_model("gemini-2.5-flash-ca")
    names = ["Charles", "Sammy"]
    chain_response = model.chain(
        "Two names for a pet pelican",
        tools=[
            llm.Tool.function(lambda: names.pop(0), name="pelican_name_generator"),
        ],
        key=None,
    )
    text = chain_response.text()
    assert "Charles and Sammy" in text
    # This one did three
    response_count = len(chain_response._responses)
    if response_count == 3:
        first, second, third = chain_response._responses
        assert len(first.tool_calls()) == 1
        assert first.tool_calls()[0].name == "pelican_name_generator"
        assert len(second.tool_calls()) == 1
        assert second.tool_calls()[0].name == "pelican_name_generator"
        assert second.prompt.tool_results[0].output == "Charles"
        assert third.prompt.tool_results[0].output == "Sammy"
        assert len(third.tool_calls()) == 0
    elif response_count == 2:
        first, second = chain_response._responses
        assert len(first.tool_calls()) == 2
        assert first.tool_calls()[0].name == "pelican_name_generator"
        assert first.tool_calls()[1].name == "pelican_name_generator"
        assert second.prompt.tool_results[0].output == "Charles"
        # The last response is combined
        assert "Sammy" in str(second)
    else:
        assert False, f"Expected three responses in the chain, got {response_count}"

def test_oauth_token_reading(tmpdir, monkeypatch):
    """Test reading OAuth token from file"""
    oauth_dir = tmpdir / ".gemini"
    oauth_dir.mkdir()
    oauth_file = oauth_dir / "oauth_creds.json"

    # Mock home directory to use tmpdir
    monkeypatch.setattr(Path, "home", lambda: tmpdir)

    # Test 1: No file exists
    assert get_oauth_token() is None

    # Test 2: File exists with access_token
    oauth_file.write_text(json.dumps({
        "access_token": "test_token_123",
        "expires_at": datetime.now().timestamp() + 3600
    }), encoding="utf-8")
    assert get_oauth_token() == "test_token_123"

    # Test 3: File exists with token field (alternative name)
    oauth_file.write_text(json.dumps({
        "token": "test_token_456",
        "expires_at": datetime.now().timestamp() + 3600
    }), encoding="utf-8")
    assert get_oauth_token() == "test_token_456"

    # Test 4: Invalid JSON
    oauth_file.write_text("not valid json", encoding="utf-8")
    assert get_oauth_token() is None


def test_oauth_token_refresh_success(tmpdir, monkeypatch):
    """Test successful OAuth token refresh"""
    oauth_dir = tmpdir / ".gemini"
    oauth_dir.mkdir()
    oauth_file = oauth_dir / "oauth_creds.json"

    monkeypatch.setattr(Path, "home", lambda: tmpdir)

    # Create expired token
    expired_time = datetime.now().timestamp() - 100
    oauth_file.write_text(json.dumps({
        "access_token": "expired_token",
        "expires_at": expired_time,
        "refresh_token": "CENSORED-REFRESH-TOKEN",
        "client_id": "client_id_123",
        "client_secret": "client_secret_123"
    }), encoding="utf-8")

    # Mock the Credentials.refresh method
    from google.oauth2.credentials import Credentials
    original_refresh = Credentials.refresh

    def mock_refresh(self, request):
        # Update the token as if refresh succeeded
        self.token = "new_token_789"
        self.expiry = datetime.now() + __import__('datetime').timedelta(seconds=3600)

    with patch.object(Credentials, 'refresh', mock_refresh):
        token = get_oauth_token()

        # Verify new token is returned
        assert token == "new_token_789"

        # Verify file was updated
        updated_creds = json.loads(oauth_file.read_text(encoding="utf-8"))
        assert updated_creds["access_token"] == "new_token_789"
        assert "expiry_date" in updated_creds


def test_oauth_token_refresh_missing_refresh_token(tmpdir, monkeypatch):
    """Test OAuth refresh fails when refresh_token is missing"""
    oauth_dir = tmpdir / ".gemini"
    oauth_dir.mkdir()
    oauth_file = oauth_dir / "oauth_creds.json"

    monkeypatch.setattr(Path, "home", lambda: tmpdir)

    # Create expired token without refresh_token
    expired_time = datetime.now().timestamp() - 100
    oauth_file.write_text(json.dumps({
        "access_token": "expired_token",
        "expires_at": expired_time
    }), encoding="utf-8")

    with pytest.raises(OAuthError, match="no refresh_token is available"):
        get_oauth_token()


def test_oauth_token_refresh_missing_credentials(tmpdir, monkeypatch):
    """Test OAuth refresh fails when client credentials are missing"""
    oauth_dir = tmpdir / ".gemini"
    oauth_dir.mkdir()
    oauth_file = oauth_dir / "oauth_creds.json"

    monkeypatch.setattr(Path, "home", lambda: tmpdir)

    # Create expired token with refresh_token but missing client credentials
    expired_time = datetime.now().timestamp() - 100
    oauth_file.write_text(json.dumps({
        "access_token": "expired_token",
        "expires_at": expired_time,
        "refresh_token": "CENSORED-REFRESH-TOKEN"
    }), encoding="utf-8")

    # Google's library will fail to refresh with actual error message
    with pytest.raises(OAuthError, match="Failed to refresh OAuth token"):
        get_oauth_token()


def test_oauth_token_refresh_failed_request(tmpdir, monkeypatch):
    """Test OAuth refresh fails when HTTP request fails"""
    oauth_dir = tmpdir / ".gemini"
    oauth_dir.mkdir()
    oauth_file = oauth_dir / "oauth_creds.json"

    monkeypatch.setattr(Path, "home", lambda: tmpdir)

    # Create expired token
    expired_time = datetime.now().timestamp() - 100
    oauth_file.write_text(json.dumps({
        "access_token": "expired_token",
        "expires_at": expired_time,
        "refresh_token": "CENSORED-REFRESH-TOKEN",
        "client_id": "client_id_123",
        "client_secret": "client_secret_123"
    }), encoding="utf-8")

    # Google's library will make real request and fail
    with pytest.raises(OAuthError, match="Failed to refresh OAuth token"):
        get_oauth_token()


def test_oauth_header_generation(tmpdir, monkeypatch):
    """Test that OAuth tokens generate Bearer auth headers"""
    oauth_dir = tmpdir / ".gemini"
    oauth_dir.mkdir()
    oauth_file = oauth_dir / "oauth_creds.json"

    monkeypatch.setattr(Path, "home", lambda: tmpdir)

    # Create valid token
    oauth_file.write_text(json.dumps({
        "access_token": "test_oauth_token",
        "expires_at": datetime.now().timestamp() + 3600
    }), encoding="utf-8")

    # Test with model
    model: _SharedGemini = llm.get_model("gemini-2.5-flash-lite-ca")

    # The get_auth_headers method should return Bearer token
    headers = model.get_auth_headers()
    assert headers == {"Authorization": "Bearer test_oauth_token"}


@pytest.mark.skipif(
    os.environ.get("PYTEST_GEMINI_OAUTH_TOKEN") is None,
    reason="This is a live integration test and requires PYTEST_GEMINI_OAUTH_TOKEN"
)
@pytest.mark.asyncio
async def test_oauth_integration(tmpdir, monkeypatch):
    """Integration test: Make actual API call using OAuth token from cache"""
    oauth_dir = tmpdir / ".gemini"
    oauth_dir.mkdir()
    oauth_file = oauth_dir / "oauth_creds.json"

    # Mock home directory to use tmpdir
    monkeypatch.setattr(Path, "home", lambda: tmpdir)

    # Get OAuth token from environment for real testing
    # This allows developers to run: PYTEST_GEMINI_OAUTH_TOKEN=<token> pytest
    oauth_token = os.environ.get("PYTEST_GEMINI_OAUTH_TOKEN", "oauth-test-token")

    # Create OAuth cache file with token
    oauth_file.write_text(json.dumps({
        "access_token": oauth_token,
        "expires_at": datetime.now().timestamp() + 3600
    }), encoding="utf-8")

    # Ensure no API key is set so it falls back to OAuth
    monkeypatch.delenv("LLM_GEMINI_KEY", raising=False)

    # Make API call using OAuth
    model = llm.get_model("gemini-2.5-flash-lite-ca")
    response = model.prompt("Name for a pet pelican, just the name")
    text = str(response)

    # Verify we got a response
    assert len(text) > 0
    assert response.response_json is not None

    # Verify the response contains expected fields
    assert "candidates" in response.response_json
    assert "modelVersion" in response.response_json

    # Also test async
    async_model = llm.get_async_model("gemini-2.5-flash-lite-ca")
    async_response = await async_model.prompt("Name for a pet pelican, just the name")
    async_text = await async_response.text()
    assert len(async_text) > 0
