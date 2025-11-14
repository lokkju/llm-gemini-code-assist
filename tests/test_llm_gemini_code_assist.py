from click.testing import CliRunner
from datetime import datetime
import llm
from llm.cli import cli
import nest_asyncio
import json
import os
import pytest
import pydantic
from llm_gemini_code_assist import cleanup_schema, get_oauth_token, get_oauth_credentials, OAuthError, _SharedGemini, OAUTH_CREDENTIALS_FILE, _save_json_to_plugin_cache, _clean_plugin_cache, _load_json_from_plugin_cache, authenticate
from unittest.mock import patch
from pathlib import Path
import textwrap as tw
from tests.asserts import assert_dict_contains, assert_structure_matches, assert_gemini_2_5_flash_lite_response

nest_asyncio.apply()

@pytest.mark.vcr
@pytest.mark.dependency()
@pytest.mark.usefixtures("mock_llm_user_path")
def test_authenticate(mock_llm_user_path):
    credentials = authenticate(reauthenticate=False, use_gemini_cli_creds=False, use_oauth=False)
    assert credentials.valid

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
def test_prompt_with_pydantic_schema():
    class Dog(pydantic.BaseModel):
        name: str
        age: int
        bio: str

    model = llm.get_model("gemini-2.5-flash-ca")
    response = model.prompt(
        "Invent a cool dog", key=None, schema=Dog, stream=False
    )
    assert_structure_matches(json.loads(str(response)), {
        "age": int,
        "bio": str,
        "name": str,
    })
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

    # Mock home directory to use tmpdir
    monkeypatch.setattr(Path, "home", lambda: tmpdir)
    _clean_plugin_cache()

    # Test 1: No file exists
    assert get_oauth_token() is None

    # Test 2: File exists with access_token
    _save_json_to_plugin_cache(OAUTH_CREDENTIALS_FILE,{
        "access_token": "test_token_123",
        "expires_at": datetime.now().timestamp() + 3600
    })
    assert get_oauth_token() == "test_token_123"

    # Test 3: Invalid JSON
    _save_json_to_plugin_cache(OAUTH_CREDENTIALS_FILE,"not valid json")
    with pytest.raises(OAuthError):
        assert get_oauth_token()


def test_oauth_token_refresh_success(tmpdir, monkeypatch):
    """Test successful OAuth token refresh"""
    monkeypatch.setattr(Path, "home", lambda: tmpdir)
    _clean_plugin_cache()

    # Create expired token
    #
    expired_time = int(datetime.utcnow().timestamp() - 100) * 1000
    _save_json_to_plugin_cache(OAUTH_CREDENTIALS_FILE,{
        "access_token": "expired_token",
        "expires_at": expired_time,
        "refresh_token": "CENSORED-REFRESH-TOKEN",
        "client_id": "client_id_123",
        "client_secret": "client_secret_123"
    })

    # Mock the Credentials.refresh method
    from google.oauth2.credentials import Credentials

    def mock_refresh(self, request):
        # Update the token as if refresh succeeded
        self.token = "new_token_789"
        self.expiry = datetime.now() + __import__('datetime').timedelta(seconds=3600)

    with patch.object(Credentials, 'refresh', mock_refresh):
        token = get_oauth_token()

        # Verify new token is returned
        assert token == "new_token_789"

        # Verify file was updated
        updated_creds = _load_json_from_plugin_cache(OAUTH_CREDENTIALS_FILE)
        assert updated_creds["access_token"] == "new_token_789"
        assert "expiry_date" in updated_creds


def test_oauth_token_refresh_missing_refresh_token(tmpdir, monkeypatch):
    """Test OAuth refresh fails when refresh_token is missing"""
    monkeypatch.setattr(Path, "home", lambda: tmpdir)
    _clean_plugin_cache()

    # Create expired token without refresh_token
    expired_time = datetime.utcnow().timestamp() - 100
    _save_json_to_plugin_cache(OAUTH_CREDENTIALS_FILE,{
        "access_token": "expired_token",
        "expires_at": expired_time
    })

    with pytest.raises(OAuthError, match="no refresh_token is available"):
        get_oauth_token()


@pytest.mark.vcr
def test_oauth_token_refresh_failed_request(tmpdir, monkeypatch):
    """Test OAuth refresh fails when HTTP request fails"""
    monkeypatch.setattr(Path, "home", lambda: tmpdir)
    _clean_plugin_cache()

    # Create expired token
    expired_time = int(datetime.utcnow().timestamp() - 100) * 1000
    _save_json_to_plugin_cache(OAUTH_CREDENTIALS_FILE,{
        "access_token": "expired_token",
        "expires_at": expired_time,
        "refresh_token": "CENSORED-REFRESH-TOKEN",
        "client_id": "client_id_123",
        "client_secret": "client_secret_123"
    })

    # Google's library will make real request and fail
    with pytest.raises(OAuthError, match="Failed to refresh OAuth token"):
        print(get_oauth_token())


def test_oauth_header_generation(tmpdir, monkeypatch):
    """Test that OAuth tokens generate Bearer auth headers"""
    monkeypatch.setattr(Path, "home", lambda: tmpdir)
    _clean_plugin_cache()

    # Create valid token
    _save_json_to_plugin_cache(OAUTH_CREDENTIALS_FILE,{
        "access_token": "test_oauth_token",
        "expires_at": int(datetime.utcnow().timestamp() + 3600) * 1000
    })

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
