import json
import os
import textwrap as tw
from datetime import datetime
from unittest.mock import patch

import llm
import nest_asyncio
import pydantic
import pytest
from click.testing import CliRunner
from llm.cli import cli

from llm_gemini_code_assist import (
    OAUTH_CREDENTIALS_FILE,
    SCOPES,
    AuthenticationError,
    _clean_plugin_cache,
    _load_json_from_plugin_cache,
    _save_json_to_plugin_cache,
    _SharedGemini,
    authenticate,
    cleanup_schema,
    get_oauth_token,
)
from tests.asserts import (
    assert_dict_contains,
    assert_gemini_2_5_flash_lite_response,
    assert_structure_matches,
)


nest_asyncio.apply()


@pytest.mark.usefixtures("shared_mock_llm_user_path", "mock_oauth_credentials")
def test_authenticate(shared_mock_llm_user_path):
    """Test authentication - credentials are provided by mock_oauth_credentials fixture"""
    # The mock_oauth_credentials fixture handles both recording and playback
    # During recording: it copies real credentials
    # During playback: it mocks get_oauth_credentials
    # This test doesn't make HTTP requests, so it doesn't need VCR
    credentials = authenticate()
    assert credentials is not None
    assert credentials.valid


@pytest.mark.vcr
@pytest.mark.usefixtures("shared_mock_llm_user_path", "mock_oauth_credentials")
def test_prompt_sync():
    model = llm.get_model("gemini-ca/gemini-2.5-flash-lite")
    response = model.prompt("Most popular search engine, just the name", key=None)
    assert "Google" in str(response)
    assert_gemini_2_5_flash_lite_response(response)
    assert_dict_contains(
        response.token_details,
        {
            "candidatesTokenCount": 1,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 8}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 1}],
        },
    )
    assert response.input_tokens == 8
    assert response.output_tokens == 1


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.usefixtures("shared_mock_llm_user_path", "mock_oauth_credentials")
async def test_prompt_async():
    # And try it async too
    async_model = llm.get_async_model("gemini-ca/gemini-2.5-flash-lite")
    response = await async_model.prompt("Most popular search engine, just the name", key=None)
    text = await response.text()
    assert "Google" in str(text)


@pytest.mark.vcr
@pytest.mark.usefixtures("shared_mock_llm_user_path", "mock_oauth_credentials")
def test_prompt_with_pydantic_schema():
    class Dog(pydantic.BaseModel):
        name: str
        age: int
        bio: str

    model = llm.get_model("gemini-2.5-flash-ca")
    response = model.prompt("Invent a cool dog", key=None, schema=Dog, stream=False)
    assert_structure_matches(
        json.loads(str(response)),
        {
            "age": int,
            "bio": str,
            "name": str,
        },
    )
    # assert_gemini_2_5_flash_response(response)
    assert response.input_tokens == 17


@pytest.mark.parametrize(
    ("schema", "expected"),
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
                        # title should be removed because it's not inside a "properties" block
                        "title": "Info title",
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


@pytest.mark.usefixtures("shared_mock_llm_user_path")
def test_cli_gemini_ca_models():
    runner = CliRunner()
    result = runner.invoke(cli, ["gemini-ca", "models"])
    assert (
        result.output
        == tw.dedent("""
    [
      "gemini-2.5-flash",
      "gemini-2.5-flash-lite",
      "gemini-2.5-pro"
    ]
    """).lstrip()
    )


@pytest.mark.vcr
@pytest.mark.usefixtures("shared_mock_llm_user_path", "mock_oauth_credentials")
def test_prompt_tools():
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
        raise AssertionError(f"Expected three responses in the chain, got {response_count}")


@pytest.mark.usefixtures("mock_llm_user_path")
def test_oauth_token_reading():
    """Test reading OAuth token from file"""
    _clean_plugin_cache()

    # Test 1: No file exists
    with pytest.raises(AuthenticationError):
        assert get_oauth_token()

    # Test 2: File exists with access_token
    _save_json_to_plugin_cache(
        OAUTH_CREDENTIALS_FILE,
        {
            "access_token": "test_token_123",
            "expiry_date": int((datetime.utcnow().timestamp() + 3600) * 1000),
        },
    )
    assert get_oauth_token() == "test_token_123"

    # Test 3: Invalid JSON
    _save_json_to_plugin_cache(OAUTH_CREDENTIALS_FILE, "not valid json")
    with pytest.raises(AuthenticationError):
        assert get_oauth_token()


@pytest.mark.usefixtures("mock_llm_user_path")
def test_oauth_token_refresh_success():
    """Test successful OAuth token refresh"""
    # Create expired token
    #
    expiry_date = int(datetime.utcnow().timestamp() - 100) * 1000
    _save_json_to_plugin_cache(
        OAUTH_CREDENTIALS_FILE,
        {
            "access_token": "expired_token",
            "expiry_date": expiry_date,
            "refresh_token": "CENSORED-REFRESH-TOKEN",
            "client_id": "client_id_123",
            "client_secret": "client_secret_123",
        },
    )

    # Mock the Credentials.refresh method
    from google.oauth2.credentials import Credentials

    def mock_refresh(self, request):
        # Update the token as if refresh succeeded
        self.token = "new_token_789"
        self.expiry = datetime.now() + __import__("datetime").timedelta(seconds=3600)

    with patch.object(Credentials, "refresh", mock_refresh):
        token = get_oauth_token()

        # Verify new token is returned
        assert token == "new_token_789"

        # Verify file was updated
        updated_creds = _load_json_from_plugin_cache(OAUTH_CREDENTIALS_FILE)
        assert updated_creds["access_token"] == "new_token_789"
        assert "expiry_date" in updated_creds


@pytest.mark.usefixtures("mock_llm_user_path")
def test_oauth_token_refresh_missing_refresh_token(tmpdir, monkeypatch):
    """Test OAuth refresh fails when refresh_token is missing"""
    # Create expired token without refresh_token
    expiry_date = datetime.utcnow().timestamp() - 100
    _save_json_to_plugin_cache(
        OAUTH_CREDENTIALS_FILE, {"access_token": "expired_token", "expiry_date": expiry_date}
    )

    with pytest.raises(AuthenticationError, match="no refresh_token is available"):
        get_oauth_token()


@pytest.mark.vcr
@pytest.mark.usefixtures("mock_llm_user_path")
def test_oauth_token_refresh_failed_request():
    """Test OAuth refresh fails when HTTP request fails"""

    # Create expired token with a dummy refresh token that will be censored
    expiry_date = int(datetime.utcnow().timestamp() - 100) * 1000
    _save_json_to_plugin_cache(
        OAUTH_CREDENTIALS_FILE,
        {
            "access_token": "expired_token",
            "expiry_date": expiry_date,
            "refresh_token": "invalid_refresh",  # This will be censored to CENSORED-REFRESH-TOKEN
            "client_id": "client_id_123",
            "client_secret": "client_secret_123",
            "scope": " ".join(SCOPES),
        },
    )

    # Google's library will make real request and fail
    with pytest.raises(AuthenticationError, match="invalid_grant"):
        assert get_oauth_token()


@pytest.mark.usefixtures("mock_llm_user_path")
def test_oauth_header_generation():
    """Test that OAuth tokens generate Bearer auth headers"""

    # Create valid token
    _save_json_to_plugin_cache(
        OAUTH_CREDENTIALS_FILE,
        {
            "access_token": "test_oauth_token",
            "expiry_date": int(datetime.utcnow().timestamp() + 3600) * 1000,
        },
    )

    # Test with model
    model: _SharedGemini = llm.get_model("gemini-2.5-flash-lite-ca")

    # The get_auth_headers method should return Bearer token
    headers = model.get_auth_headers()
    assert headers == {"Authorization": "Bearer test_oauth_token"}


@pytest.mark.skipif(
    os.environ.get("PYTEST_GEMINI_OAUTH_TOKEN") is None,
    reason="This is a live integration test and requires PYTEST_GEMINI_OAUTH_TOKEN",
)
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_llm_user_path")
async def test_oauth_integration():
    """Integration test: Make actual API call using OAuth token from cache"""

    # Get OAuth token from environment for real testing
    # This allows developers to run: PYTEST_GEMINI_OAUTH_TOKEN=<token> pytest
    oauth_token = os.environ.get("PYTEST_GEMINI_OAUTH_TOKEN", "oauth-test-token")

    # Create OAuth cache file with token
    _save_json_to_plugin_cache(
        OAUTH_CREDENTIALS_FILE,
        {
            "access_token": oauth_token,
            "expiry_date": int(datetime.utcnow().timestamp() + 3600) * 1000,
        },
    )

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
