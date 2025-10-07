from click.testing import CliRunner
from datetime import datetime
import llm
from llm.cli import cli
import nest_asyncio
import json
import os
import pytest
import pydantic
from llm_gemini import cleanup_schema, get_oauth_token, refresh_oauth_token, OAuthError
from unittest.mock import patch, MagicMock
from pathlib import Path

nest_asyncio.apply()

GEMINI_API_KEY = os.environ.get("PYTEST_GEMINI_API_KEY", None) or "gm-..."


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt():
    model = llm.get_model("gemini-1.5-flash-latest")
    response = model.prompt("Name for a pet pelican, just the name", key=GEMINI_API_KEY)
    assert str(response) == "Percy\n"
    assert response.response_json == {
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
        "modelVersion": "gemini-1.5-flash-latest",
    }
    assert response.token_details == {
        "candidatesTokenCount": 2,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 9}],
        "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 2}],
    }
    assert response.input_tokens == 9
    assert response.output_tokens == 2

    # And try it async too
    async_model = llm.get_async_model("gemini-1.5-flash-latest")
    response = await async_model.prompt(
        "Name for a pet pelican, just the name", key=GEMINI_API_KEY
    )
    text = await response.text()
    assert text == "Percy\n"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt_with_pydantic_schema():
    class Dog(pydantic.BaseModel):
        name: str
        age: int
        bio: str

    model = llm.get_model("gemini-1.5-flash-latest")
    response = model.prompt(
        "Invent a cool dog", key=GEMINI_API_KEY, schema=Dog, stream=False
    )
    assert json.loads(response.text()) == {
        "age": 3,
        "bio": "A fluffy Samoyed with exceptional intelligence and a love for belly rubs. He's mastered several tricks, including fetching the newspaper and opening doors.",
        "name": "Cloud",
    }
    assert response.response_json == {
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
        "modelVersion": "gemini-1.5-flash-latest",
    }
    assert response.input_tokens == 10


@pytest.mark.vcr
@pytest.mark.parametrize(
    "model_id",
    (
        "gemini-embedding-exp-03-07",
        "gemini-embedding-exp-03-07-128",
        "gemini-embedding-exp-03-07-512",
    ),
)
def test_embedding(model_id, monkeypatch):
    monkeypatch.setenv("LLM_GEMINI_KEY", GEMINI_API_KEY)
    model = llm.get_embedding_model(model_id)
    response = model.embed("Some text goes here")
    expected_length = 3072
    if model_id.endswith("-128"):
        expected_length = 128
    elif model_id.endswith("-512"):
        expected_length = 512
    assert len(response) == expected_length


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
    # With no key set should error nicely
    runner = CliRunner()
    result = runner.invoke(cli, ["gemini", "models"])
    assert result.exit_code == 1
    assert "You must set the LLM_GEMINI_KEY environment variable, use --key" in result.output
    assert "or have OAuth credentials at ~/.gemini/oauth_creds.json" in result.output
    # Try again with --key
    result2 = runner.invoke(cli, ["gemini", "models", "--key", GEMINI_API_KEY])
    assert result2.exit_code == 0
    assert "gemini-1.5-flash-latest" in result2.output
    # And with --method
    result3 = runner.invoke(
        cli, ["gemini", "models", "--key", GEMINI_API_KEY, "--method", "embedContent"]
    )
    assert result3.exit_code == 0
    models = json.loads(result3.output)
    for model in models:
        assert "embedContent" in model["supportedGenerationMethods"]


@pytest.mark.vcr
def test_resolved_model():
    model = llm.get_model("gemini-flash-latest")
    response = model.prompt("hi", key=GEMINI_API_KEY)
    response.text()
    assert response.resolved_model == "gemini-2.5-flash-preview-09-2025"


@pytest.mark.vcr
def test_tools():
    model = llm.get_model("gemini-2.0-flash")
    names = ["Charles", "Sammy"]
    chain_response = model.chain(
        "Two names for a pet pelican",
        tools=[
            llm.Tool.function(lambda: names.pop(0), name="pelican_name_generator"),
        ],
        key=GEMINI_API_KEY,
    )
    text = chain_response.text()
    assert text == "Okay, here are two names for a pet pelican: Charles and Sammy.\n"
    # This one did three
    assert len(chain_response._responses) == 3
    first, second, third = chain_response._responses
    assert len(first.tool_calls()) == 1
    assert first.tool_calls()[0].name == "pelican_name_generator"
    assert len(second.tool_calls()) == 1
    assert second.tool_calls()[0].name == "pelican_name_generator"
    assert second.prompt.tool_results[0].output == "Charles"
    assert third.prompt.tool_results[0].output == "Sammy"


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
        "refresh_token": "refresh_123",
        "client_id": "client_id_123",
        "client_secret": "client_secret_123"
    }), encoding="utf-8")

    # Mock httpx.post for token refresh
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "new_token_789",
        "expires_in": 3600
    }
    mock_response.raise_for_status = MagicMock()

    with patch("llm_gemini.httpx.post", return_value=mock_response) as mock_post:
        token = get_oauth_token()

        # Verify refresh was called
        assert mock_post.called
        assert mock_post.call_args[0][0] == "https://oauth2.googleapis.com/token"

        # Verify new token is returned
        assert token == "new_token_789"

        # Verify file was updated
        updated_creds = json.loads(oauth_file.read_text(encoding="utf-8"))
        assert updated_creds["access_token"] == "new_token_789"
        assert "expires_at" in updated_creds


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
        "refresh_token": "refresh_123"
    }), encoding="utf-8")

    with pytest.raises(OAuthError, match="missing client_id or client_secret"):
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
        "refresh_token": "invalid_refresh",
        "client_id": "client_id_123",
        "client_secret": "client_secret_123"
    }), encoding="utf-8")

    # Mock httpx.post to simulate 401 error
    import httpx
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Unauthorized", request=MagicMock(), response=mock_response
    )

    with patch("llm_gemini.httpx.post", return_value=mock_response):
        with pytest.raises(OAuthError, match="invalid or revoked token"):
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
    model = llm.get_model("gemini-1.5-flash-latest")

    # The get_auth_headers method should return Bearer token
    headers = model.get_auth_headers()
    assert headers == {"Authorization": "Bearer test_oauth_token"}


def test_oauth_vs_api_key_priority(tmpdir, monkeypatch):
    """Test that API key takes priority over OAuth token"""
    oauth_dir = tmpdir / ".gemini"
    oauth_dir.mkdir()
    oauth_file = oauth_dir / "oauth_creds.json"

    monkeypatch.setattr(Path, "home", lambda: tmpdir)

    # Create OAuth token
    oauth_file.write_text(json.dumps({
        "access_token": "test_oauth_token",
        "expires_at": datetime.now().timestamp() + 3600
    }), encoding="utf-8")

    # Set API key
    monkeypatch.setenv("LLM_GEMINI_KEY", "api_key_123")

    # Test with model
    model = llm.get_model("gemini-1.5-flash-latest")

    # API key should take priority
    headers = model.get_auth_headers()
    assert headers == {"x-goog-api-key": "api_key_123"}


@pytest.mark.vcr
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
    model = llm.get_model("gemini-1.5-flash-latest")
    response = model.prompt("Name for a pet pelican, just the name")
    text = str(response)

    # Verify we got a response
    assert len(text) > 0
    assert response.response_json is not None

    # Verify the response contains expected fields
    assert "candidates" in response.response_json
    assert "modelVersion" in response.response_json

    # Also test async
    async_model = llm.get_async_model("gemini-1.5-flash-latest")
    async_response = await async_model.prompt("Name for a pet pelican, just the name")
    async_text = await async_response.text()
    assert len(async_text) > 0


@pytest.mark.vcr
def test_cli_gemini_models_with_oauth(tmpdir, monkeypatch):
    """Test that CLI gemini models command works with OAuth token"""
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))

    # Create OAuth cache directory and file
    oauth_dir = tmpdir / ".gemini"
    oauth_dir.mkdir()
    oauth_file = oauth_dir / "oauth_creds.json"

    # Mock home directory to use tmpdir
    monkeypatch.setattr(Path, "home", lambda: tmpdir)

    # Get OAuth token from environment for real testing
    oauth_token = os.environ.get("PYTEST_GEMINI_OAUTH_TOKEN", "oauth-test-token")

    # Create OAuth cache file with token
    oauth_file.write_text(json.dumps({
        "access_token": oauth_token,
        "expires_at": datetime.now().timestamp() + 3600
    }), encoding="utf-8")

    # Ensure no API key is set so it falls back to OAuth
    monkeypatch.delenv("LLM_GEMINI_KEY", raising=False)

    # Run the gemini models command
    runner = CliRunner()
    result = runner.invoke(cli, ["gemini", "models"])

    # Should succeed using OAuth
    assert result.exit_code == 0
    assert "gemini-1.5-flash-latest" in result.output

    # Test with --method filter
    result2 = runner.invoke(cli, ["gemini", "models", "--method", "embedContent"])
    assert result2.exit_code == 0
    models = json.loads(result2.output)
    for model in models:
        assert "embedContent" in model["supportedGenerationMethods"]


@pytest.mark.vcr
def test_cli_gemini_files_with_oauth(tmpdir, monkeypatch):
    """Test that CLI gemini files command works with OAuth token"""
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))

    # Create OAuth cache directory and file
    oauth_dir = tmpdir / ".gemini"
    oauth_dir.mkdir()
    oauth_file = oauth_dir / "oauth_creds.json"

    # Mock home directory to use tmpdir
    monkeypatch.setattr(Path, "home", lambda: tmpdir)

    # Get OAuth token from environment for real testing
    oauth_token = os.environ.get("PYTEST_GEMINI_OAUTH_TOKEN", "oauth-test-token")

    # Create OAuth cache file with token
    oauth_file.write_text(json.dumps({
        "access_token": oauth_token,
        "expires_at": datetime.now().timestamp() + 3600
    }), encoding="utf-8")

    # Ensure no API key is set so it falls back to OAuth
    monkeypatch.delenv("LLM_GEMINI_KEY", raising=False)

    # Run the gemini files command
    runner = CliRunner()
    result = runner.invoke(cli, ["gemini", "files"])

    # Should succeed using OAuth (might show "No files" but shouldn't error)
    assert result.exit_code == 0
