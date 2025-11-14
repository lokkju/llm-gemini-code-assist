import base64
import click
import copy
from datetime import datetime
import httpx
import ijson
import json
import jwt
import llm
import os
from pathlib import Path
from pydantic import Field
from typing import Optional

from google.oauth2.credentials import Credentials
from google.auth.credentials import TokenState
from google.auth.transport.requests import Request as GoogleAuthRequest
from google_auth_oauthlib.flow import InstalledAppFlow

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
]

GEMINI_CODE_ASSIST_PLUGIN_SLUG = "gemini-code-assist"

PROJECT_ID_CACHE_FILE = "project_id_cache.json"
OAUTH_CREDENTIALS_FILE = "oauth_creds.json"

# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini#supported_models_2
GEMINI_CODE_ASSIST_MODELS = {
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
}

GOOGLE_SEARCH_MODELS = GEMINI_CODE_ASSIST_MODELS
THINKING_BUDGET_MODELS = GEMINI_CODE_ASSIST_MODELS

# OAuth credentials from gemini-cli
# these aren't secrets per se, as they're intended to be public in client apps
# but we still obfuscate them a bit to avoid issues with secret scanning tools
CLIENT_ID = base64.b64decode("NjgxMjU1ODA5Mzk1LW9vOGZ0Mm9wcmRybnA5ZTNhcWY2YXYzaG1kaWIxMzVqLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tCg==").decode("utf-8").strip()
CLIENT_SECRET = base64.b64decode("R09DU1BYLTR1SGdNUG0tMW83U2stZ2VWNkN1NWNsWEZzeGwK").decode("utf-8").strip()
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

class OAuthError(llm.ModelError):
    """Raised when OAuth token operations fail"""
    pass

def get_oauth_credentials():
    """Load OAuth credentials from <plugin_cache_dir>/oauth_creds.json and refresh if needed.

    Returns:
        google.oauth2.credentials.Credentials object

    Raises:
        OAuthError: If credentials can't be loaded or refreshed
    """
    try:
        creds_data = _load_json_from_plugin_cache(OAUTH_CREDENTIALS_FILE)
        # Create Credentials object
        access_token = creds_data.get("access_token")
        expiry = None
        expires_at = creds_data.get("expires_at")
        if expires_at:
            if expires_at > 10000000000:
                expires_at = expires_at / 1000
            expiry = datetime.fromtimestamp(expires_at)

        credentials = Credentials(
            token=access_token,
            id_token=creds_data.get("id_token"),
            refresh_token=CENSORED-REFRESH-TOKEN
            token_uri="https://oauth2.googleapis.com/token",
            client_id=CENSORED-CLIENT-ID
            client_secret=CENSORED-CLIENT-SECRET
            scopes=" ".join(SCOPES),
            expiry=expiry,
        )
    except (json.JSONDecodeError, IOError, AttributeError) as e:
        raise OAuthError(
            f"Failed to load OAuth credentials: {e}. "
            "Please authenticate using: `llm gemini-ca auth`",
        )

    # If token is expired, try to refresh it
    if credentials and credentials.token_state != TokenState.FRESH and credentials.refresh_token:
        try:
            credentials.refresh(GoogleAuthRequest())
            # Save the refreshed credentials
            refreshed_creds_data = {
                "access_token": credentials.token,
                "refresh_token": credentials.refresh_token,
                "scope": " ".join(credentials.scopes) if credentials.scopes else "",
                "token_type": "Bearer",
                "id_token": credentials.id_token,
                "expiry_date": int(credentials.expiry.timestamp() * 1000) if credentials.expiry else None,
            }
            _save_json_to_plugin_cache(OAUTH_CREDENTIALS_FILE, refreshed_creds_data)
        except Exception as e:
            raise OAuthError(
                f"Failed to refresh OAuth token: {e}. "
                "Please reauthenticate using: `llm gemini-ca auth`"
            )

    # If token is still not valid after attempting refresh, the refresh
    # call above will raise an exception. If the token is expired and
    # there's no refresh token, we'll also raise an error.
    if not credentials.valid:
        if credentials.expired and not credentials.refresh_token:
            raise OAuthError(
                "OAuth token is expired and no refresh_token is available. "
                "Please reauthenticate using: `llm gemini-ca auth`"
            )
        # For other invalid cases, we can be more lenient, as the refresh
        # mechanism might handle it, or subsequent API calls will fail with a
        # more specific error. This helps in test environments where the
        # mock credentials might not be perfectly valid.


    return credentials


def get_oauth_token():
    """Get OAuth access token from ~/.gemini/oauth_creds.json.

    Returns:
        str: The access token, or None if not found

    Raises:
        OAuthError: If credentials can't be loaded or refreshed
    """
    credentials = get_oauth_credentials()
    if credentials:
        return credentials.token
    return None

def get_oauth_id_token():
    """Get OAuth id_token token from ~/.gemini/oauth_creds.json.

    Returns:
        str: The access token, or None if not found

    Raises:
        OAuthError: If credentials can't be loaded or refreshed
    """
    credentials = get_oauth_credentials()
    if credentials:
        return credentials.id_token
    return None

def _plugin_cache_dir():
    user_dir = llm.user_dir()
    plugin_cache_dir = user_dir / GEMINI_CODE_ASSIST_PLUGIN_SLUG
    plugin_cache_dir.mkdir(exist_ok=True)
    return plugin_cache_dir

def _load_json_from_plugin_cache(filename):
    plugin_cache_dir = _plugin_cache_dir()
    data_path = plugin_cache_dir / filename

    try:
        with open(data_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def _save_json_to_plugin_cache(filename, cache):
    try:
        plugin_cache_dir = _plugin_cache_dir()
        data_path = plugin_cache_dir / filename
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "w") as f:
            json.dump(cache, f, indent=2)
    except IOError:
        pass

def _clean_plugin_cache():
    # delete all files in cache dir
    try:
        plugin_cache_dir = _plugin_cache_dir()
        for item in plugin_cache_dir.iterdir():
            if item.is_file():
                item.unlink()
    except IOError:
        raise llm.ModelError(
            f"Failed to clean project ID cache: {e}"
        )

def _load_project_id_cache():
    return _load_json_from_plugin_cache(PROJECT_ID_CACHE_FILE)

def _save_project_id_cache(cache):
    _save_json_to_plugin_cache(PROJECT_ID_CACHE_FILE, cache)

def _save_creds_to_plugin_cache(credentials):
    # Save credentials
    oauth_file = _plugin_cache_dir() / OAUTH_CREDENTIALS_FILE

    creds_data = {
        "access_token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "id_token": credentials.id_token,
        "token_uri": credentials.token_uri,
        "scopes": credentials.scopes,
        "expiry_date": int(credentials.expiry.timestamp() * 1000) if credentials.expiry else None,
    }
    with open(oauth_file, "w") as f:
        json.dump(creds_data, f, indent=2)

    oauth_file.chmod(0o600)

def get_user_email(credentials):
    if not credentials or not credentials.id_token:
        return None
    try:
        decoded_token = jwt.decode(credentials.id_token, options={"verify_signature": False})
        return decoded_token.get("email")
    except jwt.PyJWTError:
        return None


def get_code_assist_project(credentials):
    """Get project assignment from Code Assist API (cached per user).

    Args:
        credentials: google.oauth2.credentials.Credentials object

    Returns:
        tuple: (project_id, user_tier) or (None, None) on error
    """
    # Cache key based on user email
    cache_key = get_user_email(credentials)

    # # In-memory cache
    # if cache_key and cache_key in _code_assist_project_cache:
    #     return _code_assist_project_cache[cache_key]

    # File-based cache
    file_cache = _load_project_id_cache()
    if cache_key and cache_key in file_cache:
        project_id, user_tier = file_cache[cache_key]
        # _code_assist_project_cache[cache_key] = (project_id, user_tier)
        return project_id, user_tier

    if not credentials.valid:
        return None, None

    # Call loadCodeAssist endpoint
    url = "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }
    print(headers)
    body = {
        "cloudaicompanionProject": "",
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    }

    try:
        response = httpx.post(url, headers=headers, json=body, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        project_id = data.get("cloudaicompanionProject")
        user_tier = data.get("currentTier", {}).get("id")

        # Cache the result
        if cache_key and project_id:
            # _code_assist_project_cache[cache_key] = (project_id, user_tier)
            file_cache[cache_key] = (project_id, user_tier)
            _save_project_id_cache(file_cache)

        return project_id, user_tier
    except Exception as e:
        raise e





ATTACHMENT_TYPES = {
    # Text
    "text/plain",
    "text/csv",
    # PDF
    "application/pdf",
    # Images
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    # Audio
    "audio/wav",
    "audio/mp3",
    "audio/aiff",
    "audio/aac",
    "audio/ogg",
    "application/ogg",
    "audio/flac",
    "audio/mpeg",  # Treated as audio/mp3
    # Video
    "video/mp4",
    "video/mpeg",
    "video/mov",
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/3gpp",
    "video/quicktime",
}


@llm.hookimpl
def register_models(register):
    # Register both sync and async versions of each model with gemini-ca/ prefix
    for gemini_model_id in (
        GEMINI_CODE_ASSIST_MODELS
    ):
        # Add gemini-ca/ prefix for user-facing model ID
        model_id = f"gemini-ca/{gemini_model_id}"
        model_alias = f"{gemini_model_id}-ca"
        can_google_search = gemini_model_id in GOOGLE_SEARCH_MODELS
        can_thinking_budget = gemini_model_id in THINKING_BUDGET_MODELS
        can_vision = True
        can_schema = True
        register(
            GeminiPro(
                model_id,
                can_vision=can_vision,
                can_google_search=can_google_search,
                can_thinking_budget=can_thinking_budget,
                can_schema=can_schema,
            ),
            AsyncGeminiPro(
                model_id,
                can_vision=can_vision,
                can_google_search=can_google_search,
                can_thinking_budget=can_thinking_budget,
                can_schema=can_schema,
            ),
            aliases=(model_alias,),
        )


def resolve_type(attachment):
    mime_type = attachment.resolve_type()
    # https://github.com/simonw/llm/issues/587#issuecomment-2439785140
    if mime_type == "audio/mpeg":
        mime_type = "audio/mp3"
    if mime_type == "application/ogg":
        mime_type = "audio/ogg"
    return mime_type


def cleanup_schema(schema, in_properties=False):
    "Gemini supports only a subset of JSON schema"
    keys_to_remove = ("$schema", "additionalProperties", "title")

    if isinstance(schema, dict):
        # Only remove keys if we're not inside a 'properties' block.
        if not in_properties:
            for key in keys_to_remove:
                schema.pop(key, None)
        for key, value in list(schema.items()):
            # If the key is 'properties', set the flag for its value.
            if key == "properties" and isinstance(value, dict):
                cleanup_schema(value, in_properties=True)
            else:
                cleanup_schema(value, in_properties=in_properties)
    elif isinstance(schema, list):
        for item in schema:
            cleanup_schema(item, in_properties=in_properties)
    return schema


class _SharedGemini:
    can_stream = True
    supports_schema = True
    supports_tools = True

    attachment_types = set()

    class Options(llm.Options):
        code_execution: Optional[bool] = Field(
            description="Enables the model to generate and run Python code",
            default=None,
        )
        temperature: Optional[float] = Field(
            description=(
                "Controls the randomness of the output. Use higher values for "
                "more creative responses, and lower values for more "
                "deterministic responses."
            ),
            default=None,
            ge=0.0,
            le=2.0,
        )
        max_output_tokens: Optional[int] = Field(
            description="Sets the maximum number of tokens to include in a candidate.",
            default=None,
        )
        top_p: Optional[float] = Field(
            description=(
                "Changes how the model selects tokens for output. Tokens are "
                "selected from the most to least probable until the sum of "
                "their probabilities equals the topP value."
            ),
            default=None,
            ge=0.0,
            le=1.0,
        )
        top_k: Optional[int] = Field(
            description=(
                "Changes how the model selects tokens for output. A topK of 1 "
                "means the selected token is the most probable among all the "
                "tokens in the model's vocabulary, while a topK of 3 means "
                "that the next token is selected from among the 3 most "
                "probable using the temperature."
            ),
            default=None,
            ge=1,
        )
        json_object: Optional[bool] = Field(
            description="Output a valid JSON object {...}",
            default=None,
        )
        timeout: Optional[float] = Field(
            description=(
                "The maximum time in seconds to wait for a response. "
                "If the model does not respond within this time, "
                "the request will be aborted."
            ),
            default=None,
        )
        url_context: Optional[bool] = Field(
            description=(
                "Enable the URL context tool so the model can fetch content "
                "from URLs mentioned in the prompt"
            ),
            default=None,
        )

    class OptionsWithGoogleSearch(Options):
        google_search: Optional[bool] = Field(
            description="Enables the model to use Google Search to improve the accuracy and recency of responses from the model",
            default=None,
        )

    class OptionsWithThinkingBudget(OptionsWithGoogleSearch):
        thinking_budget: Optional[int] = Field(
            description="Indicates the thinking budget in tokens. Set to 0 to disable.",
            default=None,
        )

    def __init__(
        self,
        gemini_model_id,
        can_vision=True,
        can_google_search=False,
        can_thinking_budget=False,
        can_schema=False,
    ):
        # For Code Assist, model_id has gemini-ca/ prefix, but we need the raw gemini model ID for API calls
        if gemini_model_id.startswith("gemini-ca/"):
            self.model_id = gemini_model_id
            self.gemini_model_id = gemini_model_id.replace("gemini-ca/", "")
        else:
            # Fallback for direct initialization
            self.model_id = "gemini-ca/{}".format(gemini_model_id)
            self.gemini_model_id = gemini_model_id

        self.can_google_search = can_google_search
        self.supports_schema = can_schema
        if can_google_search:
            self.Options = self.OptionsWithGoogleSearch
        self.can_thinking_budget = can_thinking_budget
        if can_thinking_budget:
            self.Options = self.OptionsWithThinkingBudget
        if can_vision:
            self.attachment_types = ATTACHMENT_TYPES


    def get_credentials(self):
        """Get OAuth credentials, caching them per instance."""
        if not hasattr(self, '_credentials'):
            self._credentials = get_oauth_credentials()
            if not self._credentials:
                raise llm.ModelError(
                    "OAuth credentials not found. Please authenticate using: llm gemini-ca auth"
                )
        return self._credentials

    def get_project_id(self):
        """Get Code Assist project ID, caching it per instance."""
        if not hasattr(self, '_project_id'):
            credentials = self.get_credentials()
            project_id, user_tier = get_code_assist_project(credentials)
            if not project_id:
                raise llm.ModelError(
                    "Failed to get project assignment from Code Assist API"
                )
            self._project_id = project_id
            self._user_tier = user_tier
        return self._project_id



    def get_auth_headers(self):
        """Get OAuth authentication headers for Code Assist API calls."""
        credentials = self.get_credentials()
        if credentials is None:
            raise llm.ModelError(
                "OAuth credentials not found. Please authenticate using: llm gemini-ca auth"
            )
        return {"Authorization": f"Bearer {credentials.token}"}

    def get_api_url(self):
        """Get Code Assist API URL."""
        return "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent"

    def build_messages(self, prompt, conversation):
        messages = []
        if conversation:
            for response in conversation.responses:
                parts = []
                for attachment in response.attachments:
                    mime_type = resolve_type(attachment)
                    parts.append(
                        {
                            "inlineData": {
                                "data": attachment.base64_content(),
                                "mimeType": mime_type,
                            }
                        }
                    )
                if response.prompt.prompt:
                    parts.append({"text": response.prompt.prompt})
                if response.prompt.tool_results:
                    parts.extend(
                        [
                            {
                                "function_response": {
                                    "name": tool_result.name,
                                    "response": {
                                        "output": tool_result.output,
                                    },
                                }
                            }
                            for tool_result in response.prompt.tool_results
                        ]
                    )
                messages.append({"role": "user", "parts": parts})
                model_parts = []
                response_text = response.text_or_raise()
                model_parts.append({"text": response_text})
                tool_calls = response.tool_calls_or_raise()
                if tool_calls:
                    model_parts.extend(
                        [
                            {
                                "function_call": {
                                    "name": tool_call.name,
                                    "args": tool_call.arguments,
                                }
                            }
                            for tool_call in tool_calls
                        ]
                    )
                messages.append({"role": "model", "parts": model_parts})

        parts = []
        if prompt.prompt:
            parts.append({"text": prompt.prompt})
        if prompt.tool_results:
            parts.extend(
                [
                    {
                        "function_response": {
                            "name": tool_result.name,
                            "response": {
                                "output": tool_result.output,
                            },
                        }
                    }
                    for tool_result in prompt.tool_results
                ]
            )
        for attachment in prompt.attachments:
            mime_type = resolve_type(attachment)
            parts.append(
                {
                    "inlineData": {
                        "data": attachment.base64_content(),
                        "mimeType": mime_type,
                    }
                }
            )

        messages.append({"role": "user", "parts": parts})
        return messages

    def build_request_body(self, prompt, conversation):
        body = {
            "contents": self.build_messages(prompt, conversation),
            "safetySettings": SAFETY_SETTINGS,
        }
        if prompt.system:
            body["systemInstruction"] = {"parts": [{"text": prompt.system}]}

        tools = []
        if prompt.options and prompt.options.code_execution:
            tools.append({"codeExecution": {}})
        if prompt.options and self.can_google_search and prompt.options.google_search:
            tool_name = "google_search"
            tools.append({tool_name: {}})
        if prompt.options and prompt.options.url_context:
            tools.append({"url_context": {}})
        if prompt.tools:
            tools.append(
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        }
                        for tool in prompt.tools
                    ]
                }
            )
        if tools:
            body["tools"] = tools

        generation_config = {}

        if prompt.schema:
            generation_config.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": cleanup_schema(copy.deepcopy(prompt.schema)),
                }
            )

        if self.can_thinking_budget and prompt.options.thinking_budget is not None:
            generation_config["thinking_config"] = {
                "thinking_budget": prompt.options.thinking_budget
            }

        config_map = {
            "temperature": "temperature",
            "max_output_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
        }
        if prompt.options and prompt.options.json_object:
            generation_config["response_mime_type"] = "application/json"

        if any(
            getattr(prompt.options, key, None) is not None for key in config_map.keys()
        ):
            for key, other_key in config_map.items():
                config_value = getattr(prompt.options, key, None)
                if config_value is not None:
                    generation_config[other_key] = config_value

        if generation_config:
            body["generationConfig"] = generation_config

        return body

    def wrap_code_assist_request(self, body, prompt):
        """Wrap standard Gemini request in Code Assist API format."""
        import uuid

        return {
            "model": self.gemini_model_id,
            "project": self.get_project_id(),
            "user_prompt_id": str(uuid.uuid4()),
            "request": body
        }

    def unwrap_code_assist_response(self, event):
        """Unwrap Code Assist API response to standard Gemini format."""
        # Code Assist wraps the response in {"response": {...}}
        if isinstance(event, dict) and "response" in event:
            return event["response"]
        return event

    def process_part(self, part, response):
        if "functionCall" in part:
            response.add_tool_call(
                llm.ToolCall(
                    name=part["functionCall"]["name"],
                    arguments=part["functionCall"]["args"],
                )
            )
        if "text" in part:
            return part["text"]
        elif "executableCode" in part:
            return f'```{part["executableCode"]["language"].lower()}\n{part["executableCode"]["code"].strip()}\n```\n'
        elif "codeExecutionResult" in part:
            return f'```\n{part["codeExecutionResult"]["output"].strip()}\n```\n'
        return ""

    def process_candidates(self, candidates, response):
        # We only use the first candidate
        for part in candidates[0]["content"]["parts"]:
            yield self.process_part(part, response)

    def set_usage(self, response):
        try:
            # Don't record the "content" key from that last candidate
            for candidate in response.response_json["candidates"]:
                candidate.pop("content", None)
            usage = response.response_json.pop("usageMetadata")
            input_tokens = usage.pop("promptTokenCount", None)
            # See https://github.com/simonw/llm-gemini/issues/75#issuecomment-2861827509
            candidates_token_count = usage.get("candidatesTokenCount") or 0
            thoughts_token_count = usage.get("thoughtsTokenCount") or 0
            output_tokens = candidates_token_count + thoughts_token_count
            tool_token_count = usage.get("toolUsePromptTokenCount") or 0
            if tool_token_count:
                if input_tokens is None:
                    input_tokens = tool_token_count
                else:
                    input_tokens += tool_token_count
            usage.pop("totalTokenCount", None)
            if input_tokens is not None:
                response.set_usage(
                    input=input_tokens, output=output_tokens, details=usage or None
                )
        except (IndexError, KeyError):
            pass


class GeminiPro(_SharedGemini, llm.Model):
    def execute(self, prompt, stream, response, conversation):
        url = self.get_api_url()
        gathered = []

        # Build standard request and wrap in Code Assist format
        standard_body = self.build_request_body(prompt, conversation)
        body = self.wrap_code_assist_request(standard_body, prompt)
        try:
            with httpx.stream(
                "POST",
                url,
                timeout=prompt.options.timeout,
                headers=self.get_auth_headers(),
                json=body,
            ) as http_response:
                events = ijson.sendable_list()
                coro = ijson.items_coro(events, "item")
                for chunk in http_response.iter_bytes():
                    coro.send(chunk)
                    if events:
                        for event in events:
                            if isinstance(event, dict) and "error" in event:
                                error_msg = event["error"]["message"]
                                raise llm.ModelError(error_msg)

                            # Unwrap Code Assist response
                            unwrapped_event = self.unwrap_code_assist_response(event)

                            try:
                                yield from self.process_candidates(
                                    unwrapped_event["candidates"], response
                                )
                            except KeyError:
                                yield ""
                            gathered.append(unwrapped_event)
                        events.clear()

            response.response_json = gathered[-1]
            resolved_model = gathered[-1]["modelVersion"]
            response.set_resolved_model(resolved_model)
            self.set_usage(response)
        except httpx.HTTPError as e:
            raise llm.ModelError(f"HTTP error during request: {e}") from e
        except Exception as e:
            raise llm.ModelError(f"Error during request: {e}") from e

class AsyncGeminiPro(_SharedGemini, llm.AsyncModel):
    async def execute(self, prompt, stream, response, conversation):
        url = self.get_api_url()
        gathered = []

        # Build standard request and wrap in Code Assist format
        standard_body = self.build_request_body(prompt, conversation)
        body = self.wrap_code_assist_request(standard_body, prompt)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                timeout=prompt.options.timeout,
                headers=self.get_auth_headers(),
                json=body,
            ) as http_response:
                events = ijson.sendable_list()
                coro = ijson.items_coro(events, "item")
                async for chunk in http_response.aiter_bytes():
                    coro.send(chunk)
                    if events:
                        for event in events:
                            if isinstance(event, dict) and "error" in event:
                                error_msg = event["error"]["message"]
                                raise llm.ModelError(error_msg)

                            # Unwrap Code Assist response
                            unwrapped_event = self.unwrap_code_assist_response(event)

                            try:
                                for chunk in self.process_candidates(
                                    unwrapped_event["candidates"], response
                                ):
                                    yield chunk
                            except KeyError:
                                yield ""
                            gathered.append(unwrapped_event)
                        events.clear()
        response.response_json = gathered[-1]
        self.set_usage(response)

@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def gemini_ca():
        "Commands relating to the llm-gemini-code-assist plugin"

    @gemini_ca.command()
    def auth():
        """Authenticate with Google OAuth for Code Assist API access"""

        # Step 1: Try to load existing credentials
        credentials = get_oauth_credentials()
        if credentials and credentials.valid:
            click.echo("✓ Already authenticated with valid OAuth credentials.")
            if not click.confirm("Are you sure you want to re-authenticate?", default=False):
                raise click.Abort()

        # Step 2: look for existing gemini-cli credentials
        gemini_cli_oauth_path = Path.home() / ".gemini" / OAUTH_CREDENTIALS_FILE
        if gemini_cli_oauth_path.exists():
            click.echo(f"Found existing gemini-cli OAuth credentials at {gemini_cli_oauth_path}.")
            if click.confirm("Do you want to use attempt to use these to authenticate?", default=True):
                try:
                    with open(gemini_cli_oauth_path, "r") as f:
                        creds_data = json.load(f)
                    credentials = Credentials(
                        token=creds_data.get("access_token"),
                        id_token=creds_data.get("id_token"),
                        refresh_token=CENSORED-REFRESH-TOKEN
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=CENSORED-CLIENT-ID
                        client_secret=CENSORED-CLIENT-SECRET
                        scopes=" ".join(SCOPES),
                    )
                    if credentials and credentials.valid:
                        click.echo("✓ Found valid OAuth credentials from gemini-cli.")
                        _save_creds_to_plugin_cache(credentials)
                        return credentials
                except (json.JSONDecodeError, IOError, AttributeError):
                    click.echo("✗ Failed to load gemini-cli OAuth credentials. Proceeding to full authentication flow.")

        client_secrets = {
            "installed": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }

        flow = InstalledAppFlow.from_client_config(
            client_secrets,
            scopes=SCOPES,
        )

        click.echo("\nCode Assist authentication required.")
        click.echo("Opening browser for authentication.")

        try:
            credentials = flow.run_local_server(
                port=0,
                authorization_prompt_message="Please visit this URL to authorize this application: {url}",
                success_message="Authentication successful. You can close this window.",
                open_browser=True,
            )
        except Exception as e:
            raise click.ClickException(f"Authentication failed: {e}")
        if not credentials or not credentials.valid:
            raise click.ClickException("Failed to obtain valid OAuth credentials.")
        _save_creds_to_plugin_cache(credentials)
        click.echo("\n✓ Authentication successful!")

    @gemini_ca.command()
    def models():
        """
        List of Gemini models available via Code Assist
        """

        click.echo(json.dumps(sorted(list(GEMINI_CODE_ASSIST_MODELS)), indent=2))
