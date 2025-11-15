# llm-gemini-code-assist

API access to Google's Gemini models via the Code Assist API with OAuth authentication.

This is a fork of [llm-gemini](https://github.com/simonw/llm-gemini) modified to use Google's Code Assist API, which requires OAuth authentication instead of API keys.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/):

```bash
llm install llm-gemini-code-assist
```

## Authentication

Unlike the standard llm-gemini plugin, this version uses OAuth authentication with the Code Assist API. Authentication generally is automatic if you have gemini-cli installed; else, you can authenticate manually:

```bash
llm gemini auth
```

This will:
1. Open your browser to Google's OAuth consent page
2. After you approve, save credentials to `~/.gemini/oauth_creds.json`
3. The credentials include a refresh token for automatic renewal

The OAuth credentials are stored with file permissions `0600` for security.

## Usage

Once authenticated, use the models with the `gemini-ca/` prefix:

```bash
llm -m gemini-ca/gemini-2.5-flash "Tell me a joke about a pelican"
```

You can set it as your default model:

```bash
llm models default gemini-ca/gemini-2.5-flash
llm "Tell me a joke about a pelican"
```

## Available Models

Only a limited subset of models from the standard llm-gemini plugin are available with the `gemini-ca/` prefix:

- `gemini-ca/gemini-2.5-pro` - Latest Gemini 2.5 Pro
- `gemini-ca/gemini-2.5-flash` - Latest Gemini 2.5 Flash
- `gemini-ca/gemini-2.5-flash-lite` - Gemini 2.5 Flash Lite

## Features

All features from llm-gemini are supported:

### Multi-modal Input

```bash
llm -m gemini-ca/gemini-2.5-flash 'describe image' -a image.jpg
```

### JSON Output

```bash
llm -m gemini-ca/gemini-2.5-flash -o json_object 1 \
  '3 largest cities in California'
```

### Code Execution

```bash
llm -m gemini-ca/gemini-2.0-flash -o code_execution 1 \
  'calculate factorial of 13'
```

### Google Search

```bash
llm -m gemini-ca/gemini-2.5-flash -o google_search 1 \
  'What happened today?'
```

### Chat

```bash
llm chat -m gemini-ca/gemini-2.5-flash
```

## Troubleshooting

If you get authentication errors:

1. Check if your credentials are expired:
   ```bash
   cat ~/.gemini/oauth_creds.json | python -m json.tool
   ```

2. Re-authenticate:
   ```bash
   llm gemini-ca auth
   ```

## Development

To set up the development environment:

```bash
cd llm-gemini-code-assist
uv run setup
```

This will install dependencies, set up pre-commit hooks (including secret scanning), and prepare the environment.

Run tests:

```bash
uv run pytest
```

The pre-commit hooks will automatically run linting, formatting, type checking, and secret scanning before each commit. You can also run them manually:

```bash
uv run pre-commit run --all-files
```

## Differences from llm-gemini

- Uses OAuth authentication instead of API keys
- Requires Code Assist API access
- Models use `gemini-ca/` prefix
- Tokens auto-refresh using stored refresh tokens

## License

Apache 2.0
