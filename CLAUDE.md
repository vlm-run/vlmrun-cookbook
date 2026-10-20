# CLAUDE.md

- Never give `VLMRUN_API_KEY` a default, fallback or placeholder value in code, notebooks or docs. Read it from the
  environment and raise if it is not set: `VLMRun()` already does (`ConfigurationError: Missing API key`), so pass no
  `api_key=` and leave `require_api_key` at its default.
- Code in `demos/systemone/` follows the "Adding a demo" rules in its [README](demos/systemone/README.md).
