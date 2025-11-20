<div align="center">

# DiffMind

Human‑quality commit messages. Zero hassle.

[![PyPI](https://img.shields.io/pypi/v/diffmind.svg)](https://pypi.org/project/diffmind/) 
[![Python](https://img.shields.io/pypi/pyversions/diffmind.svg)](https://pypi.org/project/diffmind/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

AI‑assisted commit message generator with a delightful terminal UX and a one‑command Git hook. Works offline (heuristics) and gets even better with OpenAI.

</div>

---

## Highlights

- Polished commit messages for your staged changes
- Beautiful terminal UI (emoji‑friendly, powered by Rich)
- One‑liner Git hook to auto‑fill commit messages
- Works offline by default; seamlessly upgrades to OpenAI when available
- Simple config with sane defaults; interactive refine/edit session

## Demo

<div align="center">
  <img src="demo.gif" alt="DiffMind demo" width="900" />
  <br/>
  <sub><a href="demo.mp4">Watch in higher quality (MP4)</a></sub>
</div>

## Install

Requires Python 3.8+.

### PyPI

- pipx (recommended, isolated CLI)

```bash
pipx install diffmind
```

- pipx with OpenAI extras

```bash
pipx install "diffmind[ai]"
```

- pip (core, offline)

```bash
pip install -U diffmind
```

- pip with OpenAI extras

```bash
pip install -U "diffmind[ai]"
```

Upgrade / uninstall:

- pipx upgrade

```bash
pipx upgrade diffmind
```

- pip upgrade

```bash
pip install -U diffmind
```

- pipx uninstall

```bash
pipx uninstall diffmind
```

- pip uninstall

```bash
pip uninstall diffmind
```

### From Source

Option A — editable install (pip):

- Clone

```bash
git clone https://github.com/dirusanov/DiffMind.git
```

- Enter directory

```bash
cd DiffMind
```

- Install (editable) with OpenAI extras

```bash
pip install -e ".[ai]"
```

- Or install (editable) core only

```bash
pip install -e .
```

Option B — development setup (Poetry):

- Clone

```bash
git clone https://github.com/dirusanov/DiffMind.git
```

- Enter directory

```bash
cd DiffMind
```

- Install dependencies

```bash
poetry install
```

- Verify CLI in venv

```bash
poetry run diffmind --help
```

## Quickstart

```bash
# One-time setup (detects OpenAI automatically if OPENAI_API_KEY is set)
diffmind init

# Suggest a message for staged changes
diffmind suggest

# Commit with the generated message (and stage all changes)
diffmind commit -a

# Interactive refinement session (arrows + free text)
diffmind session
```

Add the Git hook (if you skipped it during init):

```bash
diffmind hook install
```

## What It Looks Like

```text
┌──────────────────────────── Commit Message Suggestion ────────────────────────────┐
│ ✨ feat: add login form and validation (auth)                                     │
│                                                                                  │
│ - app/auth/LoginForm.tsx: +182 -0                                                │
│ - app/auth/validators.ts: +64 -0                                                 │
│ - i18n/en.json: +12 -0                                                           │
└──────────────────────────────────────────────────────────────────────────────────┘
💡 Use `diffmind commit` to commit with this message.
```

Interactive actions:

```text
✅ Commit   🔁 Regenerate   ✏️ Edit subject/body   📝 Open in $EDITOR   ➕ Add bullet
```

## Providers

- simple (default) — Fast, offline heuristics based on your staged diff and file paths
- openai (optional) — OpenAI chat completion for highly polished messages

Enable OpenAI:

- Set API key

```bash
export OPENAI_API_KEY=sk-...
```

- Install with extras (pip)

```bash
pip install -U "diffmind[ai]"
```

- Or install provider only (pip)

```bash
pip install -U openai
```

Or run the guided setup:

```bash
diffmind config wizard
```

## Configuration

DiffMind reads configuration from the first existing file:

- `.diffmind.toml` (repo)
- `~/.config/diffmind/config.toml` (user)

Example `.diffmind.toml`:

```toml
provider = "auto"        # auto | simple | openai
conventional = true       # Conventional Commit types
emojis = true             # emoji prefix in subject
max_subject_length = 72
scope_strategy = "topdir" # topdir | none
language = "auto"         # auto | en | ru

# OpenAI (optional)
openai_model = "gpt-4o-mini"
# openai_base_url = "https://api.openai.com/v1"
```

Environment overrides:

- `DIFFMIND_PROVIDER`
- `DIFFMIND_EMOJIS` (1/0, true/false)
- `DIFFMIND_CONVENTIONAL` (1/0, true/false)
- `OPENAI_API_KEY` (for the OpenAI provider)

## Git Hook

- Installs `prepare-commit-msg` that pre-fills an empty message using DiffMind
- Respects existing messages (never overwrites non‑comment content)

Commands:

```bash
diffmind hook install
# ... later
diffmind hook uninstall
```

## Troubleshooting

- OpenAI not detected? Ensure the `openai` package is installed and `OPENAI_API_KEY` is set. Run `diffmind doctor`.
- No staged changes? DiffMind only considers staged files. `git add -A` first or use `diffmind commit -a`.
- Prefer manual edits? Use `diffmind session` and choose “Open in $EDITOR”.

## Security & Privacy

- The simple provider never sends code anywhere and runs locally.
- The OpenAI provider sends only the staged diff and prompt context to your configured OpenAI endpoint.
- You control when OpenAI is used (auto/explicit) and can disable it any time.

## Contributing

Contributions are welcome! If you plan to add a provider or improve agent flows/prompts, please keep the UX consistent and simple. Open an issue to discuss ideas.

## License

MIT
