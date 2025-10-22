<div align="center">

# DiffMind

Human‑quality commit messages. Zero hassle.

AI‑assisted commit message generator with a delightful terminal UX and a one‑command Git hook. Works great offline (heuristics) and gets even better with OpenAI.

</div>

---

## Why DiffMind

- Instant, polished commit messages for your staged changes
- Beautiful, emoji‑friendly terminal UI (powered by Rich)
- One‑liner Git hook to auto‑fill commit messages
- Works offline by default; seamlessly upgrades to OpenAI when available
- Simple configuration with sane defaults; no vendor lock‑in
- Interactive session to refine, edit, and commit in one flow

## Try It in 60 Seconds

1) Install (PyPI or from source):

```bash
# Install from PyPI
pip install diffmind
# Or with OpenAI extras preinstalled
pip install "diffmind[ai]"

# From source (this repo), for development
poetry install
poetry run diffmind version
# Or local editable install
pip install .
```

2) First‑time setup (recommended):

```bash
diffmind init
# Detects OpenAI automatically when OPENAI_API_KEY is set
# Optionally installs a Git hook for you
```

3) Generate and commit:

```bash
# See a suggestion for staged changes
diffmind suggest

# Commit with the generated message (and stage all changes)
diffmind commit -a

# Or open an interactive refinement session
diffmind session
```

4) Add the Git hook (if you skipped it during init):

```bash
diffmind hook install
# Now `git commit` will auto‑fill a great message when the editor opens empty
```

## What It Looks Like

```text
┌──────────────────────────── Commit Message Suggestion ────────────────────────────┐
│ ✨ feat: add login form and validation (auth)                                     │
│                                                                                  │
│ - app/auth/LoginForm.tsx: +182 -0                                                │
│ - app/auth/validators.ts: +64 -0                                                 │
│ - i18n/en.json: +12 -0                                                            │
└──────────────────────────────────────────────────────────────────────────────────┘
💡 Use `diffmind commit` to commit with this message.
```

Interactive session (arrows + free text):

```text
✅ Commit   🔁 Regenerate   ✏️ Edit subject/body   📝 Open in $EDITOR   ➕ Add bullet
```

## Providers

- simple (default) — Fast, offline heuristics that analyze your staged diff and file paths
- openai (optional) — OpenAI chat completion for highly polished messages

Enable OpenAI automatically:

```bash
pip install openai   # or: poetry add openai --group ai
export OPENAI_API_KEY=sk-...
# DiffMind will auto‑switch to OpenAI when both the key and package are present
```

Or run the guided setup:

```bash
diffmind config wizard
```

## Configuration

DiffMind reads configuration from the first existing file:

- .diffmind.toml (repo)
- ~/.config/diffmind/config.toml (user)

Example .diffmind.toml:

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

- DIFFMIND_PROVIDER
- DIFFMIND_EMOJIS (1/0, true/false)
- DIFFMIND_CONVENTIONAL (1/0, true/false)
- OPENAI_API_KEY (for the OpenAI provider)

## Git Hook

- Installs prepare-commit-msg that calls DiffMind to prefill an empty message
- Respects existing messages (never overwrites non‑comment content)

Commands:

```bash
diffmind hook install
# ... later
diffmind hook uninstall
```

## Troubleshooting

- OpenAI not detected?
  - Ensure the openai package is installed and OPENAI_API_KEY is set.
  - Run: `diffmind doctor` for a quick status check.
- No staged changes?
  - DiffMind only considers staged files. Run `git add -A` first or use `diffmind commit -a`.
- Want to edit the message manually?
  - Use `diffmind session` and choose “Open in $EDITOR”.

## Security & Privacy

- The simple provider never sends data anywhere and runs locally.
- The OpenAI provider sends only the staged diff and prompt context to the configured OpenAI endpoint.
- You control when OpenAI is used (auto/explicit) and can disable it at any time.

## Contributing

Contributions are welcome! If you plan to add a provider or improve heuristics, please keep the UX consistent and simple. Open an issue to discuss ideas.

## License

MIT
