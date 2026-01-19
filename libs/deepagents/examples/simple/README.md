# Simple Example

A minimal example showing the skills plugin pattern.

## Setup

```bash
export ANTHROPIC_API_KEY='your-key-here'
pip install langchain-anthropic langgraph
```

## Run

```bash
# Default query
python run.py

# Custom query
python run.py "Write a Python function to calculate fibonacci numbers"

# Minimal example (no skills)
python minimal.py
```

## Adding Skills

Create a `.py` file in `skills/` with a `register()` function:

```python
# skills/my_skill.py
def register(registry):
    registry.register({
        "name": "my-skill",
        "description": "What this skill does",
        "system_prompt": "You are an expert at...",
        "tools": [],
    })
```

The skill will be automatically discovered.

## Files

- `run.py` - Main example with skills
- `minimal.py` - Simplest possible agent
- `skills/researcher.py` - Research skill plugin
- `skills/coder.py` - Coding skill plugin
