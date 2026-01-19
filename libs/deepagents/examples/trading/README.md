# Trading Agent Example

This example demonstrates how to create a trading-focused agent using the skills plugin pattern.

## Progressive Disclosure with Skills

Skills can be added dynamically by dropping Python files into a skills directory. Each skill module should define a `register(registry)` function that registers the skill.

## Usage

### Using Plugin-Based Skills (Recommended)

```python
from deepagents import create_skill_agent

# Create agent with skills loaded from directory
agent = create_skill_agent(
    model="anthropic:claude-sonnet-4-20250514",
    skills_directory="./skills/",  # Skills auto-discovered from this directory
    system_prompt="You are Alpha Cortex, an advanced AI trading assistant.",
)

# Run the agent
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Analyze AAPL stock"}]
})
```

### Using YAML Configuration

```python
from deepagents import create_skill_agent

agent = create_skill_agent(
    model="anthropic:claude-sonnet-4-20250514",
    skills_config_path="./skills.yaml",
)
```

## Adding New Skills

To add a new skill, create a Python file in the `skills/` directory:

```python
# skills/my_new_skill.py
def register(registry):
    registry.register({
        "name": "my-skill",
        "description": "What this skill does",
        "system_prompt": "You are an expert at...",
        "tools": [],  # Add LangChain tools here
    })
```

The skill will be automatically discovered and available to the agent.

## Available Skills

- **market-research**: Research market conditions, sentiment, and news
- **quant-analysis**: Technical indicators, statistical analysis, backtesting
- **trade-planning**: Entry/exit criteria, position sizing, risk management

## Directory Structure

```
trading/
├── README.md
├── skills.yaml          # Alternative: YAML-based skill definitions
└── skills/              # Plugin-based skills (recommended)
    ├── market_research.py
    ├── quant_analysis.py
    └── trade_planning.py
```
