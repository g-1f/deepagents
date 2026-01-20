# Claude Code Task: Backport DeepAgents with Skills Architecture

## Mission

Modify the `langchain-ai/deepagents` repo to create a **middleware-free implementation** that supports skills-based agent architecture. The goal is backward compatibility with older LangChain versions (pre-1.0) while maintaining the core deep agent capabilities.

## Context

- Current deepagents uses LangChain 1.0 `AgentMiddleware` for composability
- Our SDK doesn't support middleware yet
- Need: Skills (isolated subagents) configurable via prompts, not code
- Target name: "Alpha Cortex" style trading agent infrastructure

## Repository

```bash
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents
```

## Core Architecture to Implement

```
SUPERVISOR AGENT
├── Planning Tools (write_todos, read_todos)
├── Filesystem Tools (ls, read_file, write_file, etc.)
└── Skill Invocation Tool (invoke_skill)
    ├── Skill A (isolated context, custom tools)
    ├── Skill B (isolated context, custom tools)
    └── Skill C (isolated context, custom tools)
```

## Files to Create

### 1. `libs/deepagents/deepagents/skills/base.py`

Skill base class with:
- `SkillConfig` dataclass (name, description, system_prompt, tools, model, settings)
- Abstract `Skill` class with `build_graph()` and `invoke()` methods  
- `SimpleSkill` implementation using ReAct pattern

### 2. `libs/deepagents/deepagents/skills/registry.py`

Skill management with:
- `SkillRegistry` class for registration/lookup
- `load_from_config()` for YAML/JSON skill definitions
- `create_invoke_skill_tool()` to generate the master routing tool

### 3. `libs/deepagents/deepagents/skills/executor.py`

Isolated execution with:
- `SkillExecutor` class for context-isolated skill invocation
- Support for parallel skill execution
- Timeout handling and error recovery

### 4. `libs/deepagents/deepagents/graph_no_middleware.py`

Main supervisor agent without middleware:
- `AgentState` combining all state (messages, todos, files, skill_results)
- Built-in tool creation (planning, filesystem)
- `create_supervisor_agent()` using direct LangGraph patterns
- `create_skill_agent()` convenience function (replaces `create_deep_agent()`)

### 5. `libs/deepagents/deepagents/config/skills.yaml`

Example configuration for trading skills:
- market_research
- quant_analysis  
- trade_planning

## Key Design Decisions

1. **No middleware dependency**: Use direct LangGraph StateGraph patterns
2. **Config-driven skills**: PMs can modify behavior via YAML, not code
3. **Isolated execution**: Each skill runs in separate context, returns summary
4. **Composable state**: Single AgentState class replaces middleware state schemas

## Implementation Order

```bash
# 1. Create skill infrastructure
touch libs/deepagents/deepagents/skills/__init__.py
# Implement base.py, registry.py, executor.py

# 2. Create middleware-free supervisor  
# Implement graph_no_middleware.py

# 3. Add trading skill templates
mkdir -p libs/deepagents/deepagents/skills/builtin
# Implement trading.py

# 4. Add configuration example
mkdir -p libs/deepagents/deepagents/config
# Create skills.yaml

# 5. Update exports
# Modify __init__.py to export create_skill_agent

# 6. Test
pytest libs/deepagents/tests/ -v
```

## Usage After Implementation

```python
from deepagents import create_skill_agent

# Inline skill definition
agent = create_skill_agent(
    model="anthropic:claude-sonnet-4-20250514",
    system_prompt="You are Alpha Cortex, a trading AI assistant.",
    tools=[market_data_tool, news_api],
    skills=[
        {
            "name": "market_research",
            "description": "Research market conditions and sentiment",
            "system_prompt": "You are an expert market researcher...",
            "tools": ["web_search", "news_api"]
        },
        {
            "name": "quant_analysis", 
            "description": "Quantitative analysis of market data",
            "system_prompt": "You are a quant analyst...",
            "tools": ["execute_code", "market_data_tool"]
        }
    ]
)

# Or load from config
agent = create_skill_agent(
    skills_config_path="config/skills.yaml"
)

# Execute
result = await agent.ainvoke({
    "messages": [HumanMessage(content="Analyze AAPL for potential entry")]
})
```

## Critical Constraints

- Must work with `langchain>=0.2.0` and `langgraph>=0.1.0` (pre-middleware)
- No `from langchain.agents.middleware import AgentMiddleware`
- Skill isolation: separate message history, returns summary only
- Maintain compatibility with existing deepagents tool schemas

## Verification Checklist

- [ ] `create_skill_agent()` creates working compiled graph
- [ ] Skills can be defined via Python dict or YAML
- [ ] Skills execute in isolated context
- [ ] Supervisor can invoke skills via `invoke_skill` tool
- [ ] Planning tools (todos) work without middleware
- [ ] Filesystem tools work without middleware
- [ ] Summarization triggers at token threshold
- [ ] Tests pass

Start with `skills/base.py` and work through the implementation order. Ask clarifying questions if the architecture needs adjustment.
