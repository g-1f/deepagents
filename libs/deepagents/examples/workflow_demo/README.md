# WorkflowSkill Demo

This demo showcases the **WorkflowSkill** architecture with:
- Dynamic subagent orchestration
- Single skill activation (progressive disclosure)
- Session management with checkpointing
- Comprehensive logging of intermediate tool calls

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MAIN AGENT                                     │
│                                                                         │
│   User: "Analyze AAPL stock"                                            │
│           │                                                             │
│           ▼                                                             │
│   ┌───────────────────┐                                                 │
│   │  activate_skill   │ ──► Discovers "stock-analyzer" skill           │
│   └───────────────────┘                                                 │
│           │                                                             │
└───────────┼─────────────────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────────────────────┐
│                   WORKFLOW SKILL: stock-analyzer                         │
│                                                                         │
│   ╔═══ STEP 1: fetch_data (SERIAL) ═══╗                                │
│   ║  market_data_fetcher               ║                               │
│   ║    └── subagent with tool          ║                               │
│   ╚════════════════════════════════════╝                               │
│                    │ output                                             │
│                    ▼                                                    │
│   ╔═══ STEP 2: parallel_analysis (PARALLEL) ═══╗                       │
│   ║  technical_analyzer   sentiment_analyzer   ║                       │
│   ║    └── subagent A       └── subagent B     ║                       │
│   ╚════════════════════════════════════════════╝                       │
│                    │ combined output                                    │
│                    ▼                                                    │
│   ╔═══ STEP 3: generate_report (SERIAL) ═══╗                           │
│   ║  report_generator                       ║                          │
│   ║    └── subagent with tool               ║                          │
│   ╚═════════════════════════════════════════╝                          │
│                    │                                                    │
└────────────────────┼────────────────────────────────────────────────────┘
                     │
                     ▼
               Final Output to Main Agent
```

## Features

### 1. Single Skill Activation (Progressive Disclosure)

The main agent doesn't need to know specific skill names. It uses `activate_skill(task)` which automatically discovers and activates the appropriate skill:

```python
# Main agent just describes the task
activate_skill("Analyze AAPL stock and provide recommendations")
# → System discovers "stock-analyzer" skill and activates it
```

### 2. Workflow Skills with Subagent Orchestration

Skills can define multi-step workflows where each step spins up isolated subagents:

```yaml
workflow:
  - tools: market_data_fetcher           # Step 1: Serial
    task_template: "Get data for {{input}}"

  - tools: [technical, sentiment]         # Step 2: Parallel (2 subagents)
    task_template: "Analyze: {{previous_output}}"

  - tools: report_generator              # Step 3: Serial
    task_template: "Report: {{previous_output}}"
```

### 3. Session Management

Conversations persist across invocations:

```bash
# Start a session
python run_workflow_demo.py --session my-analysis "Analyze AAPL"

# Continue the same session later
python run_workflow_demo.py --session my-analysis "Now compare to GOOGL"
```

### 4. Comprehensive Logging

All tool calls are logged with timing and outputs:

```
╔══════════════════════════════════════════════════════════╗
║ WORKFLOW START: stock-analyzer                            ║
║ Steps: 3                                                  ║
╚══════════════════════════════════════════════════════════╝
╔═══ WORKFLOW STEP 1: fetch_data (SERIAL) ═══
║ Tools: market_data_fetcher
┌─── SUBAGENT START: market_data_fetcher (step: fetch_data) ───
└─── SUBAGENT END: market_data_fetcher (0.52s) ───
╚═══ STEP 1 COMPLETE: fetch_data (0.52s) ═══

╔═══ WORKFLOW STEP 2: parallel_analysis (PARALLEL) ═══
║ Tools: technical_analyzer, sentiment_analyzer
║ Running 2 tools in parallel...
┌─── SUBAGENT START: technical_analyzer ───
┌─── SUBAGENT START: sentiment_analyzer ───
└─── SUBAGENT END: sentiment_analyzer (0.61s) ───
└─── SUBAGENT END: technical_analyzer (0.73s) ───
╚═══ STEP 2 COMPLETE: parallel_analysis (0.73s) ═══

╔═══ WORKFLOW STEP 3: generate_report (SERIAL) ═══
...
╔══════════════════════════════════════════════════════════╗
║ WORKFLOW COMPLETE: stock-analyzer                         ║
║ Duration: 2.45s                                           ║
╚══════════════════════════════════════════════════════════╝
```

## Usage

### Prerequisites

```bash
# Set API key
export ANTHROPIC_API_KEY='sk-ant-...'

# Install dependencies
pip install langchain-anthropic langgraph pyyaml
```

### Run Examples

```bash
# Basic stock analysis
python run_workflow_demo.py "Analyze AAPL stock"

# With debug logging
python run_workflow_demo.py --debug "Analyze AAPL stock"

# Interactive mode
python run_workflow_demo.py --interactive

# With session persistence
python run_workflow_demo.py --session my-session "Analyze AAPL"
python run_workflow_demo.py --session my-session "Compare to GOOGL"
```

### Command Line Options

```
usage: run_workflow_demo.py [-h] [--session SESSION] [--debug] [--interactive] [query]

positional arguments:
  query                 Query to process (default: stock analysis example)

optional arguments:
  -h, --help            show this help message and exit
  --session, -s         Session ID for conversation continuity
  --debug, -d           Enable debug logging
  --interactive, -i     Run in interactive mode
```

## Code Example

```python
from deepagents import create_skill_agent
from deepagents.skills import create_workflow_skill

# Define tools
tool_registry = {
    "market_data": market_data_tool,
    "technical": technical_analysis_tool,
    "sentiment": sentiment_analysis_tool,
    "report": report_generator_tool,
}

# Create agent with workflow skill
agent = create_skill_agent(
    model="anthropic:claude-sonnet-4-20250514",
    skills=[
        {
            "name": "stock-analyzer",
            "description": "Comprehensive stock analysis",
            "workflow": [
                {"tools": "market_data", "task_template": "Get data for {{input}}"},
                {"tools": ["technical", "sentiment"]},  # parallel
                {"tools": "report"},
            ],
        }
    ],
    tool_registry=tool_registry,
    use_skill_discovery=True,  # Single activation point
)

# Run - skill is auto-discovered
result = await agent.ainvoke({
    "messages": [HumanMessage(content="Analyze AAPL stock")]
})
```

## Key Concepts

### Tool Registry

Tools are registered by name and bound dynamically at runtime:

```python
tool_registry = {
    "web_search": web_search_tool,
    "calculator": calculator_tool,
    ...
}
```

### Workflow Steps

Each step can be:
- **Serial**: Single tool, runs alone
- **Parallel**: Multiple tools, run concurrently as separate subagents

### Task Templates

Templates support variable substitution:
- `{{input}}` - Original input to the skill
- `{{previous_output}}` - Output from the previous step
- `{{step_N}}` - Output from step N (0-indexed)
- `{{tool_name}}` - Output from a specific tool

### Subagent Isolation

Each tool execution runs in an isolated subagent with:
- Its own message history
- Only the relevant tool bound
- Limited iterations (max 10)
- No access to other tools or context
