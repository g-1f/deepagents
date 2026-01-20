# Claude Code Prompt: Backport DeepAgents with Skills Architecture

## Objective

Fork and modify `langchain-ai/deepagents` to create a **middleware-free, backward-compatible implementation** that supports:

1. **Skills-based architecture** - Configurable agent capabilities defined via prompts/configs rather than code
2. **Isolated subagent invocation** - Context-isolated execution for skill-specific tasks
3. **Compatibility with older LangChain/LangGraph versions** (pre-1.0 without native middleware support)

## Context

The latest `deepagents` library relies on LangChain 1.0's `AgentMiddleware` abstraction for composable agent capabilities. Our custom SDK hasn't been updated to support middleware yet. We need to replicate the core functionality using direct LangGraph patterns.

### Target Architecture: "Alpha Cortex" Style Skills System

```
┌─────────────────────────────────────────────────────────────────┐
│                     SUPERVISOR AGENT                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Planner   │  │   Router    │  │  Synthesizer│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                          │                                      │
│            ┌─────────────┼─────────────┐                       │
│            ▼             ▼             ▼                       │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│   │   Skill A   │ │   Skill B   │ │   Skill C   │  (Isolated) │
│   │ (Research)  │ │ (Analysis)  │ │ (Execution) │             │
│   └─────────────┘ └─────────────┘ └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Implementation Tasks

### Phase 1: Repository Setup & Analysis

```bash
# Clone the repo
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents

# Examine the current structure
tree libs/deepagents/deepagents/ -L 2

# Key files to analyze:
# - libs/deepagents/deepagents/graph.py (main agent graph construction)
# - libs/deepagents/deepagents/middleware/ (middleware implementations)
# - libs/deepagents/deepagents/middleware/subagents.py (subagent/task tool logic)
# - libs/deepagents/deepagents/middleware/filesystem.py (file operations)
# - libs/deepagents/deepagents/backends/ (storage backends)
```

Read and understand:
1. How `create_deep_agent()` assembles the middleware stack
2. How `SubAgentMiddleware` implements the `task` tool
3. How middleware injects tools, prompts, and state schemas

### Phase 2: Create Core Skill Infrastructure

Create a new module `libs/deepagents/deepagents/skills/` with the following structure:

```
skills/
├── __init__.py
├── base.py           # Base Skill class
├── registry.py       # Skill registration and discovery
├── executor.py       # Isolated skill execution engine
├── prompts.py        # Skill prompt templates
└── builtin/
    ├── __init__.py
    ├── research.py   # Research skill
    ├── analysis.py   # Data analysis skill
    └── execution.py  # Code/command execution skill
```

#### Task 2.1: Create `skills/base.py`

```python
"""
Base Skill class - defines the interface for all skills.
Skills are isolated, prompt-configurable agent capabilities.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

@dataclass
class SkillConfig:
    """Configuration for a skill, can be defined via YAML/JSON or code."""
    name: str
    description: str
    system_prompt: str
    tools: List[Union[BaseTool, Callable]] = field(default_factory=list)
    model: Optional[str] = None  # Override model for this skill
    max_iterations: int = 25
    return_intermediate_steps: bool = False
    
    # Context isolation settings
    inherit_parent_context: bool = False  # If True, receives parent's message history
    context_window_limit: int = 100000    # Token limit for skill context
    
    # State schema extensions
    state_schema: Optional[Type[BaseModel]] = None
    
    # Skill-specific settings (passed to skill implementation)
    settings: Dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """
    Base class for skills. A skill is an isolated agent capability
    that can be invoked by the supervisor agent.
    
    Unlike middleware, skills:
    - Run in isolated context (separate message history)
    - Are invoked explicitly via a "invoke_skill" tool
    - Return a summary/result to the parent agent
    - Can be configured via prompts without code changes
    """
    
    def __init__(self, config: SkillConfig):
        self.config = config
        self._compiled_graph: Optional[CompiledStateGraph] = None
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def description(self) -> str:
        return self.config.description
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph for this skill.
        Override this to customize the skill's execution flow.
        """
        pass
    
    def compile(self, **kwargs) -> CompiledStateGraph:
        """Compile the skill's graph. Call this once during setup."""
        if self._compiled_graph is None:
            graph = self.build_graph()
            self._compiled_graph = graph.compile(**kwargs)
        return self._compiled_graph
    
    @abstractmethod
    async def invoke(
        self, 
        task: str, 
        parent_context: Optional[List[Any]] = None,
        **kwargs
    ) -> str:
        """
        Invoke the skill with a task description.
        Returns a summary string to be passed back to the supervisor.
        """
        pass
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Return the tool schema for invoking this skill.
        Used to generate the invoke_skill tool parameters.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": f"The task to delegate to the {self.name} skill"
                    }
                },
                "required": ["task"]
            }
        }


class SimpleSkill(Skill):
    """
    A simple skill implementation that uses a standard ReAct-style loop.
    Suitable for most use cases where you just need isolated tool execution.
    """
    
    def build_graph(self) -> StateGraph:
        from langgraph.prebuilt import create_react_agent
        from langchain.chat_models import init_chat_model
        
        # Use specified model or default
        model_name = self.config.model or "anthropic:claude-sonnet-4-20250514"
        model = init_chat_model(model_name)
        
        # Build a simple ReAct agent graph
        # Note: We're NOT using middleware here - direct graph construction
        return create_react_agent(
            model=model,
            tools=self.config.tools,
            state_schema=self.config.state_schema,
        )
    
    async def invoke(
        self,
        task: str,
        parent_context: Optional[List[Any]] = None,
        **kwargs
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        
        graph = self.compile()
        
        # Build initial messages
        messages = []
        
        # Add system prompt
        if self.config.system_prompt:
            messages.append(SystemMessage(content=self.config.system_prompt))
        
        # Optionally include parent context (summarized)
        if self.config.inherit_parent_context and parent_context:
            context_summary = self._summarize_context(parent_context)
            messages.append(SystemMessage(
                content=f"Context from parent agent:\n{context_summary}"
            ))
        
        # Add the task
        messages.append(HumanMessage(content=task))
        
        # Execute
        result = await graph.ainvoke(
            {"messages": messages},
            {"recursion_limit": self.config.max_iterations}
        )
        
        # Extract final response
        final_messages = result.get("messages", [])
        if final_messages:
            return final_messages[-1].content
        return "Skill completed with no output."
    
    def _summarize_context(self, context: List[Any], max_tokens: int = 2000) -> str:
        """Summarize parent context to fit within token limit."""
        # Simple truncation - could be enhanced with actual summarization
        context_str = "\n".join(str(m) for m in context[-5:])  # Last 5 messages
        if len(context_str) > max_tokens * 4:  # Rough char to token estimate
            context_str = context_str[-(max_tokens * 4):]
        return context_str
```

#### Task 2.2: Create `skills/registry.py`

```python
"""
Skill Registry - manages skill registration, discovery, and configuration.
Supports loading skills from:
- Python classes
- YAML/JSON configuration files
- Environment variables
"""
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pathlib import Path
import yaml
import json

from .base import Skill, SkillConfig, SimpleSkill


class SkillRegistry:
    """
    Central registry for skills. Allows PMs to configure agent behavior
    via config files rather than code changes.
    """
    
    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._skill_classes: Dict[str, Type[Skill]] = {}
        self._default_skill_class: Type[Skill] = SimpleSkill
    
    def register_skill_class(self, name: str, skill_class: Type[Skill]):
        """Register a custom skill class for use in configs."""
        self._skill_classes[name] = skill_class
    
    def register_skill(self, skill: Skill):
        """Register an instantiated skill."""
        self._skills[skill.name] = skill
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)
    
    def list_skills(self) -> List[str]:
        """List all registered skill names."""
        return list(self._skills.keys())
    
    def get_all_skills(self) -> List[Skill]:
        """Get all registered skills."""
        return list(self._skills.values())
    
    def load_from_config(
        self,
        config: Union[Dict, str, Path],
        tools_registry: Optional[Dict[str, Callable]] = None
    ):
        """
        Load skills from a configuration dict, YAML file, or JSON file.
        
        Config format:
        ```yaml
        skills:
          - name: research
            description: "Research and gather information"
            system_prompt: |
              You are an expert researcher. Gather comprehensive information
              on the topic and return a structured summary.
            tools:
              - web_search
              - fetch_url
            model: "anthropic:claude-sonnet-4-20250514"
            settings:
              max_sources: 5
        ```
        """
        if isinstance(config, (str, Path)):
            path = Path(config)
            with open(path) as f:
                if path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
        
        tools_registry = tools_registry or {}
        
        for skill_config in config.get('skills', []):
            # Resolve tool references
            tool_names = skill_config.pop('tools', [])
            resolved_tools = []
            for tool_name in tool_names:
                if tool_name in tools_registry:
                    resolved_tools.append(tools_registry[tool_name])
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
            
            # Determine skill class
            skill_class_name = skill_config.pop('skill_class', None)
            if skill_class_name:
                skill_class = self._skill_classes.get(
                    skill_class_name, 
                    self._default_skill_class
                )
            else:
                skill_class = self._default_skill_class
            
            # Create config and skill
            config_obj = SkillConfig(tools=resolved_tools, **skill_config)
            skill = skill_class(config_obj)
            self.register_skill(skill)
    
    def create_invoke_skill_tool(self) -> Callable:
        """
        Create the master 'invoke_skill' tool that routes to registered skills.
        This is the tool the supervisor agent uses to delegate to skills.
        """
        from langchain_core.tools import tool
        
        skill_descriptions = "\n".join([
            f"- **{s.name}**: {s.description}"
            for s in self._skills.values()
        ])
        
        @tool
        async def invoke_skill(skill_name: str, task: str) -> str:
            """
            Invoke a specialized skill to handle a specific task.
            
            Available skills:
            {skill_descriptions}
            
            Args:
                skill_name: Name of the skill to invoke
                task: Detailed description of the task to accomplish
            
            Returns:
                Result summary from the skill execution
            """
            skill = self._skills.get(skill_name)
            if not skill:
                return f"Error: Unknown skill '{skill_name}'. Available: {list(self._skills.keys())}"
            
            try:
                result = await skill.invoke(task)
                return result
            except Exception as e:
                return f"Error executing skill '{skill_name}': {str(e)}"
        
        # Update docstring with actual skill list
        invoke_skill.__doc__ = invoke_skill.__doc__.format(
            skill_descriptions=skill_descriptions
        )
        
        return invoke_skill


# Global registry instance
_global_registry = SkillRegistry()

def get_global_registry() -> SkillRegistry:
    return _global_registry
```

#### Task 2.3: Create `skills/executor.py`

```python
"""
Skill Executor - handles isolated execution of skills with proper
context management and state isolation.
"""
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
import asyncio

from .base import Skill, SkillConfig


class SkillExecutor:
    """
    Manages the execution of skills with:
    - Context isolation (separate message history per skill invocation)
    - Parallel execution support (multiple skills can run concurrently)
    - Result aggregation and summarization
    """
    
    def __init__(
        self,
        max_concurrent_skills: int = 3,
        timeout_seconds: int = 300
    ):
        self.max_concurrent_skills = max_concurrent_skills
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent_skills)
    
    async def execute_skill(
        self,
        skill: Skill,
        task: str,
        parent_context: Optional[List[BaseMessage]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a single skill with context isolation.
        
        Returns:
            {
                "skill_name": str,
                "task": str,
                "result": str,
                "success": bool,
                "error": Optional[str],
                "metadata": Dict
            }
        """
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    skill.invoke(task, parent_context),
                    timeout=self.timeout_seconds
                )
                return {
                    "skill_name": skill.name,
                    "task": task,
                    "result": result,
                    "success": True,
                    "error": None,
                    "metadata": metadata or {}
                }
            except asyncio.TimeoutError:
                return {
                    "skill_name": skill.name,
                    "task": task,
                    "result": None,
                    "success": False,
                    "error": f"Skill execution timed out after {self.timeout_seconds}s",
                    "metadata": metadata or {}
                }
            except Exception as e:
                return {
                    "skill_name": skill.name,
                    "task": task,
                    "result": None,
                    "success": False,
                    "error": str(e),
                    "metadata": metadata or {}
                }
    
    async def execute_parallel(
        self,
        skill_tasks: List[Dict[str, Any]],
        parent_context: Optional[List[BaseMessage]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple skill invocations in parallel.
        
        Args:
            skill_tasks: List of {"skill": Skill, "task": str, "metadata": Dict}
        
        Returns:
            List of execution results
        """
        coroutines = [
            self.execute_skill(
                st["skill"],
                st["task"],
                parent_context,
                st.get("metadata")
            )
            for st in skill_tasks
        ]
        return await asyncio.gather(*coroutines)


class IsolatedSkillNode:
    """
    A LangGraph node that executes a skill in isolation.
    Can be used to build custom graphs with skill invocations.
    """
    
    def __init__(self, skill: Skill, executor: Optional[SkillExecutor] = None):
        self.skill = skill
        self.executor = executor or SkillExecutor()
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Node function for LangGraph.
        Extracts task from state, executes skill, updates state with result.
        """
        # Extract the task - assumes last message contains the skill invocation
        messages = state.get("messages", [])
        task = self._extract_task_from_messages(messages)
        
        if not task:
            return {
                "messages": messages + [
                    AIMessage(content="Error: No task found for skill invocation")
                ]
            }
        
        # Execute in isolation
        result = await self.executor.execute_skill(
            self.skill,
            task,
            parent_context=messages if self.skill.config.inherit_parent_context else None
        )
        
        # Format result as message
        if result["success"]:
            response = f"[{self.skill.name} skill completed]\n\n{result['result']}"
        else:
            response = f"[{self.skill.name} skill failed]\n\nError: {result['error']}"
        
        return {
            "messages": messages + [AIMessage(content=response)]
        }
    
    def _extract_task_from_messages(self, messages: List[BaseMessage]) -> Optional[str]:
        """Extract the task description from messages."""
        for msg in reversed(messages):
            if hasattr(msg, 'tool_calls'):
                for tc in msg.tool_calls:
                    if tc.get('name') == 'invoke_skill':
                        return tc.get('args', {}).get('task')
            if isinstance(msg, HumanMessage):
                return msg.content
        return None
```

### Phase 3: Create Supervisor Agent Without Middleware

#### Task 3.1: Create `graph_no_middleware.py`

```python
"""
Supervisor agent implementation WITHOUT LangChain 1.0 middleware.
Uses direct LangGraph patterns for backward compatibility.
"""
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Type, Union
from dataclasses import dataclass, field

from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from .skills import SkillRegistry, SkillExecutor, get_global_registry


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class TodoItem(BaseModel):
    """A single todo item for task planning."""
    id: str
    task: str
    status: Literal["pending", "in_progress", "completed", "skipped"] = "pending"
    notes: Optional[str] = None


class AgentState(BaseModel):
    """
    Complete agent state - replaces middleware-injected state schemas.
    """
    # Core message history
    messages: List[BaseMessage] = Field(default_factory=list)
    
    # Planning state (replaces TodoListMiddleware)
    todos: List[TodoItem] = Field(default_factory=list)
    
    # Filesystem state (replaces FilesystemMiddleware)
    files: Dict[str, str] = Field(default_factory=dict)  # path -> content
    
    # Skill execution tracking
    active_skill: Optional[str] = None
    skill_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Context management
    context_summary: Optional[str] = None
    total_tokens_used: int = 0
    
    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# BUILT-IN TOOLS (replaces tool injection from middleware)
# ============================================================================

def create_planning_tools() -> List[BaseTool]:
    """Create todo list management tools."""
    from langchain_core.tools import tool
    
    @tool
    def write_todos(todos: List[Dict[str, str]]) -> str:
        """
        Create or update the todo list for tracking task progress.
        
        Args:
            todos: List of todo items with 'task' and optionally 'status' keys
        
        Returns:
            Confirmation message
        """
        # Note: Actual state update happens in the graph node
        return f"Created {len(todos)} todo items. Use read_todos to view them."
    
    @tool
    def read_todos() -> str:
        """Read the current todo list state."""
        return "Todo list will be displayed by the system."
    
    @tool
    def update_todo(todo_id: str, status: str, notes: Optional[str] = None) -> str:
        """
        Update a todo item's status.
        
        Args:
            todo_id: ID of the todo to update
            status: New status (pending, in_progress, completed, skipped)
            notes: Optional notes about the update
        """
        return f"Updated todo {todo_id} to {status}"
    
    return [write_todos, read_todos, update_todo]


def create_filesystem_tools() -> List[BaseTool]:
    """Create filesystem operation tools."""
    from langchain_core.tools import tool
    
    @tool
    def ls(path: str = "/") -> str:
        """List files in a directory. Path must start with /."""
        return "Directory listing will be provided by the system."
    
    @tool
    def read_file(path: str, offset: int = 0, limit: int = 500) -> str:
        """
        Read content from a file with optional pagination.
        
        Args:
            path: Absolute path to the file (must start with /)
            offset: Line number to start reading from (0-indexed)
            limit: Maximum number of lines to read
        """
        return "File content will be provided by the system."
    
    @tool
    def write_file(path: str, content: str) -> str:
        """
        Create a new file or overwrite an existing file.
        
        Args:
            path: Absolute path for the file (must start with /)
            content: Content to write to the file
        """
        return f"File written to {path}"
    
    @tool
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """
        Perform exact string replacement in a file.
        
        Args:
            path: Path to the file to edit
            old_text: Exact text to find and replace
            new_text: Text to replace it with
        """
        return f"Edited {path}"
    
    @tool
    def glob(pattern: str) -> str:
        """Find files matching a glob pattern (e.g., '**/*.py')."""
        return "Matching files will be listed by the system."
    
    @tool
    def grep(pattern: str, path: str = "/") -> str:
        """
        Search for text patterns within files.
        
        Args:
            pattern: Text or regex pattern to search for
            path: Directory to search in
        """
        return "Search results will be provided by the system."
    
    return [ls, read_file, write_file, edit_file, glob, grep]


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

@dataclass
class SupervisorConfig:
    """Configuration for the supervisor agent."""
    model: str = "anthropic:claude-sonnet-4-20250514"
    system_prompt: Optional[str] = None
    tools: List[Union[BaseTool, Callable]] = field(default_factory=list)
    skill_registry: Optional[SkillRegistry] = None
    
    # Feature flags (replaces middleware enable/disable)
    enable_planning: bool = True
    enable_filesystem: bool = True
    enable_skills: bool = True
    enable_summarization: bool = True
    
    # Context management
    max_context_tokens: int = 170000
    summarization_threshold: int = 150000
    
    # Execution settings
    max_iterations: int = 50


def create_supervisor_agent(config: SupervisorConfig) -> CompiledStateGraph:
    """
    Create a supervisor agent WITHOUT using LangChain middleware.
    
    This is the main entry point - replaces `create_deep_agent()`.
    """
    
    # Initialize model
    model = init_chat_model(config.model)
    
    # Collect all tools
    all_tools = list(config.tools)
    
    if config.enable_planning:
        all_tools.extend(create_planning_tools())
    
    if config.enable_filesystem:
        all_tools.extend(create_filesystem_tools())
    
    if config.enable_skills:
        registry = config.skill_registry or get_global_registry()
        if registry.list_skills():
            all_tools.append(registry.create_invoke_skill_tool())
    
    # Bind tools to model
    model_with_tools = model.bind_tools(all_tools)
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # =========== NODES ===========
    
    async def supervisor_node(state: AgentState) -> Dict[str, Any]:
        """Main supervisor reasoning node."""
        messages = state.messages
        
        # Inject system prompt
        system_messages = []
        if config.system_prompt:
            system_messages.append(SystemMessage(content=config.system_prompt))
        
        # Add built-in instructions (replaces middleware prompt injection)
        builtin_instructions = _build_system_instructions(config, state)
        if builtin_instructions:
            system_messages.append(SystemMessage(content=builtin_instructions))
        
        # Add context summary if available
        if state.context_summary:
            system_messages.append(SystemMessage(
                content=f"[Context Summary]\n{state.context_summary}"
            ))
        
        # Call model
        full_messages = system_messages + messages
        response = await model_with_tools.ainvoke(full_messages)
        
        return {"messages": [response]}
    
    async def tool_executor_node(state: AgentState) -> Dict[str, Any]:
        """Execute tools, with special handling for state-modifying tools."""
        messages = state.messages
        last_message = messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {}
        
        results = []
        new_state_updates = {}
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']
            
            # Handle state-modifying tools specially
            if tool_name == 'write_todos':
                todos = [
                    TodoItem(id=f"todo_{i}", task=t['task'], status=t.get('status', 'pending'))
                    for i, t in enumerate(tool_args.get('todos', []))
                ]
                new_state_updates['todos'] = todos
                result = f"Created {len(todos)} todo items."
            
            elif tool_name == 'read_todos':
                todo_list = "\n".join([
                    f"[{t.status}] {t.id}: {t.task}"
                    for t in state.todos
                ]) or "No todos yet."
                result = todo_list
            
            elif tool_name == 'write_file':
                path = tool_args['path']
                content = tool_args['content']
                new_files = dict(state.files)
                new_files[path] = content
                new_state_updates['files'] = new_files
                result = f"Written {len(content)} chars to {path}"
            
            elif tool_name == 'read_file':
                path = tool_args['path']
                content = state.files.get(path, f"Error: File not found: {path}")
                result = content
            
            elif tool_name == 'ls':
                path = tool_args.get('path', '/')
                files_in_path = [p for p in state.files.keys() if p.startswith(path)]
                result = "\n".join(files_in_path) or "No files found."
            
            elif tool_name == 'invoke_skill':
                # Skill invocation - handled by skill executor
                skill_name = tool_args['skill_name']
                task = tool_args['task']
                registry = config.skill_registry or get_global_registry()
                skill = registry.get_skill(skill_name)
                
                if skill:
                    executor = SkillExecutor()
                    exec_result = await executor.execute_skill(skill, task, state.messages)
                    result = exec_result['result'] if exec_result['success'] else f"Error: {exec_result['error']}"
                else:
                    result = f"Unknown skill: {skill_name}"
            
            else:
                # Regular tool - use ToolNode
                tool_node = ToolNode(all_tools)
                tool_result = await tool_node.ainvoke(state)
                # Extract result from tool node response
                result = str(tool_result.get('messages', ['No result'])[-1].content if tool_result.get('messages') else 'No result')
            
            results.append(ToolMessage(content=result, tool_call_id=tool_id))
        
        return {"messages": results, **new_state_updates}
    
    async def summarization_node(state: AgentState) -> Dict[str, Any]:
        """Summarize context if it exceeds threshold."""
        if not config.enable_summarization:
            return {}
        
        # Estimate token count (rough: 4 chars per token)
        total_chars = sum(len(str(m)) for m in state.messages)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens < config.summarization_threshold:
            return {}
        
        # Summarize older messages, keep last 6
        messages_to_summarize = state.messages[:-6]
        recent_messages = state.messages[-6:]
        
        if not messages_to_summarize:
            return {}
        
        # Use model to generate summary
        summary_prompt = f"""Summarize the following conversation context concisely:

{chr(10).join(str(m) for m in messages_to_summarize)}

Provide a brief summary capturing key points, decisions, and relevant context."""
        
        summary_response = await model.ainvoke([HumanMessage(content=summary_prompt)])
        
        return {
            "messages": recent_messages,
            "context_summary": summary_response.content
        }
    
    # =========== EDGES ===========
    
    def should_continue(state: AgentState) -> Literal["tools", "summarize", "end"]:
        """Determine next step after supervisor node."""
        messages = state.messages
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # Check for tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Check if summarization needed
        total_chars = sum(len(str(m)) for m in messages)
        if total_chars // 4 > config.summarization_threshold:
            return "summarize"
        
        return "end"
    
    # =========== BUILD GRAPH ===========
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tools", tool_executor_node)
    workflow.add_node("summarize", summarization_node)
    
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "tools": "tools",
            "summarize": "summarize",
            "end": END
        }
    )
    
    workflow.add_edge("tools", "supervisor")
    workflow.add_edge("summarize", "supervisor")
    
    return workflow.compile()


def _build_system_instructions(config: SupervisorConfig, state: AgentState) -> str:
    """Build system instructions (replaces middleware prompt injection)."""
    instructions = []
    
    if config.enable_planning:
        instructions.append("""
## Task Planning
You have access to todo list tools for managing complex tasks:
- Use `write_todos` to create a structured task list before starting complex work
- Use `read_todos` to check current progress
- Use `update_todo` to mark items complete or add notes
- Break down complex tasks into manageable steps
- Don't use todos for simple, single-step tasks
""")
    
    if config.enable_filesystem:
        instructions.append("""
## File Operations
You have access to a virtual filesystem:
- All paths must start with `/`
- Use `ls` to list directory contents
- Use `read_file` with offset/limit for large files
- Use `write_file` to create or overwrite files
- Use `edit_file` for precise text replacements
- Use `glob` and `grep` for searching
- Large tool outputs (>20k tokens) will be auto-saved to files
""")
    
    if config.enable_skills and config.skill_registry:
        skills = config.skill_registry.get_all_skills()
        if skills:
            skill_list = "\n".join([f"- **{s.name}**: {s.description}" for s in skills])
            instructions.append(f"""
## Skill Delegation
You can delegate specialized tasks to isolated skill agents:
{skill_list}

Use `invoke_skill` to delegate tasks that:
- Require focused, context-heavy work
- Benefit from isolation (prevents context bloat)
- Have clear, self-contained objectives

Don't delegate:
- Trivial tasks (a few tool calls)
- Tasks where you need to see intermediate steps
- Tasks that require the full conversation context
""")
    
    return "\n".join(instructions)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_skill_agent(
    model: str = "anthropic:claude-sonnet-4-20250514",
    system_prompt: Optional[str] = None,
    tools: Optional[List[Union[BaseTool, Callable]]] = None,
    skills: Optional[List[Dict[str, Any]]] = None,
    skills_config_path: Optional[str] = None,
    **kwargs
) -> CompiledStateGraph:
    """
    Convenience function to create a skill-enabled agent.
    
    This is the main entry point - use instead of `create_deep_agent()`.
    
    Args:
        model: Model identifier (e.g., "anthropic:claude-sonnet-4-20250514")
        system_prompt: Custom system prompt (appended to built-in instructions)
        tools: Additional tools to provide
        skills: List of skill configurations (dicts with name, description, etc.)
        skills_config_path: Path to YAML/JSON skills configuration file
        **kwargs: Additional SupervisorConfig parameters
    
    Returns:
        Compiled LangGraph StateGraph
    
    Example:
        ```python
        agent = create_skill_agent(
            model="anthropic:claude-sonnet-4-20250514",
            system_prompt="You are a financial analyst.",
            tools=[my_data_tool],
            skills=[
                {
                    "name": "research",
                    "description": "Deep research on any topic",
                    "system_prompt": "You are an expert researcher...",
                    "tools": [web_search, fetch_url]
                }
            ]
        )
        
        result = await agent.ainvoke({"messages": [HumanMessage(content="...")]})
        ```
    """
    # Setup skill registry
    registry = SkillRegistry()
    tools_dict = {t.name if hasattr(t, 'name') else t.__name__: t for t in (tools or [])}
    
    if skills:
        registry.load_from_config({"skills": skills}, tools_dict)
    
    if skills_config_path:
        registry.load_from_config(skills_config_path, tools_dict)
    
    # Build config
    config = SupervisorConfig(
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],
        skill_registry=registry if registry.list_skills() else None,
        **kwargs
    )
    
    return create_supervisor_agent(config)
```

### Phase 4: Add Example Skills for Trading

#### Task 4.1: Create `skills/builtin/trading.py`

```python
"""
Built-in skills for trading/finance applications.
These serve as templates for custom skill development.
"""
from typing import Any, Dict, List, Optional
from ..base import Skill, SkillConfig, SimpleSkill


class MarketResearchSkill(SimpleSkill):
    """
    Skill for researching market conditions, news, and sentiment.
    Designed for "Alpha Cortex" style trading agent infrastructure.
    """
    
    DEFAULT_PROMPT = """You are a market research specialist. Your task is to:

1. Gather relevant market information from available data sources
2. Analyze news and events that may impact the asset/market
3. Identify key sentiment indicators
4. Summarize findings in a structured format

Output Format:
- **Summary**: 2-3 sentence overview
- **Key Findings**: Bullet points of important information
- **Sentiment**: Bullish/Neutral/Bearish with confidence level
- **Risk Factors**: Notable risks or uncertainties
- **Data Sources**: List sources used

Be thorough but concise. Focus on actionable intelligence."""


class QuantAnalysisSkill(SimpleSkill):
    """
    Skill for quantitative analysis of market data.
    """
    
    DEFAULT_PROMPT = """You are a quantitative analyst. Your task is to:

1. Analyze provided market data using statistical methods
2. Calculate relevant metrics (volatility, correlations, etc.)
3. Identify patterns or anomalies
4. Generate actionable insights

You have access to computational tools. Use them to:
- Perform calculations
- Generate visualizations
- Run statistical tests

Output structured analysis with clear methodology and conclusions."""


class TradeExecutionSkill(SimpleSkill):
    """
    Skill for trade execution planning and validation.
    """
    
    DEFAULT_PROMPT = """You are a trade execution specialist. Your task is to:

1. Validate trade parameters against risk limits
2. Calculate optimal execution strategy
3. Estimate market impact
4. Generate execution plan

IMPORTANT: You do NOT execute trades directly. You provide:
- Execution recommendations
- Risk assessment
- Order sizing suggestions
- Timing considerations

Always include:
- Position size (with Kelly criterion consideration if applicable)
- Entry/exit levels
- Stop loss levels
- Risk/reward ratio"""


def create_trading_skills(tools_registry: Dict[str, Any]) -> List[Skill]:
    """
    Factory function to create standard trading skills.
    
    Args:
        tools_registry: Dict mapping tool names to tool functions
            Expected tools: web_search, market_data, execute_code
    
    Returns:
        List of configured trading skills
    """
    skills = []
    
    # Market Research Skill
    research_tools = []
    for tool_name in ['web_search', 'fetch_url', 'news_search']:
        if tool_name in tools_registry:
            research_tools.append(tools_registry[tool_name])
    
    if research_tools:
        skills.append(MarketResearchSkill(SkillConfig(
            name="market_research",
            description="Research market conditions, news, and sentiment for any asset or market",
            system_prompt=MarketResearchSkill.DEFAULT_PROMPT,
            tools=research_tools,
            settings={"max_sources": 10}
        )))
    
    # Quant Analysis Skill
    quant_tools = []
    for tool_name in ['execute_code', 'market_data', 'calculate']:
        if tool_name in tools_registry:
            quant_tools.append(tools_registry[tool_name])
    
    if quant_tools:
        skills.append(QuantAnalysisSkill(SkillConfig(
            name="quant_analysis",
            description="Perform quantitative analysis on market data including statistics, patterns, and metrics",
            system_prompt=QuantAnalysisSkill.DEFAULT_PROMPT,
            tools=quant_tools,
            settings={"max_compute_time": 60}
        )))
    
    # Trade Execution Skill
    exec_tools = []
    for tool_name in ['risk_check', 'position_calculator', 'order_validator']:
        if tool_name in tools_registry:
            exec_tools.append(tools_registry[tool_name])
    
    if exec_tools:
        skills.append(TradeExecutionSkill(SkillConfig(
            name="trade_execution",
            description="Plan and validate trade execution with risk management",
            system_prompt=TradeExecutionSkill.DEFAULT_PROMPT,
            tools=exec_tools,
            settings={"max_position_pct": 0.05}
        )))
    
    return skills
```

### Phase 5: Create Configuration System

#### Task 5.1: Create `config/skills.yaml` (Example)

```yaml
# Example skills configuration for Alpha Cortex trading agent
# PMs can modify this file to customize agent behavior without code changes

skills:
  - name: market_research
    description: "Research market conditions, news, and sentiment for any asset"
    system_prompt: |
      You are a market research specialist focused on financial markets.
      
      Your task:
      1. Gather market information from available sources
      2. Analyze relevant news and events
      3. Assess market sentiment
      
      Output a structured research report with:
      - Executive Summary (2-3 sentences)
      - Key Findings (bullet points)
      - Sentiment Assessment (Bullish/Neutral/Bearish + confidence)
      - Risk Factors
      - Sources Used
    tools:
      - web_search
      - news_search
    model: "anthropic:claude-sonnet-4-20250514"
    settings:
      max_sources: 10
      include_social_sentiment: true

  - name: quant_analysis
    description: "Analyze market data quantitatively"
    system_prompt: |
      You are a quantitative analyst. Analyze market data using statistical methods.
      
      Available analyses:
      - Volatility metrics (historical vol, implied vol)
      - Correlation analysis
      - Regime detection (HMM-based)
      - Technical indicators
      
      Always show your methodology and calculations.
    tools:
      - execute_code
      - market_data_api
    model: "anthropic:claude-sonnet-4-20250514"
    settings:
      max_lookback_days: 252
      
  - name: trade_planning
    description: "Create trade execution plans with risk management"
    system_prompt: |
      You are a trade execution planner. Given a trade idea, create a detailed plan.
      
      Required outputs:
      1. Position sizing (using Kelly criterion where applicable)
      2. Entry strategy (limit orders, TWAP, etc.)
      3. Stop loss levels
      4. Take profit targets
      5. Risk/reward analysis
      
      You do NOT execute trades. You provide recommendations only.
    tools:
      - risk_calculator
      - position_sizer
    settings:
      max_risk_per_trade: 0.02
      kelly_fraction: 0.5

# Supervisor agent configuration
supervisor:
  model: "anthropic:claude-sonnet-4-20250514"
  system_prompt: |
    You are Alpha Cortex, an AI trading assistant for portfolio managers.
    
    Your capabilities:
    - Market research and analysis
    - Quantitative data analysis
    - Trade planning and risk management
    
    You have specialized skills you can delegate to:
    - Use `market_research` for news, sentiment, and market overview
    - Use `quant_analysis` for data-driven analysis
    - Use `trade_planning` for execution recommendations
    
    Important guidelines:
    - Always consider risk management
    - Provide clear reasoning for recommendations
    - Acknowledge uncertainty when appropriate
    - Never execute trades without explicit confirmation
    
  enable_planning: true
  enable_filesystem: true
  max_iterations: 30
```

### Phase 6: Testing & Integration

#### Task 6.1: Create `tests/test_skills.py`

```python
"""
Tests for the skills architecture.
"""
import pytest
import asyncio
from deepagents.skills import (
    Skill, 
    SkillConfig, 
    SimpleSkill,
    SkillRegistry,
    SkillExecutor,
    get_global_registry
)
from deepagents.graph_no_middleware import create_skill_agent


@pytest.fixture
def sample_tools():
    """Create sample tools for testing."""
    from langchain_core.tools import tool
    
    @tool
    def mock_search(query: str) -> str:
        """Mock search tool."""
        return f"Search results for: {query}"
    
    @tool
    def mock_calculate(expression: str) -> str:
        """Mock calculator."""
        return f"Result: {eval(expression)}"
    
    return {"mock_search": mock_search, "mock_calculate": mock_calculate}


@pytest.fixture
def skill_registry(sample_tools):
    """Create a skill registry with test skills."""
    registry = SkillRegistry()
    
    registry.load_from_config({
        "skills": [
            {
                "name": "test_research",
                "description": "Test research skill",
                "system_prompt": "You are a test researcher.",
                "tools": ["mock_search"],
            },
            {
                "name": "test_analysis",
                "description": "Test analysis skill",
                "system_prompt": "You are a test analyst.",
                "tools": ["mock_calculate"],
            }
        ]
    }, sample_tools)
    
    return registry


class TestSkillRegistry:
    def test_register_and_retrieve(self, skill_registry):
        assert "test_research" in skill_registry.list_skills()
        assert "test_analysis" in skill_registry.list_skills()
        
        skill = skill_registry.get_skill("test_research")
        assert skill is not None
        assert skill.name == "test_research"
    
    def test_create_invoke_tool(self, skill_registry):
        tool = skill_registry.create_invoke_skill_tool()
        assert tool is not None
        assert "test_research" in tool.__doc__
        assert "test_analysis" in tool.__doc__


class TestSkillExecution:
    @pytest.mark.asyncio
    async def test_simple_skill_invoke(self, sample_tools):
        config = SkillConfig(
            name="test",
            description="Test skill",
            system_prompt="Echo back the task.",
            tools=[sample_tools["mock_search"]],
        )
        skill = SimpleSkill(config)
        
        # Note: This will fail without actual model access
        # In real tests, mock the model
        # result = await skill.invoke("Test task")
        # assert result is not None


class TestSupervisorAgent:
    def test_create_agent(self, skill_registry, sample_tools):
        agent = create_skill_agent(
            model="anthropic:claude-sonnet-4-20250514",
            tools=list(sample_tools.values()),
            skills=[
                {
                    "name": "custom_skill",
                    "description": "Custom test skill",
                    "system_prompt": "You are a custom skill.",
                    "tools": ["mock_search"]
                }
            ]
        )
        
        assert agent is not None
        # Verify graph structure
        assert "supervisor" in agent.nodes
        assert "tools" in agent.nodes
```

### Phase 7: Documentation

#### Task 7.1: Create `docs/SKILLS.md`

```markdown
# Skills Architecture for DeepAgents

## Overview

The skills architecture provides a **middleware-free** approach to creating modular, 
configurable agent capabilities. Skills are isolated agent modules that can be 
invoked by a supervisor agent to handle specific tasks.

## Key Concepts

### Skills vs Middleware

| Aspect | Middleware (LangChain 1.0) | Skills (This Implementation) |
|--------|---------------------------|------------------------------|
| Execution | Inline with agent loop | Isolated context |
| Configuration | Code-based | Config/prompt-based |
| State | Shared with agent | Isolated, returns summary |
| Dependency | Requires LangChain 1.0 | Works with older versions |
| Customization | Requires Python code | YAML/JSON configs |

### When to Use Skills

✅ **Use skills for:**
- Tasks requiring deep, focused reasoning
- Context-heavy operations that would bloat main agent
- Specialized capabilities (research, analysis, execution)
- Tasks where intermediate steps don't need to be visible

❌ **Don't use skills for:**
- Simple, single-step operations
- Tasks requiring full conversation context
- Operations where you need to see/control intermediate steps

## Quick Start

```python
from deepagents import create_skill_agent

# Define skills inline
agent = create_skill_agent(
    model="anthropic:claude-sonnet-4-20250514",
    skills=[
        {
            "name": "research",
            "description": "Deep research on any topic",
            "system_prompt": "You are an expert researcher...",
            "tools": [web_search, fetch_url]
        }
    ]
)

# Or load from config file
agent = create_skill_agent(
    model="anthropic:claude-sonnet-4-20250514",
    skills_config_path="config/skills.yaml"
)

# Invoke
result = await agent.ainvoke({
    "messages": [HumanMessage(content="Research the latest AI developments")]
})
```

## Configuration Reference

### Skill Configuration (YAML)

```yaml
skills:
  - name: string          # Unique identifier
    description: string   # Shown to supervisor for routing
    system_prompt: string # Instructions for the skill agent
    tools: [string]       # Tool names from registry
    model: string         # Optional: override default model
    max_iterations: int   # Optional: limit tool calls (default: 25)
    settings:             # Optional: skill-specific settings
      key: value
```

### Supervisor Configuration

```yaml
supervisor:
  model: string
  system_prompt: string
  enable_planning: bool    # Todo list tools
  enable_filesystem: bool  # File operation tools
  enable_skills: bool      # Skill delegation
  max_iterations: int
```

## Custom Skills

Create custom skill classes for advanced use cases:

```python
from deepagents.skills import Skill, SkillConfig

class MyCustomSkill(Skill):
    def build_graph(self) -> StateGraph:
        # Custom LangGraph construction
        workflow = StateGraph(MyState)
        # ... add nodes and edges
        return workflow
    
    async def invoke(self, task: str, parent_context=None) -> str:
        # Custom execution logic
        result = await self.compile().ainvoke(...)
        return self.format_result(result)
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     create_skill_agent()                        │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              SupervisorConfig                           │   │
│  │  - model, system_prompt, tools                          │   │
│  │  - skill_registry (SkillRegistry)                       │   │
│  │  - enable_planning, enable_filesystem, etc.             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           create_supervisor_agent()                     │   │
│  │                                                         │   │
│  │   ┌──────────┐     ┌──────────┐     ┌──────────┐       │   │
│  │   │supervisor│────▶│  tools   │────▶│summarize │       │   │
│  │   │   node   │◀────│   node   │     │   node   │       │   │
│  │   └──────────┘     └──────────┘     └──────────┘       │   │
│  │        │                │                              │   │
│  │        │          ┌─────┴─────┐                        │   │
│  │        │          │invoke_skill                        │   │
│  │        │          └─────┬─────┘                        │   │
│  │        │                │                              │   │
│  │        │    ┌───────────┴───────────┐                  │   │
│  │        │    │    SkillExecutor      │                  │   │
│  │        │    │  (isolated context)   │                  │   │
│  │        │    └───────────────────────┘                  │   │
│  └────────┼────────────────────────────────────────────────┘   │
│           │                                                    │
│           ▼                                                    │
│   CompiledStateGraph                                           │
└─────────────────────────────────────────────────────────────────┘
```
```

## Summary: Key Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `skills/__init__.py` | Create | Package exports |
| `skills/base.py` | Create | Skill base class |
| `skills/registry.py` | Create | Skill registration |
| `skills/executor.py` | Create | Isolated execution |
| `skills/builtin/trading.py` | Create | Trading skill templates |
| `graph_no_middleware.py` | Create | Middleware-free supervisor |
| `config/skills.yaml` | Create | Example configuration |
| `tests/test_skills.py` | Create | Test suite |
| `docs/SKILLS.md` | Create | Documentation |
| `__init__.py` | Modify | Export new functions |

## Commands for Claude Code

```bash
# Start implementation
cd deepagents
mkdir -p libs/deepagents/deepagents/skills/builtin
mkdir -p libs/deepagents/deepagents/config
mkdir -p libs/deepagents/tests

# Create files in order (run each, verify, then next)
# 1. skills/base.py
# 2. skills/registry.py  
# 3. skills/executor.py
# 4. graph_no_middleware.py
# 5. skills/builtin/trading.py
# 6. config/skills.yaml
# 7. tests/test_skills.py
# 8. Update __init__.py exports

# Test
cd libs/deepagents
pytest tests/test_skills.py -v

# Verify imports work
python -c "from deepagents import create_skill_agent; print('OK')"
```
