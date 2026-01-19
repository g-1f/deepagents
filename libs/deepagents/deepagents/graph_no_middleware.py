"""Middleware-free supervisor agent with skills architecture.

This module provides a `create_skill_agent` function that creates a supervisor agent
without relying on LangChain 1.0 middleware. It's designed to work with:
- langchain>=0.2.0
- langgraph>=0.1.0

The agent has access to:
- Planning tools (write_todos, read_todos)
- Filesystem tools (ls, read_file, write_file, edit_file, glob, grep)
- Skill invocation (invoke_skill)

Skills are isolated subagents with custom tools and system prompts.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from deepagents.skills import SkillConfig, SkillRegistry, SimpleSkill

logger = logging.getLogger(__name__)


# =============================================================================
# State Definitions
# =============================================================================


class TodoItem(TypedDict):
    """A todo item."""

    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"]
    created_at: str


class FileData(TypedDict):
    """File data structure."""

    content: list[str]
    created_at: str
    modified_at: str


def _file_data_reducer(
    left: dict[str, FileData] | None, right: dict[str, FileData | None]
) -> dict[str, FileData]:
    """Merge file updates with support for deletions."""
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def _todo_reducer(
    left: list[TodoItem] | None, right: list[TodoItem]
) -> list[TodoItem]:
    """Replace todos entirely."""
    return right


class AgentState(TypedDict):
    """Combined state for the skill agent.

    This single state class replaces the middleware state schemas.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation messages."""

    todos: NotRequired[Annotated[list[TodoItem], _todo_reducer]]
    """Todo list for planning."""

    files: NotRequired[Annotated[dict[str, FileData], _file_data_reducer]]
    """Virtual filesystem for file operations."""

    skill_results: NotRequired[dict[str, str]]
    """Results from skill invocations."""


# =============================================================================
# Built-in Tools
# =============================================================================


def _create_todo_tools() -> list[BaseTool]:
    """Create planning tools for todo management."""

    def write_todos(
        todos: list[dict[str, Any]],
    ) -> str:
        """Update the todo list.

        Args:
            todos: List of todo items with 'content' and 'status' fields.
                   Status can be 'pending', 'in_progress', or 'completed'.

        Returns:
            Confirmation message.
        """
        # This will be wrapped to update state
        return f"Updated {len(todos)} todos"

    def read_todos() -> str:
        """Read the current todo list.

        Returns:
            Formatted todo list.
        """
        # This will be wrapped to read from state
        return "No todos"

    return [
        StructuredTool.from_function(
            func=write_todos,
            name="write_todos",
            description="Update the todo list for task planning and tracking.",
        ),
        StructuredTool.from_function(
            func=read_todos,
            name="read_todos",
            description="Read the current todo list.",
        ),
    ]


def _create_filesystem_tools(root_dir: str | None = None) -> list[BaseTool]:
    """Create filesystem tools.

    Args:
        root_dir: Optional root directory for filesystem operations.
                  If None, uses a virtual in-memory filesystem.

    Returns:
        List of filesystem tools.
    """

    def ls(path: str) -> str:
        """List files in a directory.

        Args:
            path: Absolute path to list (must start with /).

        Returns:
            List of files and directories.
        """
        if root_dir:
            full_path = Path(root_dir) / path.lstrip("/")
            if not full_path.exists():
                return f"Error: Directory not found: {path}"
            try:
                items = list(full_path.iterdir())
                return "\n".join(
                    f"{'[DIR] ' if p.is_dir() else ''}{p.name}" for p in sorted(items)
                )
            except PermissionError:
                return f"Error: Permission denied: {path}"
        return "Virtual filesystem - use read_file to access files"

    def read_file(
        file_path: str,
        offset: int = 0,
        limit: int = 100,
    ) -> str:
        """Read a file.

        Args:
            file_path: Absolute path to the file.
            offset: Line number to start from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            File content with line numbers.
        """
        if root_dir:
            full_path = Path(root_dir) / file_path.lstrip("/")
            if not full_path.exists():
                return f"Error: File not found: {file_path}"
            try:
                lines = full_path.read_text().splitlines()
                selected = lines[offset : offset + limit]
                numbered = [
                    f"{i + offset + 1:6}\t{line}" for i, line in enumerate(selected)
                ]
                return "\n".join(numbered)
            except Exception as e:
                return f"Error reading file: {e}"
        return "Virtual filesystem - file not found"

    def write_file(file_path: str, content: str) -> str:
        """Write content to a new file.

        Args:
            file_path: Absolute path for the new file.
            content: Content to write.

        Returns:
            Success or error message.
        """
        if root_dir:
            full_path = Path(root_dir) / file_path.lstrip("/")
            try:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                return f"Successfully wrote to {file_path}"
            except Exception as e:
                return f"Error writing file: {e}"
        return "Virtual filesystem - write_file not supported in stateless mode"

    def edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Edit a file by replacing text.

        Args:
            file_path: Path to the file to edit.
            old_string: Text to find and replace.
            new_string: Replacement text.
            replace_all: If True, replace all occurrences.

        Returns:
            Success or error message.
        """
        if root_dir:
            full_path = Path(root_dir) / file_path.lstrip("/")
            if not full_path.exists():
                return f"Error: File not found: {file_path}"
            try:
                content = full_path.read_text()
                if old_string not in content:
                    return f"Error: '{old_string}' not found in file"

                if replace_all:
                    new_content = content.replace(old_string, new_string)
                else:
                    count = content.count(old_string)
                    if count > 1:
                        return f"Error: '{old_string}' found {count} times. Use replace_all=True or provide more context."
                    new_content = content.replace(old_string, new_string, 1)

                full_path.write_text(new_content)
                return f"Successfully edited {file_path}"
            except Exception as e:
                return f"Error editing file: {e}"
        return "Virtual filesystem - edit_file not supported in stateless mode"

    def glob_search(pattern: str, path: str = "/") -> str:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '**/*.py').
            path: Directory to search in.

        Returns:
            List of matching file paths.
        """
        if root_dir:
            search_path = Path(root_dir) / path.lstrip("/")
            try:
                matches = list(search_path.glob(pattern))
                if not matches:
                    return f"No files matching '{pattern}' in {path}"
                rel_paths = [
                    "/" + str(m.relative_to(root_dir)) for m in sorted(matches)[:100]
                ]
                return "\n".join(rel_paths)
            except Exception as e:
                return f"Error searching: {e}"
        return "Virtual filesystem - glob not supported in stateless mode"

    def grep(
        pattern: str,
        path: str = "/",
        glob_pattern: str | None = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for text pattern in files.

        Args:
            pattern: Text pattern to search for.
            path: Directory to search in.
            glob_pattern: Optional glob to filter files.
            output_mode: 'files_with_matches', 'content', or 'count'.

        Returns:
            Search results.
        """
        if root_dir:
            import re

            search_path = Path(root_dir) / path.lstrip("/")
            results = []

            files_to_search = (
                list(search_path.glob(glob_pattern))
                if glob_pattern
                else list(search_path.rglob("*"))
            )

            for fp in files_to_search:
                if not fp.is_file():
                    continue
                try:
                    content = fp.read_text()
                    if pattern in content:
                        rel_path = "/" + str(fp.relative_to(root_dir))
                        if output_mode == "files_with_matches":
                            results.append(rel_path)
                        elif output_mode == "count":
                            count = content.count(pattern)
                            results.append(f"{rel_path}: {count}")
                        else:  # content
                            lines = content.splitlines()
                            for i, line in enumerate(lines):
                                if pattern in line:
                                    results.append(f"{rel_path}:{i+1}:{line}")
                except Exception:
                    continue

            if not results:
                return f"No matches for '{pattern}'"
            return "\n".join(results[:100])
        return "Virtual filesystem - grep not supported in stateless mode"

    return [
        StructuredTool.from_function(func=ls, name="ls", description="List files in a directory."),
        StructuredTool.from_function(func=read_file, name="read_file", description="Read a file with line numbers."),
        StructuredTool.from_function(func=write_file, name="write_file", description="Write content to a new file."),
        StructuredTool.from_function(func=edit_file, name="edit_file", description="Edit a file by replacing text."),
        StructuredTool.from_function(func=glob_search, name="glob", description="Find files matching a glob pattern."),
        StructuredTool.from_function(func=grep, name="grep", description="Search for text pattern in files."),
    ]


# =============================================================================
# System Prompts
# =============================================================================

BASE_SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools for completing tasks.

## Planning Tools
You have access to todo list tools for task planning:
- `write_todos`: Update your todo list to track progress
- `read_todos`: Check your current todos

## Filesystem Tools
You have access to filesystem tools:
- `ls`: List files in a directory
- `read_file`: Read a file with line numbers
- `write_file`: Create a new file
- `edit_file`: Edit an existing file
- `glob`: Find files matching a pattern
- `grep`: Search for text in files

## Skills
{skills_section}

## Guidelines
1. Break complex tasks into smaller steps using todos
2. Use appropriate tools to gather information before taking action
3. When a task matches a skill's expertise, delegate to that skill
4. Provide clear, concise responses
"""

SKILLS_SECTION_TEMPLATE = """You have access to specialized skills via the `invoke_skill` tool:

{skills_list}

When to use skills:
- When a task matches a skill's domain expertise
- When you need specialized knowledge or workflows
- When isolating a complex subtask would be beneficial

Usage:
```
invoke_skill(skill_name="skill-name", task="detailed task description")
```
"""


# =============================================================================
# Main Agent Creation
# =============================================================================


def create_skill_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    skills: list[dict[str, Any] | SkillConfig] | None = None,
    skills_config_path: str | Path | None = None,
    skills_directory: str | Path | None = None,
    root_dir: str | None = None,
    checkpointer: Any | None = None,
    store: Any | None = None,
    include_planning_tools: bool = True,
    include_filesystem_tools: bool = True,
    max_iterations: int = 100,
    debug: bool = False,
) -> CompiledStateGraph:
    """Create a skill-based supervisor agent without middleware.

    This function creates an agent that can:
    - Plan tasks using todo lists
    - Interact with the filesystem
    - Invoke specialized skills for domain-specific tasks

    Args:
        model: The language model to use. Can be a string like "anthropic:claude-sonnet-4-20250514"
               or a BaseChatModel instance. Defaults to Claude Sonnet.
        tools: Additional tools for the agent.
        system_prompt: Custom system prompt to prepend to the base prompt.
        skills: List of skill configurations (dicts or SkillConfig objects).
        skills_config_path: Path to YAML file containing skill definitions.
        skills_directory: Path to directory containing skill plugin modules (.py files).
                         Each module should have a register(registry) function.
        root_dir: Root directory for filesystem operations. If None, filesystem
                  tools operate in a limited mode.
        checkpointer: Optional checkpointer for state persistence.
        store: Optional store for persistent storage.
        include_planning_tools: Whether to include todo list tools.
        include_filesystem_tools: Whether to include filesystem tools.
        max_iterations: Maximum agent iterations before stopping.
        debug: Enable debug logging.

    Returns:
        Compiled LangGraph ready for invocation.

    Example:
        ```python
        from deepagents import create_skill_agent

        # Using skill plugins from directory (progressive disclosure)
        agent = create_skill_agent(
            model="anthropic:claude-sonnet-4-20250514",
            skills_directory="./skills/",  # Drop .py files here to add skills
        )

        # Or using inline skill definitions
        agent = create_skill_agent(
            model="anthropic:claude-sonnet-4-20250514",
            skills=[
                {
                    "name": "research",
                    "description": "Research a topic",
                    "system_prompt": "You are a researcher...",
                    "tools": [web_search]
                }
            ]
        )

        result = await agent.ainvoke({
            "messages": [HumanMessage(content="Research AI trends")]
        })
        ```
    """
    # Initialize model
    if model is None:
        try:
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(
                model_name="claude-sonnet-4-5-20250929",
                max_tokens=20000,
            )
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for the default model. "
                "Install it with: pip install langchain-anthropic"
            )
    elif isinstance(model, str):
        # Try to use init_chat_model if available (langchain 1.0+)
        # Otherwise, fall back to direct model instantiation
        try:
            from langchain.chat_models import init_chat_model

            model = init_chat_model(model)
        except ImportError:
            # Fallback: parse provider:model format and instantiate directly
            if ":" in model:
                provider, model_name = model.split(":", 1)
            else:
                provider, model_name = "anthropic", model

            if provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                model = ChatAnthropic(model_name=model_name, max_tokens=20000)
            elif provider == "openai":
                from langchain_openai import ChatOpenAI

                model = ChatOpenAI(model=model_name)
            else:
                raise ValueError(
                    f"Unknown provider '{provider}'. Use BaseChatModel instance or install langchain>=1.0"
                )

    # Build tools list
    all_tools: list[BaseTool] = []

    if include_planning_tools:
        all_tools.extend(_create_todo_tools())

    if include_filesystem_tools:
        all_tools.extend(_create_filesystem_tools(root_dir))

    # Add user-provided tools
    if tools:
        for tool in tools:
            if isinstance(tool, BaseTool):
                all_tools.append(tool)
            elif callable(tool):
                all_tools.append(StructuredTool.from_function(tool))

    # Build skill registry
    skill_registry = SkillRegistry()

    if skills:
        for skill_config in skills:
            skill_registry.register(skill_config)

    if skills_config_path:
        skill_registry.load_from_config(skills_config_path)

    if skills_directory:
        skill_registry.load_from_directory(skills_directory)

    # Create invoke_skill tool if skills are registered
    if skill_registry.skills:
        invoke_skill_tool = skill_registry.create_invoke_skill_tool(model)
        all_tools.append(invoke_skill_tool)

    # Build system prompt
    if skill_registry.skills:
        skills_list = "\n".join(
            f"- **{s['name']}**: {s['description']}"
            for s in skill_registry.list_skills()
        )
        skills_section = SKILLS_SECTION_TEMPLATE.format(skills_list=skills_list)
    else:
        skills_section = "(No skills configured)"

    final_system_prompt = BASE_SYSTEM_PROMPT.format(skills_section=skills_section)
    if system_prompt:
        final_system_prompt = system_prompt + "\n\n" + final_system_prompt

    # Bind tools to model
    if all_tools:
        model_with_tools = model.bind_tools(all_tools)
    else:
        model_with_tools = model

    # Build the graph
    builder = StateGraph(AgentState)

    # Agent node
    def agent_node(state: AgentState) -> dict[str, Any]:
        """Call the model and return response."""
        messages = state["messages"]

        # Ensure system message is present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=final_system_prompt), *messages]

        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    # Routing function
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Determine if we should continue or end."""
        messages = state["messages"]
        if not messages:
            return "end"

        last_message = messages[-1]

        # Check if the model wants to call tools
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        return "end"

    # Add nodes
    builder.add_node("agent", agent_node)
    if all_tools:
        builder.add_node("tools", ToolNode(all_tools))

    # Add edges
    builder.set_entry_point("agent")

    if all_tools:
        builder.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": "__end__"}
        )
        builder.add_edge("tools", "agent")
    else:
        builder.add_edge("agent", "__end__")

    # Compile with optional checkpointer
    graph = builder.compile(checkpointer=checkpointer, store=store)

    # Set recursion limit
    return graph.with_config({"recursion_limit": max_iterations * 2})


__all__ = [
    "AgentState",
    "create_skill_agent",
]
