"""Tests for the middleware-free skills architecture."""

import os
import tempfile
from pathlib import Path

import pytest

# Set a test API key to avoid authentication errors
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from deepagents.skills import (
    Skill,
    SkillConfig,
    SkillExecutor,
    SkillRegistry,
    SkillResult,
    SimpleSkill,
    create_invoke_skill_tool,
    load_skills_from_yaml,
)
from deepagents.graph_no_middleware import create_skill_agent, create_trading_agent


class TestSkillConfig:
    """Tests for SkillConfig."""

    def test_create_from_dict(self) -> None:
        """Test creating SkillConfig from dictionary."""
        data = {
            "name": "test-skill",
            "description": "A test skill",
            "system_prompt": "You are helpful.",
            "tools": [],
            "max_iterations": 10,
        }
        config = SkillConfig.from_dict(data)

        assert config.name == "test-skill"
        assert config.description == "A test skill"
        assert config.system_prompt == "You are helpful."
        assert config.max_iterations == 10

    def test_create_from_dict_missing_name(self) -> None:
        """Test that missing name raises ValueError."""
        with pytest.raises(ValueError, match="name"):
            SkillConfig.from_dict({"description": "test"})

    def test_create_from_dict_missing_description(self) -> None:
        """Test that missing description raises ValueError."""
        with pytest.raises(ValueError, match="description"):
            SkillConfig.from_dict({"name": "test"})

    def test_to_dict(self) -> None:
        """Test converting SkillConfig to dictionary."""
        config = SkillConfig(
            name="my-skill",
            description="My skill",
            system_prompt="Helpful assistant",
        )
        data = config.to_dict()

        assert data["name"] == "my-skill"
        assert data["description"] == "My skill"
        assert data["system_prompt"] == "Helpful assistant"


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def test_register_skill_from_dict(self) -> None:
        """Test registering a skill from dictionary."""
        registry = SkillRegistry()
        registry.register({
            "name": "test-skill",
            "description": "A test skill",
            "system_prompt": "You are helpful.",
        })

        assert "test-skill" in registry.skills
        assert registry.get("test-skill") is not None

    def test_register_skill_from_config(self) -> None:
        """Test registering a skill from SkillConfig."""
        registry = SkillRegistry()
        config = SkillConfig(
            name="config-skill",
            description="From config",
            system_prompt="Helpful",
        )
        registry.register(config)

        assert "config-skill" in registry.skills

    def test_list_skills(self) -> None:
        """Test listing all skills."""
        registry = SkillRegistry()
        registry.register({
            "name": "skill-a",
            "description": "Skill A",
            "system_prompt": "A",
        })
        registry.register({
            "name": "skill-b",
            "description": "Skill B",
            "system_prompt": "B",
        })

        skills = registry.list_skills()
        assert len(skills) == 2
        names = [s["name"] for s in skills]
        assert "skill-a" in names
        assert "skill-b" in names

    def test_unregister_skill(self) -> None:
        """Test unregistering a skill."""
        registry = SkillRegistry()
        registry.register({
            "name": "temp-skill",
            "description": "Temporary",
            "system_prompt": "Temp",
        })

        assert "temp-skill" in registry.skills
        registry.unregister("temp-skill")
        assert "temp-skill" not in registry.skills

    def test_load_from_yaml(self) -> None:
        """Test loading skills from YAML file."""
        yaml_content = """
skills:
  - name: yaml-skill
    description: From YAML
    system_prompt: YAML assistant
  - name: another-skill
    description: Another one
    system_prompt: Another assistant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            registry = SkillRegistry()
            registry.load_from_config(temp_path)

            assert len(registry.skills) == 2
            assert "yaml-skill" in registry.skills
            assert "another-skill" in registry.skills
        finally:
            Path(temp_path).unlink()

    def test_load_from_dict(self) -> None:
        """Test loading skills from dictionary."""
        config = {
            "skills": [
                {
                    "name": "dict-skill",
                    "description": "From dict",
                    "system_prompt": "Dict assistant",
                }
            ]
        }

        registry = SkillRegistry()
        registry.load_from_dict(config)

        assert "dict-skill" in registry.skills


class TestSkillResult:
    """Tests for SkillResult."""

    def test_successful_result(self) -> None:
        """Test creating a successful result."""
        result = SkillResult(
            skill_name="test-skill",
            success=True,
            result="Task completed",
            execution_time=1.5,
        )

        assert result.success
        assert result.result == "Task completed"
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test creating a failed result."""
        result = SkillResult(
            skill_name="test-skill",
            success=False,
            error="Something went wrong",
        )

        assert not result.success
        assert result.result is None
        assert result.error == "Something went wrong"


class TestCreateSkillAgent:
    """Tests for create_skill_agent function."""

    def test_create_basic_agent(self) -> None:
        """Test creating a basic agent."""
        agent = create_skill_agent(
            system_prompt="You are a test agent.",
            include_filesystem_tools=False,
            include_planning_tools=False,
        )

        # Should return a compiled graph
        assert hasattr(agent, "invoke")
        assert hasattr(agent, "ainvoke")

    def test_create_agent_with_skills(self) -> None:
        """Test creating an agent with skills."""
        agent = create_skill_agent(
            system_prompt="You are a skilled agent.",
            skills=[
                {
                    "name": "research",
                    "description": "Research topics",
                    "system_prompt": "You are a researcher.",
                },
                {
                    "name": "analysis",
                    "description": "Analyze data",
                    "system_prompt": "You are an analyst.",
                },
            ],
            include_filesystem_tools=False,
        )

        assert hasattr(agent, "invoke")

    def test_create_trading_agent(self) -> None:
        """Test creating a trading agent."""
        agent = create_trading_agent(
            system_prompt="You are Alpha Cortex.",
        )

        assert hasattr(agent, "invoke")


class TestSimpleSkill:
    """Tests for SimpleSkill class."""

    def test_skill_properties(self) -> None:
        """Test skill name and description properties."""
        config = SkillConfig(
            name="prop-skill",
            description="Property test",
            system_prompt="Test",
        )
        skill = SimpleSkill(config)

        assert skill.name == "prop-skill"
        assert skill.description == "Property test"
