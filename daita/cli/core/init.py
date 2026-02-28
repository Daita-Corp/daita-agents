"""
Simple project initialization for Daita CLI.
Creates minimal, universal project template like create-react-app.
"""
import os
import yaml
from pathlib import Path
from datetime import datetime
from ..utils import find_project_root

async def initialize_project(project_name=None, project_type='basic', template=None, force=False, verbose=False):
    """Initialize a new Daita project with minimal template."""
    
    # Get project name
    if not project_name:
        project_name = input("Project name: ").strip()
        if not project_name:
            project_name = Path.cwd().name
    
    # Determine project directory
    project_dir = Path.cwd() / project_name
    
    # Check if directory exists
    if project_dir.exists() and not force:
        if any(project_dir.iterdir()):  # Directory not empty
            confirm = input(f"Directory '{project_name}' exists and is not empty. Continue? (y/N): ")
            if confirm.lower() != 'y':
                print(" Initialization cancelled")
                return
    
    # Create project directory
    project_dir.mkdir(exist_ok=True)
    
    print(f" Creating Daita project: {project_name}")
    print(f" Location: {project_dir}")
    
    # Create minimal project structure
    _create_project_structure(project_dir, verbose)
    _create_project_config(project_dir, project_name, verbose)
    _create_starter_files(project_dir, project_name, verbose)
    _create_supporting_files(project_dir, project_name, verbose)
    
    # Import freemium utilities
    try:
        from ..utils import get_freemium_success_message
        print("")
        print(get_freemium_success_message(project_name))
        print("")
        print(" Development setup:")
        print(f"   export OPENAI_API_KEY=your_key_here    # Configure LLM or another provider")
        print(f"   pip install -r requirements.txt       # Install dependencies")
        print(f"   python agents/my_agent.py              # Test example agent")
    except ImportError:
        # Fallback to original message if utils not available
        print(f"")
        print(f"Project created successfully")
        print(f"")
        print(f"Get started:")
        print(f"   cd {project_name}")
        print(f"   export OPENAI_API_KEY=your_key_here or another provider")
        print(f"   pip install -r requirements.txt")
        print(f"   python agents/my_agent.py          # Test the example agent")
        print(f"   daita create agent new_agent       # Create a new agent")
        print(f"   daita test                         # Test all components")
        print(f"   daita test --watch                 # Watch for changes while developing")

def _create_project_structure(project_dir, verbose):
    """Create minimal directory structure."""
    directories = [
        '.daita',
        'agents',
        'workflows', 
        'data',
        'tests'
    ]
    
    for dir_name in directories:
        dir_path = project_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        
        # Create __init__.py for Python packages
        if dir_name in ['agents', 'workflows', 'tests']:
            init_file = dir_path / '__init__.py'
            init_file.write_text('"""Daita project components."""\n')
        
        if verbose:
            print(f"    Created: {dir_name}/")

def _create_project_config(project_dir, project_name, verbose):
    """Create minimal daita-project.yaml configuration."""
    
    config = {
        'name': project_name,
        'version': '1.0.0',
        'description': f'A Daita AI agent project',
        'created_at': datetime.utcnow().isoformat(),

        # Project components (will be populated as user creates them)
        'agents': [],
        'workflows': []
    }
    
    config_file = project_dir / 'daita-project.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    if verbose:
        print(f"    Created: daita-project.yaml")

def _create_starter_files(project_dir, project_name, verbose):
    """Create minimal starter agent and workflow files."""
    
    # Simple data-focused starter agent
    starter_agent = '''"""
My Agent - Data Processing Example

A simple data processing agent.
"""
from daita import Agent
from daita.core.tools import tool

# Simple data processing tool
@tool
async def calculate_stats(data: list) -> dict:
    """Calculate basic statistics for a list of numbers."""
    if not data:
        return {"error": "No data provided"}

    return {
        "count": len(data),
        "sum": sum(data),
        "avg": sum(data) / len(data),
        "min": min(data),
        "max": max(data)
    }

def create_agent():
    """Create the data processing agent."""
    agent = Agent(
        name="Data Processor",
        model="gpt-4o-mini",
        prompt="You are a data analyst. Help users analyze and process data."
    )

    # Register data processing tools
    agent.register_tool(calculate_stats)

    # Add database plugin (optional - uncomment to use)
    # from daita.plugins import postgresql
    # db = postgresql(host="localhost", database="mydb", user="user", password="pass")
    # agent = Agent(name="Data Processor", tools=[db])

    return agent

if __name__ == "__main__":
    import asyncio

    async def main():
        """Example: Process sample data with the clean API."""
        agent = create_agent()
        await agent.start()

        try:
            # Sample data
            sales_data = [100, 250, 175, 300, 225]

            # Simple usage - just get the answer
            answer = await agent.run(
                f"Analyze these sales numbers and tell me the insights: {sales_data}"
            )
            print(f"Analysis: {answer}")

            # Detailed usage - get full metadata (cost, time, tools used)
            result = await agent.run_detailed(
                f"What's the average and total of: {sales_data}?"
            )
            print(f"\\nDetailed result:")
            print(f"  Answer: {result['result']}")
            print(f"  Processing time: {result.get('processing_time_ms', 0):.0f}ms")
            print(f"  Tools used: {len(result.get('tool_calls', []))}")

        finally:
            await agent.stop()

    asyncio.run(main())
'''
    
    # Simple data pipeline workflow
    starter_workflow = '''"""
My Workflow - Data Pipeline

A simple data processing pipeline.
"""
from daita import Agent, Workflow

def create_workflow():
    """Create a data processing pipeline."""
    workflow = Workflow("Data Pipeline")

    # Agent 1: Data validator
    validator = Agent(
        name="Data Validator",
        model="gpt-4o-mini",
        prompt="You validate data quality and flag issues."
    )

    # Agent 2: Data analyzer
    analyzer = Agent(
        name="Data Analyzer",
        model="gpt-4o-mini",
        prompt="You analyze data and extract insights."
    )

    # Add agents to workflow
    workflow.add_agent("validator", validator)
    workflow.add_agent("analyzer", analyzer)

    # Connect: validator -> analyzer
    workflow.connect("validator", "validated_data", "analyzer")

    return workflow

async def run_workflow(data=None):
    """Run the data pipeline."""
    workflow = create_workflow()

    try:
        await workflow.start()

        # Sample data
        sample_data = data or {
            "records": [
                {"id": 1, "amount": 100, "status": "completed"},
                {"id": 2, "amount": 250, "status": "pending"}
            ]
        }

        await workflow.inject_data("validator", sample_data)

        return {
            'status': 'success',
            'message': f'Processed {len(sample_data.get("records", []))} records'
        }

    finally:
        await workflow.stop()

if __name__ == "__main__":
    import asyncio

    async def main():
        result = await run_workflow()
        print(f"Result: {result}")

    asyncio.run(main())
'''
    
    # Write starter files
    (project_dir / 'agents' / 'my_agent.py').write_text(starter_agent)
    (project_dir / 'workflows' / 'my_workflow.py').write_text(starter_workflow)
    
    if verbose:
        print(f"    Created: agents/my_agent.py")
        print(f"    Created: workflows/my_workflow.py")

def _create_supporting_files(project_dir, project_name, verbose):
    """Create supporting files (requirements, README, etc.)."""

    # Detect user's installed daita-agents version
    try:
        import importlib.metadata
        user_version = importlib.metadata.version('daita-agents')
        if verbose:
            print(f"    Detected daita-agents version: {user_version}")
    except Exception:
        user_version = "0.6.2"  # Default to current if detection fails

    # Minimal requirements.txt with exact user version
    requirements = f'''# Daita Agents Framework
daita-agents=={user_version}

# LLM provider (choose one)
openai>=1.0.0

# Development
pytest>=7.0.0
pytest-asyncio>=0.21.0
'''
    
    # Simple .gitignore
    gitignore = '''# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
dist/
*.egg-info/

# Virtual environments
.env
.venv
venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store

# API keys
.env.local

# Memory files (may contain sensitive data)
# Remove this line if you want to commit agent memories to git
.daita/memory/
'''
    
    # README with freemium messaging
    readme = f'''# {project_name}

A Daita AI agent project.

## Quick Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your LLM API key:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

## Free Local Development

Build and test your agents locally - completely free:

```bash
# Test the example agent
python agents/my_agent.py

# Test all components
daita test

# Watch for changes while developing
daita test --watch

# Create new components
daita create agent my_new_agent
daita create workflow my_new_workflow
```

## Production Cloud Hosting

Ready to deploy to the cloud? Get 24/7 hosting, monitoring, and insights:

```bash
# Get your API key at daita-tech.io
export DAITA_API_KEY='your-key-here'

# Deploy to cloud
daita push                   # Deploy to production

# Monitor your deployments
daita status                 # Deployment status
daita logs                   # View execution logs
```

## Project Structure

```
{project_name}/
├── agents/          # Your AI agents (free to create & test)
│   └── my_agent.py
├── workflows/       # Your workflows (free to create & test)
│   └── my_workflow.py
├── data/           # Data files
├── tests/          # Tests
└── daita-project.yaml  # Project config
```

## Command Reference

**Free Commands (Local Development):**
- `daita test` - Test all agents and workflows
- `daita test --watch` - Development mode with auto-reload
- `daita create agent <name>` - Create new agent
- `daita create workflow <name>` - Create new workflow

**Premium Commands (Cloud Hosting):**
- `daita push <env>` - Deploy to cloud
- `daita status` - Monitor deployments
- `daita logs <env>` - View execution logs
- `daita run <agent>` - Execute remotely

## Learn More

-  [Get API Key](https://daita-tech.io) - Start your free trial
-  [Documentation](https://docs.daita-tech.io)
'''
    
    # Simple test file for data processing
    test_file = '''"""
Basic tests for data processing agents and workflows.
"""
import pytest
import asyncio

@pytest.mark.asyncio
async def test_agent_run():
    """Test agent with run() API."""
    from agents.my_agent import create_agent

    agent = create_agent()
    await agent.start()

    try:
        # Test data processing
        answer = await agent.run("Calculate the average of [10, 20, 30]")
        assert isinstance(answer, str)
        assert "20" in answer or "average" in answer.lower()

    finally:
        await agent.stop()

@pytest.mark.asyncio
async def test_agent_run_detailed():
    """Test agent with run_detailed() API."""
    from agents.my_agent import create_agent

    agent = create_agent()
    await agent.start()

    try:
        # Test with metadata
        result = await agent.run_detailed("What's the sum of [5, 10, 15]?")

        assert "result" in result
        assert isinstance(result["result"], str)
        assert "processing_time_ms" in result
        assert "agent_id" in result

    finally:
        await agent.stop()

@pytest.mark.asyncio
async def test_workflow():
    """Test data pipeline workflow."""
    from workflows.my_workflow import run_workflow

    result = await run_workflow({
        "records": [
            {"id": 1, "amount": 100},
            {"id": 2, "amount": 200}
        ]
    })

    assert result["status"] == "success"
    assert "records" in result["message"]

if __name__ == "__main__":
    async def main():
        print("Running agent tests...")
        await test_agent_run()
        await test_agent_run_detailed()
        print(" Agent tests passed!")

        print("Running workflow tests...")
        await test_workflow()
        print(" Workflow tests passed!")

        print("\\n All tests passed!")

    asyncio.run(main())
'''
    
    # Write all supporting files
    (project_dir / 'requirements.txt').write_text(requirements)
    (project_dir / '.gitignore').write_text(gitignore)
    (project_dir / 'README.md').write_text(readme)
    (project_dir / 'tests' / 'test_basic.py').write_text(test_file)
    
    # Create empty data directory with placeholder
    (project_dir / 'data' / '.gitkeep').write_text('')
    
    if verbose:
        print(f"    Created: requirements.txt")
        print(f"    Created: .gitignore")
        print(f"    Created: README.md")
        print(f"    Created: tests/test_basic.py")

