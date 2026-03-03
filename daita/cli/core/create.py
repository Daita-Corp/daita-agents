"""
Simple component creation for Daita CLI.
Creates basic agents and workflows with proper path handling.
"""
from ..utils import find_project_root

def create_from_template(template, name=None, variant=None, verbose=False):
    """Create a component from a template."""
    
    # Find project root
    project_root = find_project_root()
    if not project_root:
        raise ValueError("Not in a Daita project. Run 'daita init' first.")
    
    # Get name if not provided
    if not name:
        name = input(f"Enter {template} name: ").strip()
    
    # Validate name
    if not name or not name.replace('_', '').replace('-', '').isalnum():
        raise ValueError(f"Invalid {template} name: {name}")
    
    # Convert name to valid Python identifier
    clean_name = name.replace('-', '_').lower()
    
    # Create the component
    if template == 'agent':
        _create_agent(project_root, clean_name, verbose)
    elif template == 'workflow':
        _create_workflow(project_root, clean_name, verbose)
    else:
        raise ValueError(f"Unknown template: {template}")
    
    print(f" Created {template}: {clean_name}")

def _create_agent(project_root, name, verbose=False):
    """Create a simple agent."""
    class_name = _to_class_name(name)
    
    code = f'''"""
{class_name} Agent

Replace this with your own agent logic.
"""
from daita import Agent

def create_agent():
    """Create the agent instance using direct Agent pattern."""
    # Option 1: Simple instantiation (uses defaults)
    agent = Agent(name="{class_name}")
    
    # Option 2: Direct LLM configuration (uncomment and modify as needed)
    # import os
    # agent = Agent(
    #     name="{class_name}",
    #     llm_provider="openai",
    #     model="gpt-4",
    #     api_key=os.getenv("OPENAI_API_KEY")
    # )
    
    # Optional: Add plugins
    # from daita.plugins import postgresql
    # agent.add_plugin(postgresql(host="localhost", database="mydb"))
    
    return agent

if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = create_agent()
        result = await agent.process("test_task", "Hello, world!")
        print(result)
    
    asyncio.run(main())
'''
    
    # Ensure agents directory exists
    agents_dir = project_root / 'agents'
    agents_dir.mkdir(exist_ok=True)
    
    # Create agent file
    agent_file = agents_dir / f'{name}.py'
    
    if agent_file.exists():
        raise ValueError(f"Agent {name} already exists")
    
    agent_file.write_text(code)
    
    # Prompt for display name
    default_display = name.replace('_', ' ').title()
    print(f"\n  Display name for deployment:")
    try:
        display_name = input(f"   Press Enter for '{default_display}' or type custom name: ").strip()
        if not display_name:
            display_name = default_display
    except (EOFError, KeyboardInterrupt):
        # Non-interactive environment or user cancelled
        display_name = default_display
        print(f"   Using default: '{default_display}'")
    except Exception:
        # Any other input error
        display_name = default_display
        print(f"   Input error, using default: '{default_display}'")
    
    # Update project config with display name
    _update_project_config(project_root, 'agents', name, display_name)
    
    if verbose:
        print(f"    Created: {agent_file.relative_to(project_root)}")
        print(f"    Updated project config")
    
    print(f"     Display name: '{display_name}'")

def _create_workflow(project_root, name, verbose=False):
    """Create a simple workflow."""
    class_name = _to_class_name(name)
    
    code = f'''"""
{class_name} Workflow

Replace this with your own workflow logic.
"""
from daita import Agent, Workflow

class {class_name}:
    """A simple workflow."""
    
    def __init__(self):
        self.workflow = Workflow("{name}")
        
        # Add your agents here
        # agent = Agent(name="Agent")
        # self.workflow.add_agent("agent", agent)
    
    async def run(self, data=None):
        """
        Run the workflow.
        
        Replace this with your own logic.
        """
        try:
            await self.workflow.start()
            
            # Your workflow logic here
            result = f"Workflow {class_name} processed: {{data}}"
            
            return {{
                'status': 'success',
                'result': result
            }}
            
        finally:
            await self.workflow.stop()

def create_workflow():
    """Create the workflow instance."""
    return {class_name}()

if __name__ == "__main__":
    import asyncio
    
    async def main():
        workflow = {class_name}()
        result = await workflow.run("test data")
        print(result)
    
    asyncio.run(main())
'''
    
    # Ensure workflows directory exists
    workflows_dir = project_root / 'workflows'
    workflows_dir.mkdir(exist_ok=True)
    
    # Create workflow file
    workflow_file = workflows_dir / f'{name}.py'
    
    if workflow_file.exists():
        raise ValueError(f"Workflow {name} already exists")
    
    workflow_file.write_text(code)
    
    # Prompt for display name
    default_display = name.replace('_', ' ').title()
    print(f"\n  Display name for deployment:")
    try:
        display_name = input(f"   Press Enter for '{default_display}' or type custom name: ").strip()
        if not display_name:
            display_name = default_display
    except (EOFError, KeyboardInterrupt):
        # Non-interactive environment or user cancelled
        display_name = default_display
        print(f"   Using default: '{default_display}'")
    except Exception:
        # Any other input error
        display_name = default_display
        print(f"   Input error, using default: '{default_display}'")
    
    # Update project config with display name
    _update_project_config(project_root, 'workflows', name, display_name)
    
    if verbose:
        print(f"    Created: {workflow_file.relative_to(project_root)}")
        print(f"    Updated project config")
    
    print(f"     Display name: '{display_name}'")


def _update_project_config(project_root, component_type, name, display_name=None):
    """Update daita-project.yaml with new component."""
    import yaml
    from datetime import datetime
    
    config_file = project_root / 'daita-project.yaml'
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Ensure arrays exist
    if component_type not in config:
        config[component_type] = []
    
    # Generate default display name if not provided
    if not display_name:
        display_name = name.replace('_', ' ').title()
    
    # Add component if not already present
    component_entry = {
        'name': name,
        'display_name': display_name,
        'type': 'standard' if component_type == 'agents' else 'basic',
        'created_at': datetime.now().isoformat()
    }
    
    # Check if already exists
    existing = next((c for c in config[component_type] if c['name'] == name), None)
    if not existing:
        config[component_type].append(component_entry)
        
        # Write back to file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


def _to_class_name(name):
    """Convert snake_case or kebab-case to PascalCase."""
    # Split on underscores and hyphens, capitalize each part
    parts = name.replace('-', '_').split('_')
    return ''.join(word.capitalize() for word in parts if word)