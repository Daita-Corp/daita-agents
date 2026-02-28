"""
Simple testing for Daita CLI.
Just runs agents and workflows locally.
"""
import asyncio
import json
import sys
from pathlib import Path
import importlib.util
from ..utils import find_project_root

async def run_tests(target=None, data_file=None, watch=False, verbose=False):
   """Run tests for agents or workflows."""
   
   # Find project root
   project_root = find_project_root()
   if not project_root:
       raise ValueError("Not in a Daita project. Run 'daita init' first.")
   
   # Load test data
   test_data = _load_test_data(project_root, data_file)
   
   if target:
       # Test specific agent or workflow
       await _test_single(project_root, target, test_data, verbose)
   else:
       # Test everything
       await _test_all(project_root, test_data, verbose)
   
   if watch:
       print(f"\n  Watching for changes... (Press Ctrl+C to stop)")
       try:
           while True:
               await asyncio.sleep(1)
       except KeyboardInterrupt:
           print(f"\n Stopped watching")

async def _test_single(project_root, target, test_data, verbose):
   """Test a single agent or workflow."""
   print(f" Testing: {target}")
   
   # Try to find and load the target
   agent_file = project_root / 'agents' / f'{target}.py'
   workflow_file = project_root / 'workflows' / f'{target}.py'
   
   if agent_file.exists():
       await _test_agent(agent_file, target, test_data, verbose)
   elif workflow_file.exists():
       await _test_workflow(workflow_file, target, test_data, verbose)
   else:
       print(f" Not found: {target}")

async def _test_all(project_root, test_data, verbose):
   """Test all agents and workflows."""
   agents = _list_python_files(project_root / 'agents')
   workflows = _list_python_files(project_root / 'workflows')
   
   print(f" Testing {len(agents)} agents and {len(workflows)} workflows")
   
   # Test agents
   for agent in agents:
       agent_file = project_root / 'agents' / f'{agent}.py'
       await _test_agent(agent_file, agent, test_data, verbose)
   
   # Test workflows
   for workflow in workflows:
       workflow_file = project_root / 'workflows' / f'{workflow}.py'
       await _test_workflow(workflow_file, workflow, test_data, verbose)

async def _test_agent(agent_file, name, test_data, verbose):
   """Test a single agent."""
   try:
       # Load the agent factory function
       try:
           agent_factory = _load_python_file(agent_file, 'create_agent')
       except Exception as e:
           print(f" {name}: Failed to load agent - {str(e)}")
           return
       
       # Create agent instance
       try:
           agent_instance = agent_factory()
       except Exception as e:
           print(f" {name}: Failed to create agent instance - {str(e)}")
           if verbose:
               import traceback
               traceback.print_exc()
           return
       
       # Test basic processing using v2.0+ API
       try:
           # Start the agent
           await agent_instance.start()

           # Run with test data - use run_detailed() to get full execution info
           test_prompt = f"Process this test data: {test_data}"
           result = await agent_instance.run_detailed(test_prompt)

           # Validate result format from run_detailed()
           if not isinstance(result, dict):
               print(f"  {name}: Warning - agent returned {type(result).__name__} instead of dict")

           print(f" {name}: OK")
           if verbose:
               print(f"   Answer: {result.get('result', 'N/A')[:100]}")
               print(f"   Cost: ${result.get('cost', 0):.6f}")
               print(f"   Time: {result.get('processing_time_ms', 0):.0f}ms")
               
       except Exception as e:
           print(f" {name}: Processing failed - {str(e)}")
           if verbose:
               import traceback
               traceback.print_exc()
           
   except Exception as e:
       print(f" {name}: Unexpected error - {str(e)}")
       if verbose:
           import traceback
           traceback.print_exc()

async def _test_workflow(workflow_file, name, test_data, verbose):
   """Test a single workflow."""
   try:
       # Load the workflow factory function
       try:
           workflow_factory = _load_python_file(workflow_file, 'create_workflow')
       except Exception as e:
           print(f" {name}: Failed to load workflow - {str(e)}")
           return

       # Create workflow instance
       try:
           workflow_instance = workflow_factory()
       except Exception as e:
           print(f" {name}: Failed to create workflow instance - {str(e)}")
           if verbose:
               import traceback
               traceback.print_exc()
           return

       # Test workflow execution
       try:
           # Start the workflow
           await workflow_instance.start()

           # If workflow has agents, inject test data into the first agent
           if workflow_instance.agents:
               first_agent_name = list(workflow_instance.agents.keys())[0]
               await workflow_instance.inject_data(first_agent_name, test_data)

               # Give workflow a moment to process
               await asyncio.sleep(1)

           # Stop the workflow
           await workflow_instance.stop()

           print(f" {name}: OK")
           if verbose:
               print(f"   Agents: {len(workflow_instance.agents)}")
               print(f"   Connections: {len(workflow_instance.connections)}")

       except Exception as e:
           print(f" {name}: Workflow run failed - {str(e)}")
           if verbose:
               import traceback
               traceback.print_exc()

   except Exception as e:
       print(f" {name}: Unexpected error - {str(e)}")
       if verbose:
           import traceback
           traceback.print_exc()

def _load_python_file(file_path, factory_function):
   """Load a Python file and get the factory function."""
   try:
       # Add project root to path
       project_root = file_path.parent.parent
       if str(project_root) not in sys.path:
           sys.path.insert(0, str(project_root))
       
       # Check if file exists and is readable
       if not file_path.exists():
           raise FileNotFoundError(f"File not found: {file_path}")
       
       if not file_path.is_file():
           raise ValueError(f"Path is not a file: {file_path}")
       
       # Load module with better error handling
       spec = importlib.util.spec_from_file_location("module", file_path)
       if spec is None:
           raise ImportError(f"Could not create module spec for {file_path}")
       
       module = importlib.util.module_from_spec(spec)
       if module is None:
           raise ImportError(f"Could not create module from spec for {file_path}")
       
       # Execute module with error handling
       try:
           spec.loader.exec_module(module)
       except Exception as e:
           raise ImportError(f"Failed to execute module {file_path.name}: {str(e)}")
       
       # Get factory function
       if hasattr(module, factory_function):
           return getattr(module, factory_function)
       else:
           # List available functions for better error message
           available_functions = [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')]
           raise ValueError(
               f"No {factory_function}() function found in {file_path.name}. "
               f"Available functions: {available_functions}"
           )
   
   except (ImportError, ValueError, FileNotFoundError) as e:
       # Re-raise known errors
       raise
   except Exception as e:
       # Catch any other unexpected errors
       raise RuntimeError(f"Unexpected error loading {file_path.name}: {str(e)}")

def _load_test_data(project_root, data_file):
   """Load test data from file or use default."""
   if data_file:
       data_path = Path(data_file)
       if not data_path.is_absolute():
           data_path = project_root / data_path
       
       if data_path.exists():
           if data_path.suffix == '.json':
               with open(data_path, 'r') as f:
                   return json.load(f)
           else:
               with open(data_path, 'r') as f:
                   return f.read()
   
   # Use default test data
   return {
       "test": True,
       "message": "Default test data"
   }

def _list_python_files(directory):
   """List Python files in a directory (excluding __init__.py)."""
   if not directory.exists():
       return []
   
   files = []
   for file in directory.glob('*.py'):
       if file.name != '__init__.py':
           files.append(file.stem)
   return files

