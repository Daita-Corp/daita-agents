#!/usr/bin/env python3
"""
Import Detection for Daita Lambda Layer Optimization

Analyzes user code to detect package imports and determine which Lambda layers
are needed for deployment. This enables smart layer selection to minimize
package sizes while ensuring all dependencies are available.
"""

import ast
import os
from pathlib import Path
from typing import Set, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ImportDetector:
    """Detects package imports in Python code to optimize Lambda layer selection."""
    
    def __init__(self):
        # Define packages available in each layer
        self.layer_packages = {
            'common_dependencies': {
                'requests',
                'dateutil', 'python_dateutil',  # python-dateutil can be imported as either
                'PIL', 'pillow',  # Pillow can be imported as PIL
                'tqdm',
                'joblib'
            },
            'core_dependencies': {
                'pydantic', 'pydantic_core',
                'httpx',
                'aiofiles',
                'boto3', 'botocore'
            }
        }
        
        # Mapping of import names to actual package names
        self.import_mappings = {
            'PIL': 'pillow',
            'dateutil': 'python_dateutil',
            'cv2': 'opencv_python',
            'sklearn': 'scikit_learn',
            'skimage': 'scikit_image'
        }
    
    def analyze_file(self, file_path: Path) -> Set[str]:
        """Analyze a single Python file and return set of imported packages."""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        package_name = alias.name.split('.')[0]
                        imports.add(package_name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        package_name = node.module.split('.')[0]
                        imports.add(package_name)
            
        except Exception as e:
            logger.warning(f"Could not parse {file_path}: {e}")
        
        return imports
    
    def analyze_directory(self, directory: Path, exclude_patterns: Optional[List[str]] = None) -> Set[str]:
        """Analyze all Python files in a directory and return all imported packages."""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'node_modules', '.venv', 'venv']
        
        all_imports = set()
        
        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    file_imports = self.analyze_file(file_path)
                    all_imports.update(file_imports)
        
        return all_imports
    
    def normalize_package_name(self, import_name: str) -> str:
        """Normalize import name to actual package name using mappings."""
        return self.import_mappings.get(import_name, import_name)
    
    def detect_required_layers(self, imports: Set[str]) -> Dict[str, List[str]]:
        """Determine which layers are needed based on detected imports."""
        required_layers = {}
        
        # Normalize import names
        normalized_imports = {self.normalize_package_name(imp) for imp in imports}
        
        for layer_name, layer_packages in self.layer_packages.items():
            matching_packages = []
            
            for package in layer_packages:
                if package in normalized_imports or package in imports:
                    matching_packages.append(package)
            
            if matching_packages:
                required_layers[layer_name] = matching_packages
        
        return required_layers
    
    def analyze_project(self, project_path: Path) -> Dict[str, any]:
        """Analyze an entire project and return comprehensive import analysis."""
        logger.info(f" Analyzing imports in project: {project_path}")
        
        # Detect all imports
        all_imports = self.analyze_directory(project_path)
        
        # Determine required layers
        required_layers = self.detect_required_layers(all_imports)
        
        # Calculate optimization potential
        layer_packages_found = set()
        for packages in required_layers.values():
            layer_packages_found.update(packages)
        
        analysis = {
            'total_imports': len(all_imports),
            'all_imports': sorted(list(all_imports)),
            'required_layers': required_layers,
            'layer_packages_detected': sorted(list(layer_packages_found)),
            'optimization_potential': len(layer_packages_found) > 0
        }
        
        logger.info(f" Analysis complete:")
        logger.info(f"    Total imports detected: {analysis['total_imports']}")
        logger.info(f"    Layer packages found: {len(layer_packages_found)}")
        logger.info(f"    Layers needed: {list(required_layers.keys())}")
        
        return analysis
    
    def get_layer_arns_for_project(self, project_path: Path, layer_config_path: Path) -> List[str]:
        """Get the specific layer ARNs needed for a project based on its imports."""
        import json
        
        # Analyze project imports
        analysis = self.analyze_project(project_path)
        required_layers = analysis['required_layers']
        
        # Load layer configuration
        if not layer_config_path.exists():
            logger.warning(f"Layer config not found at {layer_config_path}")
            return []
        
        with open(layer_config_path, 'r') as f:
            layer_config = json.load(f)
        
        # Map layer types to ARNs
        layer_arn_mapping = {
            'common_dependencies': layer_config.get('ml_dependencies_layer_arn'),
            'core_dependencies': layer_config.get('dependencies_layer_arn')
        }
        
        # Get ARNs for required layers
        required_arns = []
        for layer_name in required_layers.keys():
            if layer_name in layer_arn_mapping and layer_arn_mapping[layer_name]:
                required_arns.append(layer_arn_mapping[layer_name])
                logger.info(f" Including {layer_name} layer: {layer_arn_mapping[layer_name]}")
        
        # Always include framework layer
        framework_arn = layer_config.get('framework_layer_arn')
        if framework_arn:
            required_arns.append(framework_arn)
            logger.info(f" Including framework layer: {framework_arn}")
        
        return required_arns

def main():
    """CLI entry point for import detection."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python import_detector.py <project_path>")
        sys.exit(1)
    
    project_path = Path(sys.argv[1])
    if not project_path.exists():
        print(f"Error: Project path {project_path} does not exist")
        sys.exit(1)
    
    detector = ImportDetector()
    analysis = detector.analyze_project(project_path)
    
    print("\n" + "="*60)
    print(" IMPORT ANALYSIS RESULTS")
    print("="*60)
    print(f" Total imports: {analysis['total_imports']}")
    print(f" Layer-optimizable packages: {len(analysis['layer_packages_detected'])}")
    print(f" Optimization potential: {'Yes' if analysis['optimization_potential'] else 'No'}")
    
    if analysis['required_layers']:
        print("\n Required layers:")
        for layer, packages in analysis['required_layers'].items():
            print(f"   {layer}: {', '.join(packages)}")
    
    print("\n All detected imports:")
    for imp in analysis['all_imports']:
        status = "" if imp in analysis['layer_packages_detected'] else ""
        print(f"   {status} {imp}")
    
    print("="*60)

if __name__ == "__main__":
    main()