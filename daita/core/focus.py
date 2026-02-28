"""
Focus parameter handling for Daita Agents.

This module provides utilities for applying focus parameters to different data types.
"""
import logging
from typing import Any, Optional, Union, List, Dict

from .exceptions import ValidationError, InvalidDataError

logger = logging.getLogger(__name__)

def apply_focus(
    data: Any,
    focus: Optional[Union[List[str], str, Dict[str, Any]]]
) -> Any:
    """
    Apply focus parameter to data.
    
    Args:
        data: Input data
        focus: Focus specification
        
    Returns:
        Focused data
        
    Raises:
        ValidationError: If focus configuration is invalid
        InvalidDataError: If data cannot be processed with the given focus
    """
    if focus is None or data is None:
        return data
    
    # Validate focus parameter first - this will raise ValidationError for empty strings
    _validate_focus_parameter(focus)
    
    try:
        # Handle pandas DataFrame
        if hasattr(data, 'columns'):
            return _apply_dataframe_focus(data, focus)
        
        # Handle dictionaries
        elif isinstance(data, dict):
            return _apply_dict_focus(data, focus)
        
        # Handle lists of dictionaries
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            return _apply_list_dict_focus(data, focus)
        
        # For other data types, check if focus makes sense
        else:
            if isinstance(focus, (list, dict)):
                raise InvalidDataError(
                    f"Cannot apply structured focus to {type(data).__name__}. "
                    f"Focus requires dict, DataFrame, or list of dicts."
                )
            # For simple string focus on other types, just return original data
            logger.info(f"Focus '{focus}' not applicable to {type(data).__name__}, returning original data")
            return data
            
    except (ValidationError, InvalidDataError):
        # Re-raise validation errors
        raise
    except Exception as e:
        # Wrap unexpected errors with context
        raise InvalidDataError(f"Failed to apply focus '{focus}' to {type(data).__name__}: {str(e)}")

def _validate_focus_parameter(focus: Union[List[str], str, Dict[str, Any]]) -> None:
    """
    Validate focus parameter format.
    
    Args:
        focus: Focus parameter to validate
        
    Raises:
        ValidationError: If focus format is invalid
    """
    if isinstance(focus, str):
        # Empty or whitespace-only strings are invalid
        if not focus or not focus.strip():
            raise ValidationError("Focus string cannot be empty or whitespace-only")
    
    elif isinstance(focus, list):
        if not focus:
            raise ValidationError("Focus list cannot be empty")
        
        # Check all items are strings and not empty
        for i, item in enumerate(focus):
            if not isinstance(item, str):
                raise ValidationError(f"Focus list item {i} must be string, got {type(item).__name__}")
            if not item or not item.strip():
                raise ValidationError(f"Focus list item {i} cannot be empty or whitespace-only")
    
    elif isinstance(focus, dict):
        if not focus:
            raise ValidationError("Focus dict cannot be empty")
        
        # Validate dict structure
        valid_keys = {'include', 'exclude', 'primary', 'secondary'}
        invalid_keys = set(focus.keys()) - valid_keys
        if invalid_keys:
            raise ValidationError(f"Invalid focus keys: {invalid_keys}. Valid keys: {valid_keys}")
        
        # Validate include/exclude values
        for key in ['include', 'exclude', 'primary', 'secondary']:
            if key in focus:
                value = focus[key]
                if isinstance(value, str):
                    if not value or not value.strip():
                        raise ValidationError(f"Focus '{key}' string cannot be empty or whitespace-only")
                elif isinstance(value, list):
                    if not value:
                        raise ValidationError(f"Focus '{key}' list cannot be empty")
                    for i, item in enumerate(value):
                        if not isinstance(item, str):
                            raise ValidationError(f"Focus '{key}' list item {i} must be string")
                        if not item or not item.strip():
                            raise ValidationError(f"Focus '{key}' list item {i} cannot be empty or whitespace-only")
                else:
                    raise ValidationError(f"Focus '{key}' must be string or list, got {type(value).__name__}")
    
    else:
        raise ValidationError(f"Focus must be string, list, or dict, got {type(focus).__name__}")

def _apply_dataframe_focus(data, focus):
    """Apply focus to pandas DataFrame."""
    try:
        available_columns = list(data.columns)
        
        if isinstance(focus, list):
            # List of column names
            missing_columns = [col for col in focus if col not in available_columns]
            if missing_columns:
                raise InvalidDataError(f"Columns not found in DataFrame: {missing_columns}")
            
            return data[focus]
        
        elif isinstance(focus, str):
            # Single column name
            if focus not in available_columns:
                raise InvalidDataError(f"Column '{focus}' not found in DataFrame. Available: {available_columns}")
            
            return data[[focus]]
        
        elif isinstance(focus, dict):
            # Dictionary-based focus
            if 'include' in focus:
                include_cols = focus['include'] if isinstance(focus['include'], list) else [focus['include']]
                missing_columns = [col for col in include_cols if col not in available_columns]
                if missing_columns:
                    raise InvalidDataError(f"Include columns not found: {missing_columns}")
                
                result = data[include_cols]
                
                # Apply exclusions
                if 'exclude' in focus:
                    exclude_cols = focus['exclude'] if isinstance(focus['exclude'], list) else [focus['exclude']]
                    # Only exclude columns that actually exist in the result
                    exclude_cols = [col for col in exclude_cols if col in result.columns]
                    if exclude_cols:
                        result = result.drop(columns=exclude_cols)
                
                return result
            
            # Primary/secondary focus
            elif 'primary' in focus:
                primary_cols = focus['primary'] if isinstance(focus['primary'], list) else [focus['primary']]
                missing_columns = [col for col in primary_cols if col not in available_columns]
                if missing_columns:
                    raise InvalidDataError(f"Primary columns not found: {missing_columns}")
                
                return data[primary_cols]
            
            else:
                raise ValidationError("Dict focus must contain 'include' or 'primary' key")
        
        return data
        
    except Exception as e:
        if isinstance(e, (ValidationError, InvalidDataError)):
            raise
        raise InvalidDataError(f"DataFrame focus application failed: {str(e)}")

def _apply_dict_focus(data, focus):
    """Apply focus to dictionary."""
    try:
        available_keys = list(data.keys())
        
        if isinstance(focus, list):
            # List of keys
            missing_keys = [key for key in focus if key not in available_keys]
            if missing_keys:
                raise InvalidDataError(f"Keys not found in dict: {missing_keys}")
            
            return {k: data[k] for k in focus}
        
        elif isinstance(focus, str):
            # Single key
            if focus not in available_keys:
                raise InvalidDataError(f"Key '{focus}' not found in dict. Available: {available_keys}")
            
            return {focus: data[focus]}
        
        elif isinstance(focus, dict):
            # Dictionary-based focus
            if 'include' in focus:
                include_keys = focus['include'] if isinstance(focus['include'], list) else [focus['include']]
                missing_keys = [key for key in include_keys if key not in available_keys]
                if missing_keys:
                    raise InvalidDataError(f"Include keys not found: {missing_keys}")
                
                result = {k: data[k] for k in include_keys}
                
                # Apply exclusions
                if 'exclude' in focus:
                    exclude_keys = focus['exclude'] if isinstance(focus['exclude'], list) else [focus['exclude']]
                    result = {k: v for k, v in result.items() if k not in exclude_keys}
                
                return result
            
            elif 'primary' in focus:
                primary_keys = focus['primary'] if isinstance(focus['primary'], list) else [focus['primary']]
                missing_keys = [key for key in primary_keys if key not in available_keys]
                if missing_keys:
                    raise InvalidDataError(f"Primary keys not found: {missing_keys}")
                
                return {k: data[k] for k in primary_keys}
            
            else:
                raise ValidationError("Dict focus must contain 'include' or 'primary' key")
        
        return data
        
    except Exception as e:
        if isinstance(e, (ValidationError, InvalidDataError)):
            raise
        raise InvalidDataError(f"Dictionary focus application failed: {str(e)}")

def _apply_list_dict_focus(data, focus):
    """Apply focus to list of dictionaries."""
    if not data:
        return data
    
    try:
        # Apply focus to first item to validate, then apply to all
        _apply_dict_focus(data[0], focus)
        return [_apply_dict_focus(item, focus) for item in data]
    except Exception as e:
        if isinstance(e, (ValidationError, InvalidDataError)):
            raise
        raise InvalidDataError(f"List of dicts focus application failed: {str(e)}")