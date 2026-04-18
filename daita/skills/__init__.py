"""
Daita Skills — reusable units of agent capability.

Skills bundle related tools with domain-specific instructions,
sitting between raw tools and full plugins in the abstraction hierarchy.
"""

from .base import BaseSkill, Skill

__all__ = ["BaseSkill", "Skill"]
