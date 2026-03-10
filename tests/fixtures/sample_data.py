"""
Sample data sets for testing the Daita framework.

Provides realistic test data for various scenarios and complexity levels.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Any


class SampleData:
    """Container for all test data sets."""

    @staticmethod
    def simple_json_data() -> Dict[str, Any]:
        """Simple JSON data for basic testing."""
        return {
            "test": True,
            "message": "Simple test data",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "numbers": [1, 2, 3, 4, 5],
            "metadata": {"source": "test_fixtures", "version": "1.0"},
        }

    @staticmethod
    def sales_data() -> Dict[str, Any]:
        """Sales data for business intelligence testing."""
        return {
            "sales_data": [15000, 22000, 18000, 25000, 30000],
            "region": "West Coast",
            "quarter": "Q4 2024",
            "products": [
                {"name": "Product A", "revenue": 15000, "units": 150},
                {"name": "Product B", "revenue": 22000, "units": 200},
                {"name": "Product C", "revenue": 18000, "units": 180},
            ],
            "sales_reps": [
                {"name": "Alice Johnson", "quota": 20000, "achieved": 25000},
                {"name": "Bob Smith", "quota": 18000, "achieved": 22000},
            ],
        }

    @staticmethod
    def customer_feedback() -> Dict[str, Any]:
        """Customer feedback data for sentiment analysis."""
        return {
            "customer_feedback": [
                "Great product, very satisfied!",
                "Could be better, had some issues with delivery",
                "Excellent service, will definitely buy again",
                "Average experience, nothing special",
                "Outstanding quality, exceeded expectations",
                "Poor customer support, took too long to resolve",
            ],
            "ratings": [5, 3, 5, 3, 5, 2],
            "customers": [
                {"id": "cust_001", "segment": "premium", "lifetime_value": 2500},
                {"id": "cust_002", "segment": "standard", "lifetime_value": 800},
                {"id": "cust_003", "segment": "premium", "lifetime_value": 3200},
            ],
            "date_range": "March 2024",
            "channel": "online_survey",
        }

    @staticmethod
    def financial_data() -> Dict[str, Any]:
        """Financial data for analysis and reporting."""
        return {
            "financial_metrics": {
                "revenue": 125000,
                "expenses": 85000,
                "profit_margin": 0.32,
                "growth_rate": 0.15,
            },
            "monthly_data": [
                {"month": "Jan", "revenue": 40000, "expenses": 28000},
                {"month": "Feb", "revenue": 42000, "expenses": 29000},
                {"month": "Mar", "revenue": 43000, "expenses": 28000},
            ],
            "department_breakdown": {
                "engineering": {"headcount": 25, "budget": 350000},
                "sales": {"headcount": 15, "budget": 225000},
                "marketing": {"headcount": 8, "budget": 120000},
            },
        }

    @staticmethod
    def user_activity_data() -> Dict[str, Any]:
        """User activity data for behavioral analysis."""
        return {
            "user_sessions": [
                {
                    "user_id": "user_123",
                    "session_duration": 1800,
                    "pages_viewed": 12,
                    "actions": ["login", "browse", "search", "purchase"],
                    "timestamp": "2024-03-15T10:30:00Z",
                },
                {
                    "user_id": "user_456",
                    "session_duration": 600,
                    "pages_viewed": 4,
                    "actions": ["login", "browse", "logout"],
                    "timestamp": "2024-03-15T11:15:00Z",
                },
            ],
            "conversion_funnel": {
                "visitors": 10000,
                "signups": 1200,
                "purchases": 240,
                "conversion_rate": 0.024,
            },
            "cohort_data": {
                "week_1_retention": 0.85,
                "week_2_retention": 0.72,
                "week_4_retention": 0.58,
            },
        }

    @staticmethod
    def database_records() -> List[Dict[str, Any]]:
        """Sample database records for testing DB operations."""
        return [
            {
                "id": 1,
                "name": "John Doe",
                "email": "john@example.com",
                "status": "active",
                "created_at": "2024-01-15T08:00:00Z",
                "metadata": {"role": "admin", "permissions": ["read", "write"]},
            },
            {
                "id": 2,
                "name": "Jane Smith",
                "email": "jane@example.com",
                "status": "active",
                "created_at": "2024-02-01T09:30:00Z",
                "metadata": {"role": "user", "permissions": ["read"]},
            },
            {
                "id": 3,
                "name": "Bob Johnson",
                "email": "bob@example.com",
                "status": "inactive",
                "created_at": "2024-01-20T14:15:00Z",
                "metadata": {"role": "user", "permissions": ["read"]},
            },
        ]

    @staticmethod
    def api_response_data() -> Dict[str, Any]:
        """Sample API response data for REST testing."""
        return {
            "status": "success",
            "data": {
                "results": [
                    {"id": 1, "value": "first result"},
                    {"id": 2, "value": "second result"},
                ],
                "pagination": {
                    "page": 1,
                    "per_page": 10,
                    "total": 25,
                    "total_pages": 3,
                },
            },
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": "req_12345",
                "api_version": "v1",
            },
        }

    @staticmethod
    def complex_nested_data() -> Dict[str, Any]:
        """Complex nested data for advanced processing tests."""
        return {
            "organization": {
                "id": "org_001",
                "name": "Acme Corporation",
                "departments": [
                    {
                        "name": "Engineering",
                        "teams": [
                            {
                                "name": "Backend",
                                "members": [
                                    {
                                        "name": "Alice",
                                        "role": "Senior Engineer",
                                        "skills": ["Python", "PostgreSQL"],
                                    },
                                    {
                                        "name": "Bob",
                                        "role": "Engineer",
                                        "skills": ["JavaScript", "MongoDB"],
                                    },
                                ],
                                "projects": [
                                    {
                                        "name": "API Redesign",
                                        "status": "in_progress",
                                        "priority": "high",
                                    },
                                    {
                                        "name": "Database Migration",
                                        "status": "completed",
                                        "priority": "medium",
                                    },
                                ],
                            }
                        ],
                    }
                ],
                "metrics": {
                    "employee_satisfaction": 4.2,
                    "retention_rate": 0.92,
                    "productivity_score": 87,
                },
            }
        }

    @staticmethod
    def error_scenarios() -> Dict[str, Any]:
        """Data designed to trigger various error conditions."""
        return {
            "invalid_data": {
                "malformed_json": '{"incomplete": json}',
                "missing_required_field": {"name": "test"},  # missing "id"
                "invalid_type": {"id": "should_be_number", "value": 123},
                "null_values": {"id": 1, "name": None, "data": None},
            },
            "edge_cases": {
                "empty_string": "",
                "empty_list": [],
                "empty_dict": {},
                "very_long_string": "x" * 10000,
                "unicode_text": "Testing unicode:  émojis ànd spëcial chars ñ",
                "sql_injection_attempt": "'; DROP TABLE users; --",
            },
        }


# Test configuration presets
class TestConfigs:
    """Predefined test configurations for different scenarios."""

    @staticmethod
    def basic_agent_config() -> Dict[str, Any]:
        """Basic agent configuration for simple tests."""
        return {
            "name": "test_agent",
            "type": "standard",
            "enable_retry": False,
            "timeout": 30,
        }

    @staticmethod
    def retry_agent_config() -> Dict[str, Any]:
        """Agent configuration with retry enabled."""
        return {
            "name": "retry_test_agent",
            "type": "standard",
            "enable_retry": True,
            "retry_policy": {
                "max_retries": 3,
                "strategy": "exponential",
                "base_delay": 1.0,
            },
            "timeout": 60,
        }

    @staticmethod
    def reliability_workflow_config() -> Dict[str, Any]:
        """Workflow configuration with reliability features."""
        return {
            "name": "reliable_workflow",
            "reliability": {
                "acknowledgments": True,
                "task_tracking": True,
                "backpressure_control": True,
                "max_concurrent_tasks": 5,
                "max_queue_size": 100,
            },
        }

    @staticmethod
    def database_plugin_configs() -> Dict[str, Dict[str, Any]]:
        """Database plugin configurations for testing."""
        return {
            "postgresql": {
                "host": "localhost",
                "port": 5432,
                "database": "daita_test",
                "username": "test_user",
                "password": "test_password",
            },
            "mysql": {
                "host": "localhost",
                "port": 3306,
                "database": "daita_test",
                "username": "test_user",
                "password": "test_password",
            },
            "mongodb": {"host": "localhost", "port": 27017, "database": "daita_test"},
        }


# Helper functions for test data manipulation
def generate_large_dataset(size: int = 1000) -> List[Dict[str, Any]]:
    """Generate a large dataset for performance testing."""
    import random

    dataset = []
    for i in range(size):
        dataset.append(
            {
                "id": i,
                "value": random.randint(1, 1000),
                "category": random.choice(["A", "B", "C", "D"]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {"nested": {"value": random.random()}},
            }
        )

    return dataset


def create_test_file_content(format_type: str = "json") -> str:
    """Create test file content in various formats."""
    data = SampleData.sales_data()

    if format_type == "json":
        return json.dumps(data, indent=2)
    elif format_type == "csv":
        return "product,revenue,units\nProduct A,15000,150\nProduct B,22000,200\nProduct C,18000,180"
    elif format_type == "txt":
        return "Test file content\nLine 2\nLine 3"
    else:
        return str(data)
