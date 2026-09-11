"""Product controls exercise the same public authority and evidence boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

import pytest
from textual.widgets import Input, OptionList, Static

from daita import (
    Agent,
    ApprovalDecision,
    EffectResolutionDecision,
    OnceSchedule,
    cli,
)
from daita._json import FrozenJsonObject
from daita.artifacts.models import ArtifactAuthorship
from daita.distribution import ArtifactRequirement, OutcomeState
from daita.llm.models import ModelSensitivity, ToolCall
from daita.tui.app import DaitaApp
from daita.tui.projection import approval_review_document, effect_receipt_mapping
from daita.tui.screens.effects import EffectsScreen
from daita.tui.screens.permissions import PermissionsScreen
from daita.tui.screens.routines import render_routine_inspection
from daita.tui.screens.selection import SelectionScreen
from daita.tui.widgets.approval import ApprovalPanel
from tests.support.mcp_actions import ActionFixture, response
from tests.support.native_writes import create_fixture
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.acceptance


def choose(app, identities, *, multi=False):
    screen = app.screen
    assert isinstance(screen, SelectionScreen)
    listing = screen.query_one("#picker-options", OptionList)
    for identity in identities:
        listing.highlighted = next(
            i
            for i in range(listing.option_count)
            if listing.get_option_at_index(i).id == identity
        )
        if multi:
            screen.action_toggle_selected()
    screen.action_confirm()
