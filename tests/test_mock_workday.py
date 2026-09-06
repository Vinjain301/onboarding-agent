import pytest

from mock_workday import MockWorkdayClient
from mock_workday.api import EmployeeNotFoundError


def test_get_employee_returns_expected_fields():
    client = MockWorkdayClient()
    employee = client.get_employee("E1001")
    assert employee["full_name"] == "Priya Nathan"
    assert employee["role"] == "Software Engineer"
    assert employee["department"] == "Engineering"


def test_get_employee_unknown_id_raises():
    client = MockWorkdayClient()
    with pytest.raises(EmployeeNotFoundError):
        client.get_employee("E9999")


def test_onboarding_context_includes_milestones_and_role_profile():
    client = MockWorkdayClient()
    context = client.get_onboarding_context("E1002")
    assert context["role"] == "Data Analyst"
    assert set(context["milestones"].keys()) == {"day_30", "day_60", "day_90"}
    assert "Snowflake" in context["role_profile"]["default_systems"]
    assert context["onboarding_phase"] in {
        "pre-boarding",
        "day_1_30",
        "day_31_60",
        "day_61_90",
        "post_90",
    }


def test_find_employee_by_name_case_insensitive():
    client = MockWorkdayClient()
    employee = client.find_employee_by_name("priya nathan")
    assert employee is not None
    assert employee["employee_id"] == "E1001"


def test_find_employee_by_name_no_match_returns_none():
    client = MockWorkdayClient()
    assert client.find_employee_by_name("Nobody Here") is None
