"""A mock Workday-style HCM API.

Simulates the subset of endpoints an onboarding agent would call against a
real Workday tenant: employee profile lookup, role/job-profile metadata, and
derived onboarding milestones. Swapping this module for a real Workday
integration (RaaS report or Workday REST API) should not require changes to
the RAG pipeline or agent layers, which only depend on the dict shapes
returned here.
"""

from datetime import datetime, timedelta

from .mock_data import EMPLOYEES, ROLE_PROFILES


class EmployeeNotFoundError(Exception):
    pass


class MockWorkdayClient:
    """Read-only client over the mock employee directory."""

    def __init__(self):
        self._employees_by_id = {e["employee_id"]: e for e in EMPLOYEES}

    def list_employees(self) -> list[dict]:
        return list(EMPLOYEES)

    def get_employee(self, employee_id: str) -> dict:
        employee = self._employees_by_id.get(employee_id)
        if employee is None:
            raise EmployeeNotFoundError(f"No employee found with id '{employee_id}'")
        return dict(employee)

    def find_employee_by_name(self, full_name: str) -> dict | None:
        needle = full_name.strip().lower()
        for employee in EMPLOYEES:
            if employee["full_name"].lower() == needle:
                return dict(employee)
        return None

    def get_role_profile(self, role: str) -> dict:
        profile = ROLE_PROFILES.get(role)
        if profile is None:
            return {
                "department": "Unknown",
                "default_systems": [],
                "required_training_days": 14,
            }
        return dict(profile)

    def get_onboarding_context(self, employee_id: str) -> dict:
        """Everything the agent needs to personalize guidance for one employee."""
        employee = self.get_employee(employee_id)
        role_profile = self.get_role_profile(employee["role"])
        start_date = datetime.strptime(employee["start_date"], "%Y-%m-%d")

        milestones = {
            "day_30": (start_date + timedelta(days=30)).strftime("%Y-%m-%d"),
            "day_60": (start_date + timedelta(days=60)).strftime("%Y-%m-%d"),
            "day_90": (start_date + timedelta(days=90)).strftime("%Y-%m-%d"),
        }

        today = datetime.today()
        days_since_start = (today - start_date).days

        return {
            **employee,
            "role_profile": role_profile,
            "milestones": milestones,
            "days_since_start": days_since_start,
            "onboarding_phase": _phase_for(days_since_start),
        }


def _phase_for(days_since_start: int) -> str:
    if days_since_start < 0:
        return "pre-boarding"
    if days_since_start <= 30:
        return "day_1_30"
    if days_since_start <= 60:
        return "day_31_60"
    if days_since_start <= 90:
        return "day_61_90"
    return "post_90"
