---
description: Analyze test coverage for a service across unit, integration, and contract tests
---

Analyze test and code coverage for the **{{args}}** service.

## Tasks

1. **Identify test structure**
   - Locate unit tests (tests/unit/)
   - Locate integration tests (tests/integration/)
   - Locate contract tests (tests/contract/)
   - Locate end to end tests (tests/e2e/)
   - Note any pytest markers used to distinguish test types

2. **Run code coverage analysis**
   - Run pytest with coverage for each test type separately (if possible via markers/paths)
   - Run combined coverage to get total percentage
   - Use `.venv/bin/pytest` from the project root

3. **Infer test coverage analysis**
   - Go beyond code coverage and think about user flows and edge cases that might be of value.
   - Use the project's README to understand the application's goal

4. **Present findings**
   Show a structured breakdown:
   - **Total code coverage**: X%
   - **By test type**:
     - Unit tests: X% (what they cover)
     - Integration tests: X% (what they cover)
     - Contract tests: X% (what they cover)
   - **Overal test coverage**: 
   - **Uncovered areas**: Critical paths or modules with no coverage
   - **Gaps**: Code covered by no test type, or only by one type, missing flows or scenarios

4. **Recommendations**
   - Suggest areas that need more coverage
   - Highlight if critical paths (API endpoints, business logic) lack coverage

## Notes
- Focus on the service specified in the argument
- If no service argument is provided, analyze the service in the current working directory
- Use pytest-cov for code coverage measurement
- Be concise but thorough
