import os

# Project root
project_root = "DeepAgentPrototype"

# Folder structure
folders = [
    "data",
    "agents",
    "ml",
    "backend",
    "frontend",
    "logs",
    "rules",
    "commands/@branch"
]

# .mdc files with enterprise-style content
mdc_files = {
    "rules/writing.mdc": """# Writing Rules - Enterprise Prototype (FAANG-style)

1. Modular Design:
   - Each agent and ML module must be fully modular and independently testable.
   - Maintain clear input/output contracts for every function.

2. Readability & Documentation:
   - Use descriptive variable names, consistent formatting, and comments.
   - Document all assumptions and data transformations.

3. Version Control:
   - One feature per branch; merge only after passing tests and review.
   - Maintain changelog and architecture diagram updates.

4. Data Handling:
   - Validate all inputs; handle empty/loading/error/offline states.
   - Use synthetic data for prototyping; clearly label mocks vs real data.
""",

    "rules/use-bun.mdc": """# Usage Rules - Enterprise Prototype (FAANG-style)

1. API Keys & External Services:
   - Store OpenAI and other API keys securely in environment variables.
   - Limit API calls to control latency and cost during prototype.

2. Dependencies:
   - Only approved libraries allowed; remove unused packages.
   - Prefer lightweight, stable packages for prototyping.

3. Caching & Performance:
   - Implement caching for repeated ML predictions or agent outputs.
   - Reduce unnecessary API calls to improve latency.

4. Data Storage:
   - MongoDB for unstructured prototype data.
   - Preprocessed data can be cached in memory for fast iteration.

5. Fallbacks:
   - Provide default/simulated data if external services fail.
""",

    "rules/ultracite.mdc": """# Governance Rules - Enterprise Prototype (FAANG-style)

1. Error States:
   - Handle empty/loading/error/offline states in frontend and backend.
   - Log all errors with timestamps and context.

2. API Compatibility:
   - Ensure backward compatibility; increment version for breaking changes.

3. Testing:
   - High-quality unit and integration tests required for every feature.
   - Validate ML and agent outputs against synthetic data.

4. Security & Permissions:
   - Review auth flows and permissions on backend changes.
   - Confirm no sensitive data exposure.

5. Feature Flags:
   - Use feature flags for experimental features.
   - Check caching and logging for feature-flagged modules.

6. Logging & Monitoring:
   - Log critical steps of agents, ML predictions, and decisions.
   - Include feature/branch metadata for traceability.

7. One Feature at a Time:
   - Implement and fully validate one feature per branch before moving to the next.
""",

    "commands/@branch/feature-template.mdc": """# Feature Implementation Checklist - Enterprise Prototype (FAANG-style)

## Branch Name: <feature-name>
## Developer: <name>
## Date: <YYYY-MM-DD>

1. Data Flow:
   - Ensure agents read and process synthetic/real data correctly.
   - Handle empty/loading/error/offline states.

2. Frontend Review:
   - Review Ally changes; ensure components reflect backend outputs.
   - Validate user interactions.

3. API Validation:
   - Confirm API endpoints are backward compatible.
   - Increment version if breaking changes.

4. Dependency Cleanup:
   - Remove unused packages or modules.

5. Testing:
   - Run unit and integration tests.
   - Verify ML predictions, agent decisions, and edge cases.

6. Security & Auth:
   - Run security review if permissions/auth flows change.
   - Ensure no sensitive data exposure.

7. Feature Flags & Caching:
   - Implement feature flags if needed.
   - Verify caching for performance optimization.

8. Logging:
   - Log critical steps in backend, ML, and agent workflows.
   - Include timestamp and branch metadata.

9. Final Review:
   - Confirm all steps completed before merge.
   - Only merge after passing tests and review.

10. Notes:
   - Document deviations, observations, or pending items.
"""
}

# Create folders
for folder in folders:
    folder_path = os.path.join(project_root, folder)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created folder: {folder_path}")

# Create .mdc files
for path, content in mdc_files.items():
    file_path = os.path.join(project_root, path)
    with open(file_path, "w") as f:
        f.write(content)
    print(f"Created file: {file_path}")

print("\n✅ DeepAgentPrototype structure with FAANG-style .mdc files created!")