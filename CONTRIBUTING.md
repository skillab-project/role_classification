# Contributing to Role Classification

Thank you for your interest in contributing! This document outlines the process for reporting issues, proposing changes, and submitting code to the project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Submitting Code Changes](#submitting-code-changes)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project is part of the [SkillLab](https://github.com/skillab-project) research initiative. All contributors are expected to engage respectfully and constructively. Harassment, discrimination, or disruptive behaviour of any kind will not be tolerated.

---

## Getting Started

Before contributing, please:

1. Read the [README](README.md) to understand the project's purpose and architecture.
2. Check the [open issues](https://github.com/skillab-project/role_classification/issues) to see if your bug or feature has already been reported.
3. For significant changes, open an issue first to discuss your proposed approach before writing code.

---

## How to Contribute

### Reporting Bugs

If you encounter a bug, please open an issue and include:

- A clear, descriptive title.
- Steps to reproduce the problem.
- Expected vs. actual behaviour.
- Relevant environment details (OS, Python version, Docker version if applicable).
- Any error messages or stack traces.

### Suggesting Enhancements

Feature requests are welcome. When opening an enhancement issue, please describe:

- The problem you are trying to solve or the use case you have in mind.
- Your proposed solution or approach.
- Any alternatives you considered.

### Submitting Code Changes

All code contributions are made via **Pull Requests (PRs)**. The general workflow is:

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make your changes.
4. Run the test suite and ensure all tests pass.
5. Open a PR against `main`.

---

## Development Setup

1. **Fork and clone your fork:**

   ```bash
   git clone https://github.com/<your-username>/role_classification.git
   cd role_classification
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install all dependencies (including test dependencies):**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your `.env` file:**

   ```env
   TRACKER_API=https://skillab-tracker.csd.auth.gr/api
   TRACKER_USERNAME=your_username
   TRACKER_PASSWORD=your_password
   ```

5. **Start the development server:**

   ```bash
   uvicorn role_classification:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) for all Python code.
- Use descriptive variable and function names.
- Keep functions focused on a single responsibility.
- Add docstrings to any new functions or classes.
- Do not commit credentials, secrets, or any sensitive data. Use environment variables via `.env`.
- Avoid adding large binary files or model artefacts to the repository.

---

## Testing

The test suite lives in the `tests/` directory. Run it with:

```bash
pytest tests/
```

When contributing:

- Add or update tests to cover any new behaviour you introduce.
- Ensure all existing tests continue to pass before opening a PR.
- For endpoint changes, include tests that exercise the relevant query parameters and response structure.

---

## Commit Message Guidelines

Use clear, imperative-mood commit messages. A good format is:

```
<type>: <short summary>

<optional longer description>
```

Common types:

| Type       | When to use                                      |
|------------|--------------------------------------------------|
| `feat`     | A new feature or endpoint                        |
| `fix`      | A bug fix                                        |
| `refactor` | Code restructuring without behaviour change      |
| `test`     | Adding or updating tests                         |
| `docs`     | Documentation changes only                      |
| `chore`    | Dependency updates, CI config, tooling changes   |

Examples:

```
feat: add naive_bayes model_type option to classifier endpoint
fix: remove stale lock file when streaming fails mid-response
docs: document radar_profile fields in README
```

---

## Pull Request Process

1. **Branch naming:** Use a descriptive name, e.g. `feat/add-svm-classifier` or `fix/stale-lock-cleanup`.
2. **Keep PRs focused:** One logical change per PR. Avoid bundling unrelated fixes together.
3. **Fill in the PR description:** Explain what the change does, why it is needed, and how it was tested.
4. **Link related issues:** Reference any relevant issue with `Closes #<issue-number>` in the PR description.
5. **Review:** At least one maintainer review is required before merging. Be responsive to feedback and update your branch as needed.
6. **CI:** Ensure the automated checks pass. PRs with failing tests will not be merged.

---

## Questions

If you have questions that are not covered here, feel free to open a [discussion](https://github.com/skillab-project/role_classification/issues) or reach out via the issue tracker.
