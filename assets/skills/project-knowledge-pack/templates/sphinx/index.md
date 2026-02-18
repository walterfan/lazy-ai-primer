# Project Knowledge Pack

Welcome to the project documentation! This documentation is designed to help both humans and AI understand and contribute to this project.

## Overview

This documentation follows the **Project Knowledge Pack** methodology, providing structured information about the project's architecture, workflows, and development practices.

```{admonition} Purpose
:class: tip

This knowledge pack enables developers and AI assistants to:
- **Locate**: Quickly find code, configs, and scripts
- **Understand**: Grasp architecture, dependencies, and business logic
- **Execute**: Set up environment, run tests, and modify code
- **Verify**: Design tests, locate bugs, and perform regression testing
```

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: Project Documentation

00-overview
01-repo-map
02-architecture
03-workflows
04-data-and-api
05-conventions
06-runbook
07-testing
```

```{toctree}
:maxdepth: 1
:caption: Architecture Decisions

adr/index
```

```{toctree}
:maxdepth: 1
:caption: Change Proposals

changes/index
```

## Quick Start

### For New Developers

1. Read the [Project Overview](00-overview.md) to understand the project's purpose and scope
2. Study the [Repository Map](01-repo-map.md) to navigate the codebase
3. Review the [Runbook](06-runbook.md) to set up your development environment
4. Check the [Conventions](05-conventions.md) to understand coding standards

### For AI Assistants

This documentation is structured to support AI-assisted development:

1. **Round 1**: Read Overview, Repo Map, and Architecture to build a mental model
2. **Round 2**: Study 2-3 key Workflows to understand business logic
3. **Round 3**: Review module-specific documentation and tests for hands-on contributions

See [How to Use This Documentation for AI](ai-guide.md) for detailed guidance.

## Key Concepts

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🗺️ Navigation
:link: 01-repo-map
:link-type: doc

Understand the project structure, find entry points, and locate key components.
:::

:::{grid-item-card} 🏗️ Architecture
:link: 02-architecture
:link-type: doc

Learn about the system design, component interactions, and key patterns.
:::

:::{grid-item-card} ⚙️ Workflows
:link: 03-workflows
:link-type: doc

Explore business processes, data flows, and integration points.
:::

:::{grid-item-card} 🧪 Testing
:link: 07-testing
:link-type: doc

Discover the testing strategy, coverage goals, and how to run tests.
:::

::::

## Documentation Maintenance

```{admonition} Living Documentation
:class: important

This documentation should evolve with the codebase:
- Update docs in the same PR as code changes
- Create ADRs for important architectural decisions
- Document new workflows and conventions
- Keep runbooks up to date with operational changes
```

## Contributing

See [Conventions](05-conventions.md) for coding standards and contribution guidelines.

## Indices and tables

- {ref}`genindex`
- {ref}`search`

---

**Version**: {sub-ref}`release`
**Last Updated**: {sub-ref}`today`
