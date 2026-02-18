# How to Use This Documentation for AI

This guide explains how AI assistants should consume and utilize this Project Knowledge Pack.

## The Three-Round Learning Process

### Round 1: Build the Mental Model

**Objective**: Establish a high-level understanding of the system.

**Documents to read**:
- [Project Overview](00-overview.md)
- [Repository Map](01-repo-map.md)
- [Architecture](02-architecture.md)

**Expected output**:
1. A one-page summary of:
   - System components and their relationships
   - Key entry points (main files, routers, services)
   - Critical data flows
2. A prioritized list of 10 questions about missing information
3. A small "first hands-on task" proposal with tests

**Validation**:
- Can you describe the system to a new developer?
- Can you locate the code for a typical user request?
- Do you understand the deployment topology?

### Round 2: Deep Dive into Business Logic

**Objective**: Master 2-3 critical business workflows.

**Documents to read**:
- [Workflows](03-workflows.md) - Pick 2-3 key workflows
- [Data and API](04-data-and-api.md) - Related data models
- Relevant code files identified in workflows

**Expected output** for each workflow:
1. A call chain diagram showing:
   - Function/module boundaries
   - Transaction boundaries
   - Idempotency points
   - Retry mechanisms
2. Five boundary conditions most likely to cause bugs
3. A comprehensive test checklist:
   - Unit tests: What to test
   - Integration tests: What to test
   - E2E tests: What to test

**Validation**:
- Can you trace a user request end-to-end?
- Can you identify edge cases and error handling?
- Can you propose meaningful test cases?

### Round 3: Module-Level Mastery

**Objective**: Gain deep understanding of individual modules to make production-ready contributions.

**Documents to read** for each module:
- Module README or documentation
- Module tests
- Sample data or fixtures

**Expected output** for each module:
1. A summary of:
   - Module responsibilities
   - Public APIs
   - Dependencies and coupling
2. Three improvement suggestions
3. A small PR-level change with tests:
   - The change should be meaningful but contained
   - Must include appropriate tests
   - Should follow the project's conventions

**Validation**:
- Can you make a change without breaking tests?
- Do you understand the module's role in the system?
- Can you identify areas for improvement?

## Key Capabilities to Develop

### 1. Locate 🗺️

**What**: Quickly find relevant code, configs, and scripts.

**How**:
- Study the [Repository Map](01-repo-map.md)
- Understand naming conventions
- Know the entry points
- Recognize common patterns

**Test yourself**:
- Where is the authentication logic?
- Where are database queries made?
- Where are API routes defined?

### 2. Understand 🧠

**What**: Comprehend architecture, dependencies, and business logic.

**How**:
- Read [Architecture](02-architecture.md) for system design
- Read [Workflows](03-workflows.md) for business logic
- Review [ADRs](adr/index.md) for design rationale

**Test yourself**:
- Why was this architectural choice made?
- What are the key trade-offs?
- What happens if a dependency fails?

### 3. Execute ⚙️

**What**: Set up environment, run tests, modify code, create PRs.

**How**:
- Follow the [Runbook](06-runbook.md)
- Understand [Conventions](05-conventions.md)
- Learn the [Testing Strategy](07-testing.md)

**Test yourself**:
- Can you start the development environment?
- Can you run all tests?
- Can you make a change and verify it works?

### 4. Verify ✅

**What**: Design tests, locate bugs, perform regression testing.

**How**:
- Study [Testing Strategy](07-testing.md)
- Understand test pyramid (unit/integration/e2e)
- Learn debugging approaches from the Runbook

**Test yourself**:
- Can you write meaningful tests?
- Can you debug a failing test?
- Can you identify what else needs testing?

## AI-Specific Guidelines

### When Reading Code

```{admonition} Code References
:class: tip

Pay attention to code references in the format `file.py:123`:
- These point to specific lines of code
- Always read the referenced code
- Understand the context around the reference
```

### When Making Changes

```{admonition} Change Checklist
:class: important

Before proposing a change:
- [ ] Have you read the relevant documentation?
- [ ] Have you reviewed related [ADRs](adr/index.md)?
- [ ] Does your change follow [Conventions](05-conventions.md)?
- [ ] Have you considered the impact on existing workflows?
- [ ] Can you write tests for your change?
- [ ] Have you identified rollback steps?
```

### When Answering Questions

```{admonition} Answer Quality
:class: tip

Good answers should:
- Reference specific documentation sections
- Include code file paths
- Consider edge cases
- Acknowledge limitations or assumptions
- Suggest validation steps
```

## Common Pitfalls to Avoid

### ❌ Don't

- Assume you understand without reading the docs
- Propose changes without understanding the architecture
- Ignore conventions and patterns
- Make changes without tests
- Forget about error handling and edge cases

### ✅ Do

- Ask for clarification when documentation is unclear
- Validate your understanding with specific questions
- Propose small, testable changes
- Consider the impact on other components
- Reference ADRs when making architectural suggestions

## Example Workflows

### Answering "How does authentication work?"

1. Read [Architecture](02-architecture.md) - Find the auth component
2. Read [Workflows](03-workflows.md) - Find the login workflow
3. Check [ADRs](adr/index.md) - Understand auth decisions
4. Look at code references - Read actual implementation
5. Check [Runbook](06-runbook.md) - Understand auth setup

**Answer should include**:
- Authentication mechanism (JWT, OAuth, etc.)
- Where auth is enforced (middleware, decorators, etc.)
- How tokens are stored and validated
- What happens on auth failure
- Code references: `auth/middleware.py:45`

### Implementing a New Feature

1. Create a [Change Proposal](changes/index.md)
2. Design the solution referencing [Architecture](02-architecture.md)
3. Identify affected [Workflows](03-workflows.md)
4. Follow [Conventions](05-conventions.md)
5. Plan tests per [Testing Strategy](07-testing.md)
6. Update relevant documentation

### Debugging an Issue

1. Check [Runbook](06-runbook.md) for common issues
2. Review relevant [Workflows](03-workflows.md)
3. Check [Conventions](05-conventions.md) for logging standards
4. Identify affected components in [Architecture](02-architecture.md)
5. Propose a fix with tests

## Feedback Loop

```{admonition} Continuous Improvement
:class: note

As you use this documentation:
- Report gaps or unclear sections
- Suggest additional examples
- Propose documentation improvements
- Help keep documentation up to date
```

## Next Steps

1. Start with [Round 1](#round-1-build-the-mental-model)
2. Complete the expected outputs
3. Validate your understanding
4. Move to [Round 2](#round-2-deep-dive-into-business-logic)
5. Eventually reach [Round 3](#round-3-module-level-mastery)

---

**Remember**: The goal is not to memorize everything, but to know where to find information and how to use it effectively.
