# Generate Git Commit Message

Generate a commit message for the current changes following our team's **Git Commit Message Conventions**.

**Reference:** [Git Commit Message Conventions - Google Docs](https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y/edit?tab=t.0#heading=h.uyo6cb12dt6w)

## What you must do

1. **Gather context**
   - Use `@git` or run `git status` and `git diff --staged` (or `git diff` if nothing staged) to see what changed.
   - If the user pastes a diff or describes the change, use that too.

2. **Follow the convention**
   - Format: **type(scope): subject**
   - **type** (required): one of
     - `feat` – new feature
     - `fix` – bug fix
     - `docs` – documentation only
     - `style` – formatting, whitespace, no code logic change
     - `refactor` – code change that is not a fix nor a feature
     - `test` – adding or updating tests
     - `chore` – build, tooling, deps, config, etc.
   - **scope** (optional): short name of the area (e.g. `api`, `auth`, `docs`, `deps`).
   - **subject** (required):
     - Imperative mood, lowercase (e.g. "add X" not "added X" or "Adds X").
     - No period at the end.
     - Aim for ~50 characters; wrap in body if more is needed.

3. **Optional body and footer**
   - Add a **body** (blank line after subject) when you need to explain why or what in more detail.
   - Add a **footer** for:
     - `BREAKING CHANGE: description` when the change is breaking.
     - `Refs: TICKET-123` or `Fixes #123` when referencing issues.

4. **Output**
   - Output exactly **one** commit message block, ready to copy into `git commit -m "..."` or the multi-line commit message editor.
   - If the message is a single line, you can give it in quotes for easy copy-paste.
   - If it has body/footer, show the full message in a fenced block so the user can paste it into the editor.

## Example (single-line)

```
feat(docs): add Eino tutorial index and quickstart
```

## Example (with body)

```
fix(auth): prevent nil deref when session is missing

Check SessionExists before accessing Session in middleware.
Add unit test for unauthenticated request path.
```

## Example (with footer)

```
feat(api): change pagination default to 20 items

BREAKING CHANGE: default page size is now 20 instead of 10.
Refs: PROJ-456
```

Generate the commit message for the **current** changes only. If nothing is staged and no diff is provided, ask the user to stage changes or paste a diff.
