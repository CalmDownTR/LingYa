# LingYa Release Checklist

## Pre-release

- [ ] All ROADMAP deliverables for this version are done or explicitly deferred
- [ ] `uv run pytest` passes
- [ ] `uv run ruff check lingya/` passes
- [ ] `pyproject.toml` version matches the target release
- [ ] `product/ROADMAP.md` "已完成版本" section updated
- [ ] `.claude/specs/architecture.md` synced with code changes
- [ ] No data loss: upgrade from previous version's DB works

## Version Number Rules

- **0.x.0** (minor): New ROADMAP theme or new user-facing capability
- **0.x.y** (patch): Bug fix, small enhancement within a theme, sub-deliverable
- **1.0.0** (major): API stability commitment — not yet

### Special: v0.x allows breaking changes

In 0.x, minor versions may contain breaking interface changes.
Document them explicitly in the release notes.

## Release Steps

```bash
# 1. Ensure main is clean and up-to-date
git checkout main && git pull

# 2. Update version in pyproject.toml (if not already done)

# 3. Run full validation
uv run pytest && uv run ruff check lingya/

# 4. Commit version bump
git add -A && git commit -m "chore: bump version to v0.x.y"

# 5. Create annotated tag
git tag -a v0.x.y -m "v0.x.y — <theme summary in Chinese>"

# 6. Push code + tags
git push origin main --follow-tags

# 7. (Optional) GitHub Release with notes
gh release create v0.x.y --title "v0.x.y — <theme>" --notes "<brief description>"
```

## Post-release

- [ ] Verify `git tag -l` shows the new tag
- [ ] Verify `pyproject.toml` version matches
- [ ] Update next version's ROADMAP section with "📋 下一个" marker
