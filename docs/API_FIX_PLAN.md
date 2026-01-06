# API Request/Response Fix Plan

## Status: COMPLETED ✅

Both `input` and `query` fields now work for research requests.

## Issues Identified

### Issue 1: Wrong Request Field Name (User Error)
The curl request uses `query` but the API expects `input`:

**Wrong:**
```json
{"query": "...", "config": {...}}
```

**Correct:**
```json
{"input": "...", "config": {...}}
```

### Issue 2: Misleading Error Messages (Code Issue)
When validation fails, the error response says "Database connection unavailable" instead of showing the actual validation error. This happens because:

1. The `except Exception` blocks in `dependencies.py` are too broad
2. They catch HTTPException/ValidationError from other sources
3. The error gets re-wrapped with misleading messages

## Fix Strategy

### WP-A: Fix Exception Handling in Dependencies
**Files**: `src/api/dependencies.py`
**Changes**:
1. Import specific exception types to exclude from catch
2. Re-raise HTTPException without wrapping
3. Only catch actual connection errors

### WP-B: Add Input Alias for Backward Compatibility
**Files**: `src/api/routes.py`
**Changes**:
1. Add `query` as an alias for `input` field in ResearchRequest
2. Use Pydantic's `Field(alias=...)` or custom validator
3. This allows both `input` and `query` to work

### WP-C: Update API Documentation
**Files**: Update examples in README or API docs

## Work Packet Dependencies

```
WP-A ─────┐
          ├──► WP-C (Documentation)
WP-B ─────┘
```

WP-A and WP-B can run in parallel.

---

## Verification Results

### Test with `query` (now works):
```bash
curl -X POST http://localhost:8000/api/v1/interactions \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformers?", "config": {"max_iterations": 2}}'
```
Response: `{"id":"...", "status":"queued", ...}`

### Test with `input` (canonical):
```bash
curl -X POST http://localhost:8000/api/v1/interactions \
  -H "Content-Type: application/json" \
  -d '{"input": "What are transformers?", "config": {"max_iterations": 2}}'
```
Response: `{"id":"...", "status":"queued", ...}`

### Commits
- `8aaf9b3` - fix(api): Improve exception handling in dependencies
- `684921e` - feat(api): Accept 'query' as alias for 'input'
