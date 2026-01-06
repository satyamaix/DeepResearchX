# Phoenix Tracing Fix Plan

## Status: COMPLETED ✅

Traces are now successfully being sent to Phoenix and the `drx-research` project is visible in the dashboard.

## Root Cause Analysis

### Problem 1: SSL/TLS Handshake Failure
**Error**: `SSL_ERROR_SSL: error:100000f7:SSL routines:OPENSSL_internal:WRONG_VERSION_NUMBER`

**Cause**: The OTLP gRPC exporter is attempting TLS/SSL connection but Phoenix container is serving plain gRPC without TLS. The `phoenix.otel.register()` function infers protocol from endpoint and defaults to secure gRPC for port 4317.

### Problem 2: No Project in Dashboard
**Cause**: Since traces fail to export (SSL error), no data reaches Phoenix, thus no project is created.

## Solution Strategy

### Fix Approach
Use HTTP/protobuf protocol instead of gRPC for internal Docker communication:
- Change endpoint from `phoenix:4317` to `http://phoenix:6006/v1/traces`
- Explicitly set `protocol="http/protobuf"` in register() call
- HTTP transport doesn't have the TLS inference issue

## Work Packets

### WP-A: Update Phoenix Configuration Code
**Files**: `src/observability/phoenix.py`, `src/config.py`
**Changes**:
1. Update `setup_phoenix()` to use http/protobuf protocol
2. Change endpoint to use HTTP format
3. Add protocol configuration option

### WP-B: Update Docker Compose Environment
**Files**: `deployment/docker-compose.yaml`, `.env.example`
**Changes**:
1. Update `PHOENIX_COLLECTOR_ENDPOINT` to HTTP format
2. Add `PHOENIX_PROTOCOL` environment variable

### WP-C: Verification and Testing
**Tasks**:
1. Restart Docker stack
2. Verify traces appear in Phoenix
3. Confirm project name is set correctly

---

## Verification Results

### API Logs (After Fix)
```
🔭 OpenTelemetry Tracing Details 🔭
|  Phoenix Project: drx-research
|  Span Processor: BatchSpanProcessor
|  Collector Endpoint: http://phoenix:6006/v1/traces
|  Transport: HTTP + protobuf
```

### Phoenix Projects (GraphQL Query)
```json
{
  "data": {
    "projects": {
      "edges": [
        {"node": {"name": "drx-research", "traceCount": 2}},
        {"node": {"name": "default", "traceCount": 0}}
      ]
    }
  }
}
```

### Commits
- `0d1eb8c` - fix(phoenix): Use HTTP/protobuf protocol to fix SSL errors
- `c8af339` - fix(docker): Update Phoenix endpoint to use HTTP/protobuf
