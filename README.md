# Fraud Streaming Platform

A real-time fraud-scoring reference architecture built around event-time processing, online features, model scoring and analytical storage.

## Target architecture

```text
transaction producers
        |
        v
      Kafka
        |
        v
Spark Structured Streaming
  |     |       |
  |     |       +--> data quality / dead-letter stream
  |     +----------> rolling velocity features
  +----------------> model scoring
        |
        v
PostgreSQL / ClickHouse
        |
        +--> alert API
        +--> Grafana / monitoring
```

## Engineering themes

- event-time vs processing-time semantics
- watermarking and late-arriving events
- idempotent writes
- sliding-window velocity features
- schema validation and data contracts
- dead-letter handling
- model-version tagging
- auditable decisions
- backpressure-aware streaming design

The public examples use synthetic transactions only.
