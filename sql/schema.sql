CREATE TABLE IF NOT EXISTS fraud_decisions (
  transaction_id TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  model_version TEXT NOT NULL,
  fraud_probability DOUBLE PRECISION NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
  decision TEXT NOT NULL CHECK (decision IN ('allow', 'review', 'block')),
  feature_snapshot JSONB NOT NULL,
  reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_fraud_customer_event_time
  ON fraud_decisions(customer_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_fraud_decision_time
  ON fraud_decisions(decision, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_fraud_features_gin
  ON fraud_decisions USING GIN(feature_snapshot);
