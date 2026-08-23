from __future__ import annotations

import argparse
import json
import os
import signal
from dataclasses import asdict

import psycopg
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

from src.scoring import StreamingFraudScorer, Transaction

DDL = """
CREATE TABLE IF NOT EXISTS fraud_assessments (
    transaction_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    action TEXT NOT NULL,
    signals JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_fraud_assessments_queue
    ON fraud_assessments (risk_score DESC, created_at ASC)
    WHERE action IN ('block_and_review', 'step_up_and_review');
"""


class Worker:
    def __init__(self, bootstrap: str, topic: str, alert_topic: str, dsn: str) -> None:
        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap,
                "group.id": "fraud-scoring-v1",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        self.producer = Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True})
        self.topic = topic
        self.alert_topic = alert_topic
        self.dsn = dsn
        self.scorer = StreamingFraudScorer()
        self.running = True

    def stop(self, *_: object) -> None:
        self.running = False

    def process(self, conn: psycopg.Connection, payload: dict[str, object]) -> dict[str, object]:
        transaction = Transaction(
            transaction_id=str(payload["transaction_id"]),
            customer_id=str(payload["customer_id"]),
            event_time=str(payload["event_time"]),
            amount=float(payload["amount"]),
            merchant_category=str(payload.get("merchant_category", "unknown")),
            country_code=str(payload.get("country_code", "XX")),
        )
        assessment = self.scorer.score(transaction)
        output = assessment.to_dict()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fraud_assessments (
                    transaction_id, customer_id, event_time, risk_score,
                    confidence, action, signals
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                (
                    assessment.transaction_id,
                    assessment.customer_id,
                    assessment.event_time,
                    assessment.risk_score,
                    assessment.confidence,
                    assessment.action,
                    json.dumps([asdict(item) for item in assessment.signals]),
                ),
            )
        return output

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        with psycopg.connect(self.dsn) as conn:
            conn.execute(DDL)
            conn.commit()
            self.consumer.subscribe([self.topic])
            try:
                while self.running:
                    message = self.consumer.poll(1.0)
                    if message is None:
                        continue
                    if message.error():
                        if message.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        raise KafkaException(message.error())
                    try:
                        payload = json.loads(message.value())
                        assessment = self.process(conn, payload)
                        conn.commit()
                        if assessment["action"] != "allow":
                            self.producer.produce(
                                self.alert_topic,
                                key=str(assessment["customer_id"]),
                                value=json.dumps(assessment, separators=(",", ":")),
                            )
                            self.producer.poll(0)
                        # Offset is committed only after durable DB persistence.
                        self.consumer.commit(message=message, asynchronous=False)
                    except Exception:
                        conn.rollback()
                        raise
            finally:
                self.producer.flush(5)
                self.consumer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume transactions, score fraud risk and persist review cases.")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:19092"))
    parser.add_argument("--topic", default="transactions")
    parser.add_argument("--alert-topic", default="fraud-alerts")
    parser.add_argument("--dsn", default=os.getenv("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5433/fraud"))
    args = parser.parse_args()
    Worker(args.bootstrap, args.topic, args.alert_topic, args.dsn).run()


if __name__ == "__main__":
    main()
