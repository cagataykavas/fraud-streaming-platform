from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timedelta, timezone

from confluent_kafka import Producer


def event(index: int, rng: random.Random, current: datetime) -> dict[str, object]:
    customer_id = f"cust-{rng.randint(1, 250):05d}"
    category = rng.choices(
        ["grocery", "travel", "electronics", "restaurant", "crypto", "cash_equivalent", "gambling"],
        weights=[35, 10, 15, 20, 3, 5, 2],
        k=1,
    )[0]
    base = 120 if category in {"grocery", "restaurant"} else 650
    amount = max(1.0, rng.lognormvariate(0.0, 1.0) * base)
    if rng.random() < 0.015:
        amount *= rng.uniform(8, 25)
    return {
        "transaction_id": f"tx-{index:010d}",
        "customer_id": customer_id,
        "event_time": current.isoformat().replace("+00:00", "Z"),
        "amount": round(amount, 2),
        "merchant_category": category,
        "country_code": rng.choices(["TR", "DE", "NL", "GB", "US", "JP"], weights=[65, 8, 5, 7, 10, 5], k=1)[0],
    }


def produce(bootstrap: str, topic: str, count: int, seed: int, sleep_ms: float) -> None:
    producer = Producer({"bootstrap.servers": bootstrap, "client.id": "fraud-demo-producer", "enable.idempotence": True})
    rng = random.Random(seed)
    current = datetime.now(timezone.utc)
    for index in range(count):
        current += timedelta(milliseconds=rng.randint(50, 1800))
        payload = event(index, rng, current)
        producer.produce(
            topic,
            key=str(payload["customer_id"]),
            value=json.dumps(payload, separators=(",", ":")),
        )
        producer.poll(0)
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000)
    producer.flush(10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish synthetic transactions to Kafka/Redpanda.")
    parser.add_argument("--bootstrap", default="localhost:19092")
    parser.add_argument("--topic", default="transactions")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep-ms", type=float, default=0.0)
    args = parser.parse_args()
    produce(args.bootstrap, args.topic, args.count, args.seed, args.sleep_ms)


if __name__ == "__main__":
    main()
