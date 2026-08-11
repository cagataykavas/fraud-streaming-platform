from __future__ import annotations

from pyspark.sql import SparkSession, functions as F, types as T

SCHEMA = T.StructType(
    [
        T.StructField("transaction_id", T.StringType(), False),
        T.StructField("customer_id", T.StringType(), False),
        T.StructField("event_time", T.TimestampType(), False),
        T.StructField("amount", T.DoubleType(), False),
        T.StructField("merchant_category", T.StringType(), True),
        T.StructField("country_code", T.StringType(), True),
    ]
)


def build_stream(kafka_bootstrap: str, topic: str):
    spark = SparkSession.builder.appName("fraud-streaming-platform").getOrCreate()
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        raw.select(F.from_json(F.col("value").cast("string"), SCHEMA).alias("tx"))
        .select("tx.*")
        .withWatermark("event_time", "10 minutes")
    )

    velocity_5m = (
        parsed.groupBy(F.window("event_time", "5 minutes", "1 minute"), "customer_id")
        .agg(
            F.count("transaction_id").alias("txn_count_5m"),
            F.sum("amount").alias("amount_sum_5m"),
            F.max("amount").alias("max_amount_5m"),
        )
    )

    return spark, parsed, velocity_5m


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="transactions")
    args = parser.parse_args()

    spark, _, features = build_stream(args.bootstrap, args.topic)
    query = features.writeStream.format("console").outputMode("update").start()
    query.awaitTermination()
    spark.stop()
