# Kafka

## Consumer Groups

Kafka consumer groups let multiple consumers cooperate to read a topic.
Each partition is assigned to exactly one consumer within the group, which is how
Kafka parallelizes consumption across many machines.

## Rebalancing

When a consumer joins or leaves, Kafka triggers a rebalance: partitions are
reassigned across the surviving consumers. Frequent rebalances hurt throughput,
so tuning session.timeout.ms and max.poll.interval.ms matters in production.

## Offsets

Offsets track how far a group has read. They are committed back to Kafka, either
automatically or manually. To reprocess data, reset offsets with the
kafka-consumer-groups tool, choosing --to-earliest, --to-latest, or a timestamp.

```bash
# reset a group to the beginning of the topic
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group my-group --topic events --reset-offsets --to-earliest --execute
```
