# run_ingest.py
import db
from ingest import ingest

conn = db.get_conn()
db.setup_schema(conn)          # DROP + CREATE table (пустая)
ingest(
  conn,
  "/Users/aliakseichakhlou/aiengineer/website/content/en/docs/concepts/workloads",
  source="k8s",
)
