from alembic import context
from job_backend.models import Base

connection = context.config.attributes.get("connection")
if connection is None:
    raise RuntimeError("use backend_cli.py migrate with an explicit DATABASE_URL")
context.configure(connection=connection, target_metadata=Base.metadata)
with context.begin_transaction():
    context.run_migrations()
