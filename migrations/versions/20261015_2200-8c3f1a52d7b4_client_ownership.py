"""client ownership

Conversations get an owner and usage rows get the client that spent. Both stay nullable:
rows from before authentication existed have no owner, and inventing one would be wrong.

Revision ID: 8c3f1a52d7b4
Revises: 1967f795ee07
Create Date: 2026-10-15 22:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8c3f1a52d7b4"
down_revision: str | Sequence[str] | None = "1967f795ee07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("conversations", sa.Column("owner", sa.String(length=64), nullable=True))
    op.create_index(op.f("ix_conversations_owner"), "conversations", ["owner"], unique=False)
    op.add_column("usage_logs", sa.Column("client", sa.String(length=64), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "client")
    op.drop_index(op.f("ix_conversations_owner"), table_name="conversations")
    op.drop_column("conversations", "owner")
