"""Declarative base and naming conventions.

A single explicit naming convention means every constraint and index has a
predictable name in the database, which is what lets Alembic generate reversible
migrations instead of emitting anonymous ``DROP CONSTRAINT`` statements it
cannot name later.
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for every persisted entity."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
