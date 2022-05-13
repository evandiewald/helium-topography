from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Text, BigInteger, Float, DateTime, ForeignKey, CheckConstraint, Enum
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
import os
import enum


Base = declarative_base()


class Transactions(Base):
    __tablename__ = "transactions"

    block = Column(BigInteger, nullable=False)
    hash = Column(Text, nullable=False, primary_key=True)
    type = Column(Text, nullable=False)
    fields = Column(JSONB, nullable=False)
    time = Column(BigInteger, nullable=False)


class Blocks(Base):
    __tablename__ = "blocks"

    height = Column(BigInteger, nullable=False, primary_key=True)
    time = Column(BigInteger, nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    prev_hash = Column(Text, nullable=True)
    block_hash = Column(Text, nullable=False)
    transaction_count = Column(BigInteger)
    hbbft_round = Column(BigInteger)
    election_epoch = Column(BigInteger)
    epoch_start = Column(BigInteger)
    rescue_signature = Column(Text)
    snapshot_hash = Column(Text)
    created_at = Column(TIMESTAMP)


class TopographyResults(Base):
    __tablename__ = "topography_results"

    address = Column(Text, nullable=False, primary_key=True)
    percent_predictions_within_5_res8_krings = Column(Text)
    prediction_error_km = Column(Float)
    n_outliers = Column(BigInteger)
    n_beaconers_heard = Column(BigInteger)
    block = Column(BigInteger)