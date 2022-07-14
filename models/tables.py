from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Text, BigInteger, Float, Integer, Boolean, ForeignKey, Enum
import enum
import os


Base = declarative_base(bind=os.getenv("POSTGRES_CONNECTION_STR"))


class witness_invalid_reason_type(enum.Enum):
    witness_rssi_too_high = 1
    incorrect_frequency = 2
    witness_not_same_region = 3
    witness_too_close = 4
    witness_on_incorrect_channel = 5
    witness_too_far = 6


class ChallengeReceiptsParsed(Base):
    __tablename__ = "challenge_receipts_parsed"

    block = Column(BigInteger, nullable=False)
    hash = Column(Text, nullable=False, primary_key=True)
    time = Column(BigInteger, nullable=False)
    challenger = Column(Text, nullable=False)
    transmitter_address = Column(Text, ForeignKey("gateway_inventory.address"), nullable=False, index=True)
    tx_power = Column(Integer)
    origin = Column(Text)
    witness_address = Column(Text, ForeignKey("gateway_inventory.address"), nullable=False, primary_key=True)
    witness_is_valid = Column(Boolean, index=True)
    witness_invalid_reason = Column(Enum(witness_invalid_reason_type))
    witness_signal = Column(Integer)
    witness_snr = Column(Float)
    witness_channel = Column(Integer)
    witness_datarate = Column(Text)
    witness_frequency = Column(Float)
    witness_timestamp = Column(BigInteger)
    distance_km = Column(Float)


class TopographyResults(Base):
    __tablename__ = "topography_results"

    address = Column(Text, nullable=False, primary_key=True)
    percent_predictions_within_5_res8_krings = Column(Text)
    prediction_error_km = Column(Float)
    n_outliers = Column(BigInteger)
    n_beaconers_heard = Column(BigInteger)
    block = Column(BigInteger)