from pydantic import BaseModel
from typing import List, Optional


class Witness(BaseModel):
    channel: int
    datarate: str
    frequency: float
    gateway: str
    is_valid: Optional[bool]
    invalid_reason: Optional[str]
    packet_hash: str
    signal: int
    snr: float
    timestamp: int
    location: str


class Receipt(BaseModel):
    channel: int
    data: str
    datarate: Optional[str]
    frequency: float
    gateway: str
    origin: str
    signal: int
    snr: float
    timestamp: int
    tx_power: int


class PathElement(BaseModel):
    challengee: str
    challengee_owner: str
    challengee_location: str
    receipt: Optional[Receipt]
    witnesses: List[Witness]


class PocReceiptsV2(BaseModel):
    type: str
    secret: str
    block_hash: str
    challenger: str
    onion_key_hash: str
    challenger_owner: str
    fee: int
    hash: str
