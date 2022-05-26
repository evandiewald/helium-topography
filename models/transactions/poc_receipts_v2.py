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
    receipt: Optional[Receipt]
    witnesses: List[Witness]


class PocReceiptsV2(BaseModel):
    block: int
    block_hash: str
    type: str
    challenger: str
    secret: str
    onion_key_hash: str
    path: List[PathElement]
    fee: int
    block_hash: str
