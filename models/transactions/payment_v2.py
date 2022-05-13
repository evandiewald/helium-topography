from pydantic import BaseModel
from typing import List, Optional


class PaymentV2Payment(BaseModel):
    amount: int
    memo: Optional[str]
    payee: str


class PaymentV2(BaseModel):
    hash: str
    fee: int
    # fee is in DC
    nonce: int
    payer: str
    payments: List[PaymentV2Payment]