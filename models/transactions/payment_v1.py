from pydantic import BaseModel


class PaymentV1(BaseModel):
    hash: str
    amount: int
    fee: int
    # fee is in DC
    nonce: int
    payer: str
    payee: str