from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder
from sqlalchemy.engine import create_engine, Engine
from typing import Optional
import os


load_dotenv()


def connect() -> Optional[Engine]:
    try:
        tunnel = SSHTunnelForwarder(
            (os.getenv("ETL_ADDRESS"), 22),
            ssh_username=os.getenv("ETL_USERNAME"),
            ssh_pkey=os.getenv("SSH_PKEY_PATH"),
            ssh_private_key_password=os.getenv("SSH_PKEY_PW"),
            remote_bind_address=("127.0.0.1", 5432)
        )

        tunnel.start()

        engine = create_engine(
            f'postgresql://{os.getenv("PG_USERNAME")}:{os.getenv("PG_PW")}@127.0.0.1:{tunnel.local_bind_port}/{os.getenv("PG_DB")}?sslmode=allow')
        return engine
    except:
        raise ConnectionError("Connection Failed")

