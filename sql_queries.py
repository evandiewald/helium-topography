from sqlalchemy.engine import Engine
import h3


def get_hotspot_dict_sql(engine: Engine, hotspot_address: str):
    res = engine.execute(f"select location, name from gateway_inventory where address = '{hotspot_address}';").one()
    (lat, lon) = h3.h3_to_geo(res[0])
    return {"address": hotspot_address, "longitude": lon, "latitude": lat, "name": res[1]}


def get_witnesses_for_hotspot_sql(engine: Engine, witness_address: str, limit: int = 1000):
    sql = f"""with a as
    
    (select
    
    distinct on (transmitter_address)
    witness_address,
    transmitter_address,
    distance_km * 1000 as distance_m,
    tx_power,
    witness_signal as rssi,
    witness_snr as snr
    
    
    from challenge_receipts_parsed 
    where witness_address = '{witness_address}' 
    limit {limit})
    
    select
    transmitter_address,
    witness_address,
    distance_m,
    b.location as location_beacon,
    w.location as location_witness,
    w.elevation as elevation_witness,
    w.gain as gain_witness,
    b.elevation as elevation_beacon,
    b.gain as gain_beacon,
    tx_power,
    rssi,
    snr,
    w.owner as witness_owner
    
    from a join gateway_inventory b on b.address = a.transmitter_address 
    join gateway_inventory w on w.address = a.witness_address;"""
    res = engine.execute(sql).all()
    return [{"_from": r[0], "_to": r[1],
             "distance_m": r[2],
             "coords_beacon": h3.h3_to_geo(r[3]),
             "coords_witness": h3.h3_to_geo(r[4]),
             "elevation_witness": r[5],
             "gain_witness": r[6],
             "elevation_beacon": r[7],
             "gain_beacon": r[8],
             "tx_power": r[9],
             "rssi": r[10],
             "snr": r[11],
             "witness_owner": r[12]} for r in res]


def get_witnesses_of_hotspot_sql(engine: Engine, transmitter_address: str, limit: int = 1000):
    # note the bizarre swap of transmitter_address and witness_address labels. this is

    sql = f"""with a as

        (select

        distinct on (witness_address)
        witness_address,
        transmitter_address,
        distance_km * 1000 as distance_m,
        tx_power,
        witness_signal as rssi,
        witness_snr as snr


        from challenge_receipts_parsed 
        where transmitter_address = '{transmitter_address}' 
        limit {limit})

        select
        transmitter_address as witness_address,
        witness_address as transmitter_address,
        distance_m,
        b.location as location_beacon,
        w.location as location_witness,
        w.elevation as elevation_witness,
        w.gain as gain_witness,
        b.elevation as elevation_beacon,
        b.gain as gain_beacon,
        tx_power,
        rssi,
        snr,
        w.owner as witness_owner

        from a join gateway_inventory b on b.address = a.witness_address 
        join gateway_inventory w on w.address = a.transmitter_address;"""
    res = engine.execute(sql).all()
    return [{"_from": r[0], "_to": r[1],
             "distance_m": r[2],
             "coords_beacon": h3.h3_to_geo(r[3]),
             "coords_witness": h3.h3_to_geo(r[4]),
             "elevation_witness": r[5],
             "gain_witness": r[6],
             "elevation_beacon": r[7],
             "gain_beacon": r[8],
             "tx_power": r[9],
             "rssi": r[10],
             "snr": r[11],
             "witness_owner": r[12]} for r in res]