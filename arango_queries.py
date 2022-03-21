from pyArango.connection import Database


def list_hotspots_near_coords(db: Database, coordinates=None, limit: int = 100, search_radius_m: int = 75000):
    if coordinates is None:
        coordinates = [37.7749, -122.4194]
    # select N most recently active hotspots within radius
    aql = f"""let coordinates = {coordinates}
    for hotspot in hotspots
        let dist_from_root = DISTANCE(coordinates[0], coordinates[1], hotspot.location_geo.coordinates[1], hotspot.location_geo.coordinates[0])
        filter dist_from_root < {search_radius_m}
        sort hotspot.last_block desc
        limit {limit}
        return {{address: hotspot.address, 
                owner: hotspot.owner,
                distance_from_root: dist_from_root,
                longitude: hotspot.location_geo.coordinates[0], 
                latitude: hotspot.location_geo.coordinates[1]}}"""
    return db.fetch_list(aql)


def get_witnesses_for_hotspot(db: Database, address: str, limit: int = 1000):
    aql = f"""
            let hotspot = DOCUMENT('hotspots/{address}')
            for v, e, p in 1..1 inbound hotspot poc_receipts
                collect beaconer = e._from, witness = e._to into witnessPair = {{
                    distance_m: GEO_DISTANCE(DOCUMENT(e._from).location_geo, DOCUMENT(e._to).location_geo),
                    coords_beacon: DOCUMENT(e._from).location_geo.coordinates,
                    coords_witness: DOCUMENT(e._to).location_geo.coordinates,
                    elevation_witness: DOCUMENT(e._to).elevation,
                    gain_witness: DOCUMENT(e._to).gain,
                    tx_power: e.tx_power,
                    rssi: e.signal,
                    snr: e.snr,
                    witness_owner: DOCUMENT(e._to).owner
                }}
                limit {limit}
                RETURN {{_from: LAST(SPLIT(beaconer, "/")), _to: LAST(SPLIT(witness, "/")), 
                    distance_m: witnessPair[0].distance_m, 
                    coords_beacon: witnessPair[0].coords_beacon,
                    coords_witness: witnessPair[0].coords_witness,
                    elevation_witness: witnessPair[0].elevation_witness,
                    gain_witness: witnessPair[0].gain_witness,
                    elevation_beacon: hotspot.elevation,
                    gain_beacon: hotspot.gain,
                    tx_power: AVG(witnessPair[*].tx_power),
                    rssi: AVG(witnessPair[*].rssi),
                    snr: AVG(witnessPair[*].snr),
                    witness_owner: witnessPair[0].witness_owner}}"""
    return db.fetch_list(aql)


def get_witnesses_of_hotspot(db: Database, address: str, limit: int = 1000):
    aql = f"""
            let hotspot = DOCUMENT('hotspots/{address}')
            for v, e, p in 1..1 outbound hotspot poc_receipts
                collect beaconer = e._from, witness = e._to into witnessPair = {{
                    distance_m: GEO_DISTANCE(DOCUMENT(e._from).location_geo, DOCUMENT(e._to).location_geo),
                    coords_beacon: DOCUMENT(e._from).location_geo.coordinates,
                    coords_witness: DOCUMENT(e._to).location_geo.coordinates,
                    elevation_witness: DOCUMENT(e._to).elevation,
                    gain_witness: DOCUMENT(e._to).gain,
                    tx_power: e.tx_power,
                    rssi: e.signal,
                    snr: e.snr,
                    witness_owner: DOCUMENT(e._to).owner
                }}
                limit {limit}
                RETURN {{_from: LAST(SPLIT(beaconer, "/")), _to: LAST(SPLIT(witness, "/")), 
                    distance_m: witnessPair[0].distance_m, 
                    coords_beacon: witnessPair[0].coords_beacon,
                    coords_witness: witnessPair[0].coords_witness,
                    elevation_witness: witnessPair[0].elevation_witness,
                    gain_witness: witnessPair[0].gain_witness,
                    elevation_beacon: hotspot.elevation,
                    gain_beacon: hotspot.gain,
                    tx_power: AVG(witnessPair[*].tx_power),
                    rssi: AVG(witnessPair[*].rssi),
                    snr: AVG(witnessPair[*].snr),
                    witness_owner: witnessPair[0].witness_owner}}"""
    return db.fetch_list(aql)


def get_hotspot_dict(db: Database, hotspot_address: str):
    aql = f"""
    for hotspot in hotspots
        filter hotspot.address == '{hotspot_address}'
        return {{address: hotspot.address, longitude: hotspot.location_geo.coordinates[0], latitude: hotspot.location_geo.coordinates[1], name: hotspot.name}}"""
    return db.fetch_list(aql)[0]