from pathlib import Path

import pandas as pd


MATCHES_PATH = Path("data/raw/matches.csv")
OUTPUT_PATH = Path("data/raw/team_mapping.csv")

TEAM_MAPPING = [
    {
        "team_name": "Arsenal",
        "stadium_name": "Emirates Stadium",
        "city": "London",
        "latitude": 51.5549,
        "longitude": -0.1084,
        "stadium_capacity": 60704,
        "pitch_type": "grass",
        "altitude_m": 25,
    },
    {
        "team_name": "Aston Villa",
        "stadium_name": "Villa Park",
        "city": "Birmingham",
        "latitude": 52.5092,
        "longitude": -1.8849,
        "stadium_capacity": 42682,
        "pitch_type": "grass",
        "altitude_m": 130,
    },
    {
        "team_name": "Bournemouth",
        "stadium_name": "Vitality Stadium",
        "city": "Bournemouth",
        "latitude": 50.7352,
        "longitude": -1.8384,
        "stadium_capacity": 11307,
        "pitch_type": "grass",
        "altitude_m": 14,
    },
    {
        "team_name": "Brentford",
        "stadium_name": "Gtech Community Stadium",
        "city": "London",
        "latitude": 51.4908,
        "longitude": -0.2888,
        "stadium_capacity": 17250,
        "pitch_type": "grass",
        "altitude_m": 10,
    },
    {
        "team_name": "Brighton",
        "stadium_name": "American Express Stadium",
        "city": "Brighton",
        "latitude": 50.8616,
        "longitude": -0.0837,
        "stadium_capacity": 31876,
        "pitch_type": "grass",
        "altitude_m": 18,
    },
    {
        "team_name": "Burnley",
        "stadium_name": "Turf Moor",
        "city": "Burnley",
        "latitude": 53.7890,
        "longitude": -2.2303,
        "stadium_capacity": 21944,
        "pitch_type": "grass",
        "altitude_m": 123,
    },
    {
        "team_name": "Chelsea",
        "stadium_name": "Stamford Bridge",
        "city": "London",
        "latitude": 51.4817,
        "longitude": -0.1910,
        "stadium_capacity": 40343,
        "pitch_type": "grass",
        "altitude_m": 8,
    },
    {
        "team_name": "Crystal Palace",
        "stadium_name": "Selhurst Park",
        "city": "London",
        "latitude": 51.3983,
        "longitude": -0.0856,
        "stadium_capacity": 25486,
        "pitch_type": "grass",
        "altitude_m": 67,
    },
    {
        "team_name": "Everton",
        "stadium_name": "Goodison Park",
        "city": "Liverpool",
        "latitude": 53.4388,
        "longitude": -2.9664,
        "stadium_capacity": 39414,
        "pitch_type": "grass",
        "altitude_m": 41,
    },
    {
        "team_name": "Fulham",
        "stadium_name": "Craven Cottage",
        "city": "London",
        "latitude": 51.4749,
        "longitude": -0.2217,
        "stadium_capacity": 25700,
        "pitch_type": "grass",
        "altitude_m": 8,
    },
    {
        "team_name": "Leeds United",
        "stadium_name": "Elland Road",
        "city": "Leeds",
        "latitude": 53.7778,
        "longitude": -1.5722,
        "stadium_capacity": 37908,
        "pitch_type": "grass",
        "altitude_m": 78,
    },
    {
        "team_name": "Leicester City",
        "stadium_name": "King Power Stadium",
        "city": "Leicester",
        "latitude": 52.6204,
        "longitude": -1.1422,
        "stadium_capacity": 32312,
        "pitch_type": "grass",
        "altitude_m": 62,
    },
    {
        "team_name": "Liverpool",
        "stadium_name": "Anfield",
        "city": "Liverpool",
        "latitude": 53.4308,
        "longitude": -2.9608,
        "stadium_capacity": 61276,
        "pitch_type": "grass",
        "altitude_m": 27,
    },
    {
        "team_name": "Luton Town",
        "stadium_name": "Kenilworth Road",
        "city": "Luton",
        "latitude": 51.8842,
        "longitude": -0.4315,
        "stadium_capacity": 12227,
        "pitch_type": "grass",
        "altitude_m": 124,
    },
    {
        "team_name": "Manchester City",
        "stadium_name": "Etihad Stadium",
        "city": "Manchester",
        "latitude": 53.4831,
        "longitude": -2.2004,
        "stadium_capacity": 53400,
        "pitch_type": "grass",
        "altitude_m": 45,
    },
    {
        "team_name": "Manchester United",
        "stadium_name": "Old Trafford",
        "city": "Manchester",
        "latitude": 53.4631,
        "longitude": -2.2913,
        "stadium_capacity": 74310,
        "pitch_type": "grass",
        "altitude_m": 22,
    },
    {
        "team_name": "Newcastle United",
        "stadium_name": "St James' Park",
        "city": "Newcastle upon Tyne",
        "latitude": 54.9756,
        "longitude": -1.6217,
        "stadium_capacity": 52305,
        "pitch_type": "grass",
        "altitude_m": 51,
    },
    {
        "team_name": "Norwich City",
        "stadium_name": "Carrow Road",
        "city": "Norwich",
        "latitude": 52.6221,
        "longitude": 1.3092,
        "stadium_capacity": 27244,
        "pitch_type": "grass",
        "altitude_m": 17,
    },
    {
        "team_name": "Nottingham Forest",
        "stadium_name": "City Ground",
        "city": "Nottingham",
        "latitude": 52.9398,
        "longitude": -1.1322,
        "stadium_capacity": 30445,
        "pitch_type": "grass",
        "altitude_m": 17,
    },
    {
        "team_name": "Sheffield United",
        "stadium_name": "Bramall Lane",
        "city": "Sheffield",
        "latitude": 53.3703,
        "longitude": -1.4701,
        "stadium_capacity": 32050,
        "pitch_type": "grass",
        "altitude_m": 74,
    },
    {
        "team_name": "Southampton",
        "stadium_name": "St Mary's Stadium",
        "city": "Southampton",
        "latitude": 50.9058,
        "longitude": -1.3911,
        "stadium_capacity": 32384,
        "pitch_type": "grass",
        "altitude_m": 6,
    },
    {
        "team_name": "Tottenham Hotspur",
        "stadium_name": "Tottenham Hotspur Stadium",
        "city": "London",
        "latitude": 51.6043,
        "longitude": -0.0664,
        "stadium_capacity": 62850,
        "pitch_type": "grass",
        "altitude_m": 15,
    },
    {
        "team_name": "Watford",
        "stadium_name": "Vicarage Road",
        "city": "Watford",
        "latitude": 51.6498,
        "longitude": -0.4017,
        "stadium_capacity": 22200,
        "pitch_type": "grass",
        "altitude_m": 63,
    },
    {
        "team_name": "West Bromwich Albion",
        "stadium_name": "The Hawthorns",
        "city": "West Bromwich",
        "latitude": 52.5090,
        "longitude": -1.9639,
        "stadium_capacity": 26688,
        "pitch_type": "grass",
        "altitude_m": 160,
    },
    {
        "team_name": "West Ham United",
        "stadium_name": "London Stadium",
        "city": "London",
        "latitude": 51.5386,
        "longitude": -0.0165,
        "stadium_capacity": 62500,
        "pitch_type": "grass",
        "altitude_m": 6,
    },
    {
        "team_name": "Wolverhampton Wanderers",
        "stadium_name": "Molineux Stadium",
        "city": "Wolverhampton",
        "latitude": 52.5903,
        "longitude": -2.1304,
        "stadium_capacity": 31750,
        "pitch_type": "grass",
        "altitude_m": 128,
    },
]


def validate_mapping(mapping_df: pd.DataFrame, matches_df: pd.DataFrame) -> None:
    unique_teams = set(pd.concat([matches_df["home_team"], matches_df["away_team"]]).unique())
    mapped_teams = set(mapping_df["team_name"])

    missing_teams = sorted(unique_teams - mapped_teams)
    unexpected_teams = sorted(mapped_teams - unique_teams)
    duplicate_teams = mapping_df[mapping_df.duplicated(subset=["team_name"], keep=False)]["team_name"].tolist()

    if missing_teams:
        raise ValueError(f"Missing teams in mapping: {missing_teams}")
    if unexpected_teams:
        raise ValueError(f"Unexpected teams in mapping: {unexpected_teams}")
    if duplicate_teams:
        raise ValueError(f"Duplicate teams in mapping: {sorted(set(duplicate_teams))}")
    if mapping_df.isna().any().any():
        raise ValueError("Mapping contains missing values")

    if not mapping_df["latitude"].between(-90, 90).all():
        raise ValueError("Latitude values are out of range")
    if not mapping_df["longitude"].between(-180, 180).all():
        raise ValueError("Longitude values are out of range")
    if not (mapping_df["stadium_capacity"] > 0).all():
        raise ValueError("Stadium capacities must be positive")
    if not (mapping_df["pitch_type"].astype(str).str.strip() != "").all():
        raise ValueError("Pitch types must be non-empty")


def main() -> None:
    matches_df = pd.read_csv(MATCHES_PATH)
    mapping_df = pd.DataFrame(TEAM_MAPPING)
    validate_mapping(mapping_df, matches_df)
    mapping_df = mapping_df.sort_values("team_name").reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(mapping_df)} team mappings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
