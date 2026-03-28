from pathlib import Path

import pandas as pd


SEASONS = {
    "2019-2020": "1920",
    "2020-2021": "2021",
    "2021-2022": "2122",
    "2022-2023": "2223",
    "2023-2024": "2324",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
OUTPUT_PATH = Path("data/raw/matches.csv")

TEAM_NAME_MAPPING = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Liverpool": "Liverpool",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Tottenham": "Tottenham Hotspur",
    "Watford": "Watford",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}

SOURCE_COLUMNS = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
}


def normalise_team_name(team_name: str) -> str:
    cleaned_name = str(team_name).strip()
    return TEAM_NAME_MAPPING.get(cleaned_name, cleaned_name)


def load_season_matches(_season_label: str, season_code: str) -> pd.DataFrame:
    source_url = BASE_URL.format(season_code=season_code)
    season_df = pd.read_csv(source_url)
    matches_df = season_df[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS).copy()
    matches_df = matches_df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    matches_df["date"] = pd.to_datetime(matches_df["date"], dayfirst=True, errors="coerce")
    matches_df = matches_df.dropna(subset=["date"])
    matches_df["home_team"] = matches_df["home_team"].map(normalise_team_name)
    matches_df["away_team"] = matches_df["away_team"].map(normalise_team_name)
    matches_df["home_goals"] = matches_df["home_goals"].astype(int)
    matches_df["away_goals"] = matches_df["away_goals"].astype(int)
    matches_df["date"] = matches_df["date"].dt.strftime("%Y-%m-%d")
    return matches_df


def build_matches_dataset() -> pd.DataFrame:
    season_frames = [load_season_matches(label, code) for label, code in SEASONS.items()]
    matches_df = pd.concat(season_frames, ignore_index=True)
    matches_df = matches_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return matches_df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matches_df = build_matches_dataset()
    matches_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(matches_df)} Premier League matches to {OUTPUT_PATH}")
    print(f"Seasons included: {', '.join(SEASONS)}")


if __name__ == "__main__":
    main()
