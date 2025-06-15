from datetime import date
import pandas as pd
from sqlalchemy.sql import text
import torch
from typing import Any, Dict, List

from ..database.models.database import get_db_session
from ..tensors.baseball.baseball_atomics import PitchSelect


def get_player(playerId: str, game_date: date = date.today()) -> pd.DataFrame:
    query = text("""
        SELECT 
            first_name,
            last_name,
            height,
            weight,
            bats,
            throws,
            EXTRACT(YEAR FROM AGE(:game_date, birthdate))::integer AS age,
            (EXTRACT(YEAR FROM :game_date) - rookie_season)::integer AS exp
        FROM players
        WHERE player_id = :player_id
    """)

    with get_db_session() as session:
        return pd.read_sql(query, session.bind, params={
            "player_id": playerId,
            "game_date": game_date
        })


def get_stadium(stadiumId: str) -> pd.DataFrame:
    query = text("""
            SELECT name
            FROM stadiums
            WHERE stadium_id = :stadium_id
            """)
    with get_db_session() as session:
        return pd.read_sql(query, session.bind, params={
            "stadium_id": stadiumId
        })


def setModels(batterId, pitcherId, stadiumId):
    models = {}
    models["pitchSelect"] = PitchSelect(pitcherId).eval()
    return models 


def setTensors(batter, pitcher):
    query = """
            SELECT AVG(EXTRACT(YEAR FROM AGE(g.game_date, batter.birthdate))::integer) AS avg_age,
                    STDDEV(EXTRACT(YEAR FROM AGE(g.game_date, batter.birthdate))::integer) AS std_age,
                    AVG((EXTRACT(YEAR FROM g.game_date) - batter.rookie_season)::integer) AS avg_exp,
                    STDDEV((EXTRACT(YEAR FROM g.game_date) - batter.rookie_season)::integer) AS std_exp

            """
    with get_db_session() as session:
        metrics = pd.read_sql(query, session.bind) 
    
    return {"batter": {"batter_age": torch.tensor((batter["age"]-metrics["avg_age"])/metrics["std_age"], torch.float32),
                        "batter_exp": torch.tensor((batter["exp"]-metrics["avg_exp"])/metrics["std_exp"], torch.float32),
                        "batter_faces_break": torch.tensor(int(batter["bats"] == pitcher["throws"]), torch.int8),
                        },
            "pitcher": {"pitcher_age": torch.tensor((pitcher["age"]-metrics["avg_age"])/metrics["std_age"], torch.float32),
                        "pitcher_exp": torch.tensor((pitcher["exp"]-metrics["avg_exp"])/metrics["std_exp"], torch.float32),
                        "pitcher_throws_lefty": torch.tensor(int(pitcher["throws"] == 'L'), torch.int8),
                        }
            }


class AtBatState:

    def __init__(self):

        self.pc = 0 
        self.balls = 0
        self.strikes = 0
        self.has_finished = True 
        self.outcome = "AtBatState Not Implemented"


    def _finished(self, result):
        self.outcome = result 
        self.has_finished = True


    def get_pitch_count(self):
        return { "balls": torch.tensor(self.balls, torch.int8), 
                "strikes": torch.tensor(self.strikes, torch.int8),
                "pitch_count": torch.tensor(self.pc, torch.int32) }


    def update(self, swing: bool, result: str):
        # print(result)
        if result in ("single", "double", "triple", "home run", 
                        "foul out", "in play out", "walk", "strike out"):
            self._finished(result)

        elif result == "ball":
            self.balls +=1
            if balls > 3:
                self._finished("walk")

        elif result in ("strike looking", "strike swinging"):
            self.strikes +=1
            if self.strikes > 2:
                self._finished("strike out")

        elif result == "foul ball":
            if self.strikes < 2:
                self.strikes +=1
        else:
            raise NotImplementedError

    

class AtBatSimulator:
    """
    Predicts the expected on-base percentage (OBP), slugging percentage (SLG),
    batting average (AVG), home run (HR) likelihood, and strikeout (K) likelihood
    for a single at-bat between a given batter and pitcher.
    """

    def __init__(self, batterId: str, pitcherId: str, stadiumId: str):

        self.batter = get_player(batterId)
        self.pitcher = get_player(pitcherId)    
        self.stadium = get_stadium(stadiumId)  

        print(self.batter, self.pitcher, self.stadium)
        raise

        self.tensors = setTensors(self.batter, self.pitcher)
        self.models = setModels(batterId, pitcherId, stadiumId)

        
    def simulate(self, context: Dict[str, Any]=None) -> str: 
        """
        context refers to runners on base etc. 
        implimented maybe in the future.
        """

        atBat = AtBatState()  
        
        while not atBat.has_finished:

            self.tensors["count"] = atBat.get_pitch_count()

            # 1. Predict pitch characteristics
            # pitch = [pitch_type, pitch_location, velocity]
            output = self.models["pitch_select"](self.tensors)
            self.tensors["pitch"] = [list(torch.argmax(pitch[0], dim=1), torch.argmax(pitch[1], dim=1), pitch[2]) for pitch in output][0]
            print(self.tensors["pitch"])
            
        return atBat.pc, atBat.outcome




if __name__ == "__main__":

    pitcherId = 'mlb.p.62972'  # Paul Skenes
    batterId = 'mlb.p.10835'  # Shohei Ohtani
    stadiumId = '268' # Wrigley Field

    atBat = AtBatSimulator(batterId=batterId, pitcherId=pitcherId, stadiumId=stadiumId)
    print(atBat.simulate())