from pydantic import BaseModel

class StressDetect(BaseModel):
    lex_liwc_Tone: float
    lex_liwc_posemo:float
    lex_liwc_negemo:float
    lex_liwc_anx:float
    lex_liwc_Dic:float
    sentiment:float
