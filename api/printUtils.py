from api.constraints import Constraints

def greenText(text: str) -> str:
    return Constraints.greenColor + text + Constraints.endCor

def customColorText(text: str, customColor: str) -> str:
    return customColor + text + Constraints.endCor
