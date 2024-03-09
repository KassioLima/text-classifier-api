def Response(response=None, sttCode: int = 200):
    return {"data": response, "code": sttCode}