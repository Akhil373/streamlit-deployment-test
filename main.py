from fastapi import FastAPI, File, UploadFile
import shutil
import os
from modelcode import debug_solve_local

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        result = debug_solve_local(temp_file)
        return {"result": result}
    finally:
        os.remove(temp_file)
