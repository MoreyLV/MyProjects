from fastapi import FastAPI, WebSocket
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
import json
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
operation = 2

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        raw_data = await websocket.receive_text()
        data = json.loads(raw_data)

        if data["type"] == "array":
            numbers = data["data"]
            flat_numbers = [[int(cell.strip()) for cell in row] for row in numbers]
            recognition_data_massive = np.array(flat_numbers)
            recognition_data_massive_flaten = recognition_data_massive.flatten().reshape(1, -1)

            weight = np.loadtxt("weights.txt")
            bias = np.loadtxt("bias.txt")

            final_output = sigmoid(recognition_data_massive_flaten.dot(weight) + bias)
            indexated_final = np.argmax(final_output)

            print(indexated_final)
            response = indexated_final
            await websocket.send_text(f"{response}")


