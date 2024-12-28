import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import uvicorn
from fastapi import FastAPI, File, UploadFile, Response, BackgroundTasks

app = FastAPI(
    title="AI Bootcamp CV Batch 6 Kidney Stone API",
    description="Kidney Stone Detection API"
)
model = YOLO("bestyolov8n.pt")

@app.get("/")
async def index():
    return {"message": "Hello, World!"}

@app.post("/detect/")
async def detect_objects(modelSelection: str, file: UploadFile, background_tasks: BackgroundTasks):
    match modelSelection:
        case "yolov8s":
            model = YOLO("bestyolov8s.pt")
        case "yolov8m":
            model = YOLO("bestyolov8m.pt")
        case "yolov11":
            model = YOLO("bestyolov11modded.pt")
        case _:
            model = YOLO("bestyolov8n.pt")

    # Process the uploaded image for object detection
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Perform object detection with YOLOv8
    detections = model(image)
    detect_img = detections[0].plot()
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    #print(detect_img)
    result_img = generate_img(detect_img)
    bufContents: bytes = result_img.getvalue()
    background_tasks.add_task(result_img.close)
    headers = {'Content-Disposition': 'inline; filename="FigureKidneyStone.png"'}
    return Response(bufContents, headers=headers, media_type='image/png')

def generate_img(image):
    plt.rcParams['figure.autolayout'] = True
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(image)
    ax.axis('off')
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    return img_buffer

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)