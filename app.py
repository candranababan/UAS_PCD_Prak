from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

def process_image(image, process):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if process == "rgb":
        return img

    if process == "hsv":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if process == "threshold":
        _, result = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        return result

    if process == "otsu":
        _, result = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    kernel = np.ones((5,5), np.uint8)

    if process == "erosi":
        return cv2.erode(gray, kernel, iterations=1)

    if process == "dilasi":
        return cv2.dilate(gray, kernel, iterations=1)

    if process == "opening":
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    if process == "closing":
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    if process == "floodfill":
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        h, w = thresh.shape
        mask = np.zeros((h+2, w+2), np.uint8)
        flood = thresh.copy()
        cv2.floodFill(flood, mask, (0,0), 255)
        flood_inv = cv2.bitwise_not(flood)
        return thresh | flood_inv

    return gray


@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None

    if request.method == "POST":
        file = request.files["image"]
        process = request.form["process"]

        img = Image.open(file).convert("RGB")
        img_np = np.array(img)
        result = process_image(img_np, process)

        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        _, buffer = cv2.imencode(".png", result)
        output_image = base64.b64encode(buffer).decode("utf-8")

    return render_template("index.html", output_image=output_image)


if __name__ == "__main__":
    app.run()
