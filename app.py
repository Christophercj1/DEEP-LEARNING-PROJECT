from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
import uuid

app = Flask(__name__)

upload_folder = "static/uploads"
os.makedirs(upload_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("carmodel.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route("/", methods=["GET", "POST"])
def prediction():
    pre = None
    conf = None

    if request.method == "POST":
        image = request.files["image_file"]

        filename = str(uuid.uuid4()) + "_" + image.filename
        file_path = os.path.join(upload_folder, filename)
        image.save(file_path)

        image_file = Image.open(file_path).convert("RGB")
        image_file = transform(image_file).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_file)
            prob = torch.softmax(output, dim=1)

        classes = ["car", "not_car"]
        confidence, prediction = torch.max(prob, 1)

        pre = classes[prediction.item()]
        conf = confidence.item()

    return render_template("DL WEB PAGE.html", p=pre, c=conf)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
