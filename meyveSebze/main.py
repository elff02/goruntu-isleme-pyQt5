import sys
import requests
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QTextEdit, QVBoxLayout, QHBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PIL import Image
import torch
from torchvision import transforms, models


class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🥕🍎 Sebze/Meyve Sınıflandırıcı")
        self.setFixedSize(800, 900)

        self.model = self.load_model()
        self.labels = self.load_labels()
        self.image_path = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Görsel alanı
        self.image_label = QLabel("Görsel buraya yüklenecek")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("""
            background-color: #333333;
            border: 4px solid #444444;
            border-radius: 15px;
        """)

        # Görseli ortalamak için yatay layout
        image_layout = QHBoxLayout()
        image_layout.addStretch()
        image_layout.addWidget(self.image_label)
        image_layout.addStretch()

        # Sonuç kutusu
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setFont(QFont("Arial", 36))
        self.result_box.setStyleSheet("""
            QTextEdit {
                background-color: #1a237e;
                color: #FF4081;
                border: 1px solid #666666;
                border-radius: 10px;
                padding: 20px;
            }
        """)

        # En olası sınıf etiketi
        self.top_class_label = QLabel("En olası sınıf: -")
        self.top_class_label.setAlignment(Qt.AlignCenter)
        self.top_class_label.setFont(QFont("Arial", 30, QFont.Bold))
        self.top_class_label.setStyleSheet("color: #FFFFFF;")

        # Görsel yükle butonu
        load_button = QPushButton("Görsel Yükle")
        load_button.setStyleSheet("""
            QPushButton {
                background-color: #FF4081;
                color: white;
                font-size: 18px;
                border-radius: 25px;
                padding: 20px 50px;
                margin: 20px 0;
            }
            QPushButton:hover {
                background-color: #F50057;
                box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.3);
            }
        """)
        load_button.clicked.connect(self.load_image)

        # Sınıflandır butonu
        classify_button = QPushButton("Sınıflandır")
        classify_button.setStyleSheet("""
            QPushButton {
                background-color: #FF4081;
                color: white;
                font-size: 18px;
                border-radius: 25px;
                padding: 20px 50px;
                margin: 20px 0;
            }
            QPushButton:hover {
                background-color: #F50057;
                box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.3);
            }
        """)
        classify_button.clicked.connect(self.classify_image)

        # Sayfa içerikleri
        layout = QVBoxLayout()
        layout.addLayout(image_layout)
        layout.addWidget(load_button)
        layout.addWidget(classify_button)
        layout.addWidget(self.top_class_label)
        layout.addWidget(self.result_box)

        # Ana container
        container = QWidget()
        container.setLayout(layout)

        # Ana pencere
        main_layout.addWidget(container)
        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(main_layout)

        # Arka plan siyah
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                font-family: 'Arial';
                font-size: 20px;
            }
        """)

    def load_model(self):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.eval()
        return model

    def load_labels(self):
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"HTTP Hatası: {response.status_code}")
            return {}

        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as e:
            print(f"JSON Hatası: {e}")
            return {}

        return {int(k): v[1] for k, v in data.items()}

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name).scaled(400, 400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.top_class_label.setText("En olası sınıf: -")
            self.result_box.setText("")

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch

    def classify_image(self):
        if not self.image_path:
            self.result_box.setText("Lütfen önce bir görsel yükleyin.")
            return

        input_batch = self.preprocess_image(self.image_path)

        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top2_prob, top2_catid = torch.topk(probabilities, 2)

        result_lines = []
        for i in range(2):
            class_id = top2_catid[i].item()
            label = self.labels.get(class_id, "Bilinmiyor")
            prob = top2_prob[i].item() * 100
            result_lines.append(f"{label}: {prob:.2f}%")

        self.top_class_label.setText(f"En olası sınıf: {self.labels.get(top2_catid[0].item(), '-')}")
        self.result_box.setText("\n".join(result_lines))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())
