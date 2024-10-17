import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, \
    QMessageBox, QFileDialog, QInputDialog, QComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
import yt_dlp
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set environment variable to handle potential OpenMP runtime issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_youtube_stream(url):
    ydl_opts = {'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.source = None
        try:
            self.model = YOLO('yolov8n.pt')  # Load YOLOv8 model
            logging.info("YOLO model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {str(e)}")
            self.error_signal.emit(f"Error loading YOLO model: {str(e)}")

    def set_source(self, source):
        self.source = source
        logging.info(f"Video source set to: {source}")

    def run(self):
        if self.source is None:
            logging.error("No video source specified")
            self.error_signal.emit("Error: No video source specified.")
            return

        if self.source == 'camera':
            cap = cv2.VideoCapture(0)
            logging.info("Attempting to open camera")
        elif self.source.startswith('http'):  # YouTube URL
            try:
                stream_url = get_youtube_stream(self.source)
                cap = cv2.VideoCapture(stream_url)
                logging.info(f"YouTube video opened: {self.source}")
            except Exception as e:
                logging.error(f"Error opening YouTube video: {str(e)}")
                self.error_signal.emit(f"Error opening YouTube video: {str(e)}")
                return
        else:  # Local video file
            cap = cv2.VideoCapture(self.source)
            logging.info(f"Attempting to open local video file: {self.source}")

        if not cap.isOpened():
            logging.error("Could not open video source")
            self.error_signal.emit("Error: Could not open video source.")
            return

        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                try:
                    results = self.model(cv_img)
                    annotated_frame = results[0].plot()
                    self.change_pixmap_signal.emit(annotated_frame)
                except Exception as e:
                    logging.error(f"Error processing frame: {str(e)}")
                    self.error_signal.emit(f"Error processing frame: {str(e)}")
            else:
                logging.warning("Could not read frame")
                self.error_signal.emit("Error: Could not read frame.")
                break
        cap.release()
        logging.info("Video capture released")

    def stop(self):
        self._run_flag = False
        self.wait()
        logging.info("Video thread stopped")


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Drone Surveillance System")
        self.display_width = 640
        self.display_height = 480

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        layout.addWidget(self.image_label)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Camera", "Video File", "YouTube URL"])
        layout.addWidget(self.source_combo)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.error_signal.connect(self.show_error)

        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_video)

        logging.info("Application initialized")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
        logging.info("Application closed")

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def start_video(self):
        source_type = self.source_combo.currentText()
        if source_type == "Camera":
            self.thread.set_source('camera')
        elif source_type == "Video File":
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
            if file_name:
                self.thread.set_source(file_name)
            else:
                logging.warning("No video file selected")
                return
        elif source_type == "YouTube URL":
            url, ok = QInputDialog.getText(self, "YouTube URL", "Enter YouTube URL:")
            if ok and url:
                self.thread.set_source(url)
            else:
                logging.warning("No YouTube URL entered")
                return
        self.thread.start()
        logging.info(f"Video started with source: {source_type}")

    def stop_video(self):
        self.thread.stop()
        logging.info("Video stopped")

    @pyqtSlot(str)
    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        logging.error(f"Error displayed: {error_message}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
