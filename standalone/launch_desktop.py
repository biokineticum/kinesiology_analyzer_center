import sys
import os
import subprocess
import time
import socket
from pathlib import Path
from PySide6.QtCore import QUrl, QTimer, Qt, QRect
from PySide6.QtWidgets import QApplication, QMainWindow, QSplashScreen
from PySide6.QtGui import QPixmap, QColor, QFont, QPainter, QLinearGradient
from PySide6.QtWebEngineWidgets import QWebEngineView

PORT = 8501
URL = f"http://localhost:{PORT}"

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class PhysioAppWindow(QMainWindow):
    def __init__(self, streamlit_process):
        super().__init__()
        self.streamlit_process = streamlit_process
        self.setWindowTitle("📊 Kinesiology Analyzer Center")
        self.resize(1366, 850)
        
        # Embedding the browser component
        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)
        self.browser.setUrl(QUrl(URL))
        
    def closeEvent(self, event):
        # Cleanly terminate streamlit background process
        if self.streamlit_process:
            try:
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=2)
            except Exception:
                try:
                    self.streamlit_process.kill()
                except Exception:
                    pass
        event.accept()

def create_splash_pixmap():
    width, height = 480, 300
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Modern gradient background with rounded corners
    gradient = QLinearGradient(0, 0, width, height)
    gradient.setColorAt(0.0, QColor("#1e1e2f"))  # Slate/Indigo
    gradient.setColorAt(1.0, QColor("#0d0d15"))  # Dark deep navy
    painter.setBrush(gradient)
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(0, 0, width, height, 16, 16)
    
    # Title Text
    painter.setPen(QColor("#ffffff"))
    font_title = QFont("Segoe UI", 18, QFont.Bold)
    painter.setFont(font_title)
    painter.drawText(QRect(20, 50, width - 40, 40), Qt.AlignCenter, "📊 KINESIOLOGY ANALYZER CENTER")
    
    # Subtitle / Caption
    painter.setPen(QColor("#a0a0c0"))
    font_sub = QFont("Segoe UI", 10)
    painter.setFont(font_sub)
    painter.drawText(QRect(20, 95, width - 40, 30), Qt.AlignCenter, "Advanced Biomechanical & Isokinetic Data Analytics")
    
    # Status Message
    painter.setPen(QColor("#60a5fa"))  # Sleek light blue
    font_status = QFont("Segoe UI", 11, QFont.Medium)
    painter.setFont(font_status)
    painter.drawText(QRect(20, 180, width - 40, 40), Qt.AlignCenter, "Uruchamianie serwerów i wczytywanie aplikacji...")
    
    # Loading details
    painter.setPen(QColor("#52525b"))
    font_small = QFont("Segoe UI", 8)
    painter.setFont(font_small)
    painter.drawText(QRect(20, 240, width - 40, 20), Qt.AlignCenter, "Ładowanie silnika Streamlit + RAG...")
    
    painter.end()
    return pixmap

class AppLauncher:
    def __init__(self):
        self.app = QApplication(sys.argv)
        
        # Setup splash screen
        pixmap = create_splash_pixmap()
        self.splash = QSplashScreen(pixmap)
        self.splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.SplashScreen)
        self.splash.setAttribute(Qt.WA_TranslucentBackground)
        self.splash.show()
        self.app.processEvents()
        
        # Start Streamlit subprocess silently
        root_dir = Path(__file__).resolve().parent.parent
        
        # We start Streamlit in headless mode with creationflags to suppress window popping up
        self.process = subprocess.Popen(
            ["streamlit", "run", "pdf_parser_trainer.py", "--server.headless=true", f"--server.port={PORT}"],
            cwd=str(root_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        # Poll server availability via QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_server)
        self.timer.start(200)  # Check every 200ms
        self.timeout_counter = 0
        
    def check_server(self):
        self.timeout_counter += 1
        if is_port_open(PORT):
            self.timer.stop()
            self.main_window = PhysioAppWindow(self.process)
            self.main_window.show()
            self.splash.finish(self.main_window)
        elif self.timeout_counter > 60:  # 12 seconds timeout
            self.timer.stop()
            self.splash.close()
            if self.process:
                self.process.kill()
            sys.exit(1)
            
    def run(self):
        sys.exit(self.app.exec())

if __name__ == "__main__":
    launcher = AppLauncher()
    launcher.run()
