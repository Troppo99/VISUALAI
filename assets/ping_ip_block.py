import sys
import json
import subprocess
from PyQt6.QtWidgets import QLayout, QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QFrame, QMessageBox
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect, QSize, QThread, pyqtSignal


class FlowLayout(QLayout):
    """Layout kustom yang meniru perilaku 'flex-wrap' ala CSS."""

    def __init__(self, parent=None, margin=10, spacing=10):
        super().__init__(parent)

        # Set margin dan spacing agar ada jarak di tepi dan antar blok
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.itemList = []
        self.setSpacing(spacing)

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.doLayout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            sizeHint = item.sizeHint()
            size.setWidth(max(size.width(), sizeHint.width()))
            size.setHeight(size.height() + sizeHint.height())
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing()
            spaceY = self.spacing()
            hint = wid.sizeHint()
            nextX = x + hint.width() + spaceX
            if nextX > rect.right() and lineHeight > 0:
                # Pindah ke baris berikutnya
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + hint.width() + spaceX
                lineHeight = 0
            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = nextX
            lineHeight = max(lineHeight, hint.height())

        return y + lineHeight - rect.y()


class CameraBlock(QFrame):
    """
    Satu 'blok' untuk menampilkan 1 kamera (nama + status).
    """

    def __init__(self, camera_name, parent=None):
        super().__init__(parent)

        self.camera_name = camera_name

        # Agar punya border/garis atau style
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setLineWidth(2)

        # Layout internal
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Label nama
        self.label_name = QLabel(camera_name, self)
        self.label_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label_name)

        # Garis pemisah (opsional)
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Label status
        self.label_status = QLabel("Not Checked", self)
        self.label_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label_status)

        self.setLayout(layout)

    def set_status(self, status_text):
        """
        Update teks status di blok ini.
        Jika status == "Timeout", maka background kuning, teks hitam.
        Jika status lain, kembalikan ke style default (hitam di atas).
        """
        self.label_status.setText(status_text)

        if status_text == "Timeout":
            self.setStyleSheet("background-color: yellow; color: black;")
        else:
            # Style default: tidak ada background, teks putih (misalnya)
            self.setStyleSheet("background-color: none; color: white;")


class PingWorker(QThread):
    """
    Worker (QThread) untuk melakukan ping ke semua kamera di background.
    Agar tidak memblokir UI.
    """

    # Sinyal untuk mengirim {camera_name: status} setelah selesai
    resultSignal = pyqtSignal(dict)

    def __init__(self, camera_data, parent=None):
        """
        camera_data adalah dict {camera_name: ip}
        """
        super().__init__(parent)
        self.camera_data = camera_data
        self._stop_flag = False

    def run(self):
        results = {}
        for cam_name, ip in self.camera_data.items():
            if self._stop_flag:
                break
            status = self.ping_ip(ip)
            results[cam_name] = status

        # Emit sinyal setelah selesai memproses semua kamera
        self.resultSignal.emit(results)

    def stop(self):
        self._stop_flag = True

    def ping_ip(self, ip):
        """
        Ping beneran. Return 'OK', 'Timeout', dsb.
        """
        try:
            cmd = ["ping", "-n", "1", ip]  # Windows
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            out_lower = result.stdout.lower()

            # Jika returncode != 0, biasanya IP tidak reachable
            if result.returncode != 0:
                return "Timeout"
            # Atau jika ada "unreachable"/"request timed out" di output
            if "unreachable" in out_lower or "request timed out" in out_lower:
                return "Timeout"
            return "OK"
        except subprocess.TimeoutExpired:
            return "Timeout"
        except Exception as e:
            print(f"Ping error: {e}")
            return "Error"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PING IP CCTV")
        self.resize(1000, 600)

        # Widget utama
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # FlowLayout dengan margin dan spacing
        self.layout = FlowLayout(central_widget, margin=10, spacing=10)
        central_widget.setLayout(self.layout)

        # Path file JSON
        self.config_path = r"\\10.5.0.3\VISUALAI\website-django\static\resources\conf\camera_config.json"

        # Muat nama kamera dan buat dict {camera_name: ip}
        self.camera_data = {}
        camera_names = self.get_all_camera_names()
        for cam_name in camera_names:
            ip = self.camera_config_by_name(cam_name)
            self.camera_data[cam_name] = ip

        # Dictionary {camera_name: CameraBlock}
        self.camera_blocks = {}

        # Buat block untuk setiap kamera
        for cam_name in self.camera_data.keys():
            block = CameraBlock(cam_name)
            self.layout.addWidget(block)
            self.camera_blocks[cam_name] = block

        # Timer untuk memicu pengecekan ping berkala
        self.timer = QTimer(self)
        self.timer.setInterval(3000)  # misal 3 detik
        self.timer.timeout.connect(self.start_ping_thread)
        self.timer.start()

        # Variabel untuk menampung worker (agar tidak didestroy sebelum selesai)
        self.worker = None

    def get_all_camera_names(self):
        """
        Mendapatkan semua key (nama kamera) dari file JSON.
        """
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            return list(config.keys())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal baca config: {e}")
            return []

    def camera_config_by_name(self, camera_name):
        """
        Method untuk mengambil IP dari file JSON dengan parameter camera_name.
        """
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            return config[camera_name]["ip"]
        except Exception as e:
            print(f"Error loading IP for {camera_name}: {e}")
            return "0.0.0.0"  # fallback

    def start_ping_thread(self):
        """
        Dipanggil oleh QTimer untuk memulai worker ping.
        Jika worker sebelumnya masih jalan, bisa kita skip atau stop.
        """
        if self.worker and self.worker.isRunning():
            print("Ping worker masih berjalan, skip dulu...")
            return

        self.worker = PingWorker(self.camera_data)
        self.worker.resultSignal.connect(self.handle_ping_results)
        self.worker.start()

    def handle_ping_results(self, results):
        """
        Menerima dict {camera_name: status} dari worker,
        lalu memperbarui UI di thread utama.
        """
        for cam_name, status in results.items():
            block = self.camera_blocks.get(cam_name)
            if block:
                block.set_status(status)

    def closeEvent(self, event):
        """
        Ketika jendela utama ditutup, pastikan worker (jika ada) dihentikan.
        """
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
