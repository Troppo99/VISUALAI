import sys
import json
import subprocess
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal


class PingWorker(QThread):
    """
    Kelas worker untuk melakukan ping di thread terpisah.
    """

    resultSignal = pyqtSignal(dict)  # Memancarkan {camera_name: status, ...} setelah selesai

    def __init__(self, camera_data, parent=None):
        super().__init__(parent)
        self.camera_data = camera_data
        self._stop_flag = False

    def run(self):
        results = {}
        for camera_name, details in self.camera_data.items():
            if self._stop_flag:
                break  # Berhenti lebih awal jika diperlukan

            ip = details.get("ip", None)
            if not ip:
                results[camera_name] = "No IP Found"
                continue

            status = self.ping_ip(ip)
            results[camera_name] = status

        # Emit sinyal saat sudah selesai melakukan ping semua kamera
        self.resultSignal.emit(results)

    def stop(self):
        """
        Jika Anda ingin membatalkan mid-process,
        Anda dapat mengatur flag stop ini dan melakukan pengecekan di run().
        """
        self._stop_flag = True

    def ping_ip(self, ip):
        """
        Ping ke IP. Return 'OK' jika sukses, 'RTO' jika unreachable/timed out.
        """
        try:
            cmd = ["ping", "-n", "1", ip]  # Windows
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)

            output_lower = result.stdout.lower()

            # Jika command gagal total (misal IP invalid), return RTO
            if result.returncode != 0:
                return "Request timed out"

            # Meskipun returncode == 0, cek apakah ada 'unreachable' atau 'timed out'
            if "unreachable" in output_lower or "request timed out" in output_lower:
                return "Request timed out"

            return "OK"
        except subprocess.TimeoutExpired:
            # Jika ping terlalu lama tidak respon
            return "Hardware not responding"
        except Exception as e:
            print(f"Error pinging IP: {ip}, Error: {e}")
            return "Error"


class PingCheckerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ping Checker - Camera Configuration (Threaded)")
        self.resize(600, 400)

        # Widget utama dan layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Tombol Check Ping manual (opsional)
        self.btn_check_ping = QPushButton("Check Ping All Cameras (Manual)")
        self.btn_check_ping.clicked.connect(self.start_check_all_cameras)
        main_layout.addWidget(self.btn_check_ping)

        # Tabel untuk menampilkan hasil
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Camera Name", "Ping Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        main_layout.addWidget(self.table)

        # Path file JSON
        self.camera_config_path = r"\\10.5.0.3\VISUALAI\website-django\static\resources\conf\camera_config.json"

        # Load data kamera
        self.camera_data = self.load_camera_config()
        self.setup_table()

        # QTimer untuk pengecekan real-time tanpa tekan tombol
        self.timer = QTimer(self)
        self.timer.setInterval(5000)  # 5 detik sekali
        self.timer.timeout.connect(self.start_check_all_cameras)
        self.timer.start()

        # Variabel untuk menampung worker yang sedang berjalan
        self.worker = None

    def load_camera_config(self):
        """Membaca file JSON dan mengembalikan dict berisi data kamera."""
        try:
            with open(self.camera_config_path, "r") as f:
                config = json.load(f)
            return config
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading camera config: {e}")
            return {}

    def setup_table(self):
        """Set jumlah row di tabel sesuai dengan jumlah kamera, isi nama kamera."""
        cameras = list(self.camera_data.items())  # [(cam_name, {ip: ...}), ...]
        self.table.setRowCount(len(cameras))

        for row_index, (cam_name, details) in enumerate(cameras):
            self.table.setItem(row_index, 0, QTableWidgetItem(cam_name))
            self.table.setItem(row_index, 1, QTableWidgetItem("Not Checked"))

    def start_check_all_cameras(self):
        """
        Memulai thread worker untuk melakukan ping.
        Dicek apakah ada worker yang masih berjalan, jika iya bisa di-skip atau di-stop
        sesuai kebutuhan.
        """
        if self.worker and self.worker.isRunning():
            # print("Pengecekan ping masih berlangsung... (skip atau stop)")
            # Misalnya: kita skip jika masih berlangsung
            return

        # Buat worker baru
        self.worker = PingWorker(self.camera_data)
        self.worker.resultSignal.connect(self.update_table)
        self.worker.start()

    def update_table(self, results):
        """
        Menerima sinyal hasil dari worker, lalu memperbarui tabel di GUI thread utama.
        results = {camera_name: status, ...}
        """
        cameras = list(self.camera_data.items())

        for row_index, (cam_name, details) in enumerate(cameras):
            status = results.get(cam_name, "Unknown")
            self.table.setItem(row_index, 1, QTableWidgetItem(status))

    def closeEvent(self, event):
        """
        Ketika window ditutup, pastikan worker dihentikan (jika masih berjalan).
        """
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    # Jalankan aplikasi
    app = QApplication(sys.argv)
    window = PingCheckerWindow()
    window.show()
    sys.exit(app.exec())
