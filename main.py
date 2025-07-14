import sys, random, os, time
from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt, QSize, QTimer, QThread, QObject, pyqtSignal, pyqtSlot, QRectF
from PyQt5.QtGui import QPainter, QColor, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QWidget
from smart_cropping import get_real_rect, qpixmap_to_numpy

FPS = 60
SCREEN_W, SCREEN_H = 640, 480
FLOOR = 360 / 480


def scaled_pix(path, *, w=None, h=None):
    pix = QPixmap(path)
    if pix.isNull():
        w = w or h or 32
        h = h or w
        stub = QPixmap(w, h)
        stub.fill(QColor("magenta"))
        return stub
    if w:
        return pix.scaledToWidth(w, Qt.SmoothTransformation)
    if h:
        return pix.scaledToHeight(h, Qt.SmoothTransformation)
    return pix


def load_anim_frames(folder: Path, *, h: int) -> List[QPixmap]:
    frames = []
    for f in sorted(folder.glob("frame*.png")):
        frames.append(scaled_pix(str(f), h=h))
    return frames or [scaled_pix("", h=h)]


class CVWorker(QObject):
    energyReady = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._running = True

    @pyqtSlot()
    def stop(self):
        self._running = False

    @pyqtSlot()
    def run(self):
        from energy_calculator import PoseEnergyAnalyzer, cv2
        backend = cv2.CAP_DSHOW if os.name == "nt" else 0
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            self.finished.emit()
            return
        analyser = PoseEnergyAnalyzer()
        while self._running:
            ok, frame = cap.read()
            if not ok:
                break
            _, e = analyser.process(frame)
            if e:
                inc = int(e * 50)
                self.energyReady.emit(inc)
            QThread.msleep(1)
        cap.release()
        self.finished.emit()


class Enemy:
    def __init__(self, x_world, y_world, speed, frames):
        self.x_world = x_world
        self.y_world = y_world
        self.speed = speed
        self.frames = frames
        self.frame_idx = 0
        self._ticker = 0

    def step(self):
        self.x_world += self.speed
        self._ticker += 1
        if self._ticker % 8 == 0:
            self.frame_idx = (self.frame_idx + 1) % len(self.frames)

    @property
    def pix(self):
        return self.frames[self.frame_idx]

    def rect(self):
        p = self.pix
        x, y, w, h = get_real_rect(qpixmap_to_numpy(p))
        return QRectF(x + self.x_world, y + self.y_world, w, h)


class GameWidget(QWidget):
    newEnergy = pyqtSignal(object)
    HERO_SCREEN_X_FRAC = 0.33
    CHAR_HEIGHT = 64
    ENEMY_SPAWN_MS = 1800

    def __init__(self):
        super().__init__()
        self.setMinimumSize(QSize(320, 240))
        self.bg_pix = scaled_pix("assets/background.jpg", h=SCREEN_H)
        self.hero_frames = load_anim_frames(Path(os.path.join("assets", "character1")), h=self.CHAR_HEIGHT)
        self.hero_frame_idx = 0
        self._hero_anim_tick = 0
        self._enemy_design_folders = [p for p in Path(os.path.join("assets", "enemies")).iterdir() if p.is_dir()]
        self._enemy_cache = {}
        self.offsets = (0, 0)
        self.energy = 0
        self.hero_x_world = 0
        self.hero_y_world = FLOOR * SCREEN_H - self.hero_frames[0].height() / 2
        self.camera_x = 0
        self.enemies = []
        self.difficulty = 0
        self._start_time = time.monotonic()
        self.game_over = False
        self.final_seconds = 0
        self._tick_timer = QTimer(self, timeout=self._tick, interval=1000 // FPS)
        self._spawn_timer = QTimer(self, timeout=self._spawn_enemy, interval=self.ENEMY_SPAWN_MS)
        self._tick_timer.start()
        self._spawn_timer.start()

    def resizeEvent(self, event):
        scale = min(self.width() / SCREEN_W, self.height() / SCREEN_H)
        x_off = (self.width() - SCREEN_W * scale) / 2
        y_off = (self.height() - SCREEN_H * scale) / 2
        self.offsets = (x_off, y_off)
        super().resizeEvent(event)

    @pyqtSlot(object)
    def onEnergy(self, val):
        try:
            val = int(val)
        except (TypeError, ValueError):
            return
        self.energy += val
        self.newEnergy.emit(val)

    def _tick(self):
        if self.game_over:
            return
        self._hero_anim_tick += 1
        if self._hero_anim_tick % 8 == 0:
            self.hero_frame_idx = (self.hero_frame_idx + 1) % len(self.hero_frames)
        self.energy = max(0, int(self.energy * 0.985))
        speed = 2 + self.energy / 1000.0
        self.hero_x_world += speed
        for e in self.enemies:
            e.step()
        self.enemies[:] = [e for e in self.enemies if e.x_world - self.camera_x > -e.pix.width()]
        self.camera_x = self.hero_x_world - int(self.HERO_SCREEN_X_FRAC * SCREEN_W)
        hero_pix = self.hero_frames[self.hero_frame_idx]
        x, y, w, h = get_real_rect(qpixmap_to_numpy(hero_pix))
        hero_rect = QRectF(x + self.hero_x_world, y + self.hero_y_world, w, h)
        if any(hero_rect.intersects(e.rect()) for e in self.enemies):
            self._die()
        self.update()

    def _spawn_enemy(self):
        try:
            design_dir = random.choice(self._enemy_design_folders)
            frames = self._enemy_cache.get(design_dir)
            if frames is None:
                frames = self._enemy_cache.setdefault(design_dir, load_anim_frames(design_dir, h=self.CHAR_HEIGHT))
            x = self.hero_x_world - 70 - frames[0].width()
            y = FLOOR * SCREEN_H - frames[0].height() / 2
            self.difficulty += 0.1
            speed = 2 + self.difficulty + random.random() / 30
            self.enemies.append(Enemy(x, y, speed, frames))
        except Exception:
            pass

    def _die(self):
        self.game_over = True
        self.final_seconds = int(time.monotonic() - self._start_time)
        self._tick_timer.stop()
        self._spawn_timer.stop()
        self.update()

    def paintEvent(self, _):
        qp = QPainter(self)
        scale = min(self.width() / SCREEN_W, self.height() / SCREEN_H)
        x_off = (self.width() - SCREEN_W * scale) / 2
        y_off = (self.height() - SCREEN_H * scale) / 2
        qp.fillRect(self.rect(), Qt.black)
        qp.translate(x_off, y_off)
        qp.scale(scale, scale)
        bg_w = self.bg_pix.width()
        start = -int(self.camera_x) % bg_w - bg_w * (int(self.width() / bg_w) + 1)
        for x in range(start, self.width() + bg_w * (int(self.width() / bg_w) + 1), bg_w):
            qp.drawPixmap(x, 0, self.bg_pix)
        for e in self.enemies:
            qp.drawPixmap(int(e.x_world - self.camera_x), int(e.y_world), e.pix)
        hero_screen_x = self.hero_x_world - self.camera_x
        qp.drawPixmap(int(hero_screen_x), int(self.hero_y_world), self.hero_frames[self.hero_frame_idx])
        qp.setPen(Qt.white)
        qp.drawText(20, 30, f"Energy: {self.energy}")
        if self.game_over:
            self._draw_game_over(qp)

    def _draw_game_over(self, qp):
        qp.save()
        qp.resetTransform()
        qp.fillRect(self.rect(), QColor(0, 0, 0, 160))
        qp.restore()
        w, h = 360, 160
        popup = QRectF((SCREEN_W - w) / 2, (SCREEN_H - h) / 2, w, h)
        qp.fillRect(popup, QColor("white"))
        qp.setPen(Qt.black)
        qp.setFont(QFont("Arial", 20, QFont.Bold))
        qp.drawText(popup.adjusted(0, 15, 0, 0), Qt.AlignHCenter, "💀  You Died!  💀")
        qp.setFont(QFont("Arial", 14))
        qp.drawText(popup.adjusted(0, 60, 0, 0), Qt.AlignHCenter, f"Time survived: {self.final_seconds} s")
        qp.setFont(QFont("Arial", 12, QFont.StyleItalic))
        qp.drawText(popup.adjusted(0, 110, 0, 0), Qt.AlignHCenter, "Press the X to quit…")


def main():
    app = QApplication(sys.argv)
    thread = QThread()
    worker = CVWorker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    app.aboutToQuit.connect(worker.stop)
    thread.start()
    gw = GameWidget()
    worker.energyReady.connect(gw.onEnergy)
    gw.setWindowTitle("Endless Runner")
    gw.show()

    def _cleanup():
        if thread.isRunning():
            thread.quit()
            thread.wait()

    app.aboutToQuit.connect(_cleanup)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
