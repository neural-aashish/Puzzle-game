# =========================================================
# ✦ GESTURE PUZZLE (FULL CLEAN SINGLE FILE)
# OpenCV + MediaPipe + Image Puzzle + Fixed Logic
# =========================================================

import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math

# ---------------- CONFIG ----------------
GRID = 3
TILE_SIZE = 140
GAP = 8
OFFSET_X, OFFSET_Y = 80, 80
PINCH_THRESHOLD = 0.05
SMOOTHING = 0.25

# ---------------- HAND TRACKER ----------------
class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.draw = mp.solutions.drawing_utils
        self.prev = None

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        return res.multi_hand_landmarks

    def draw_hand(self, frame, hand):
        self.draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

    def get_cursor(self, hand, frame):
        h, w, _ = frame.shape
        x = int(hand.landmark[8].x * w)
        y = int(hand.landmark[8].y * h)

        if self.prev:
            x = int(self.prev[0] + (x - self.prev[0]) * SMOOTHING)
            y = int(self.prev[1] + (y - self.prev[1]) * SMOOTHING)

        self.prev = (x, y)
        return x, y

    def is_pinch(self, hand):
        t = hand.landmark[4]
        i = hand.landmark[8]
        dist = math.hypot(t.x - i.x, t.y - i.y)
        return dist < PINCH_THRESHOLD


# ---------------- PUZZLE ENGINE ----------------
class Puzzle:
    def __init__(self, image):
        self.grid = GRID
        self.tiles = []
        self.blank = GRID * GRID - 1
        self.build(image)
        self.shuffle()

    def build(self, image):
        img = cv2.resize(image, (GRID * TILE_SIZE, GRID * TILE_SIZE))

        for i in range(GRID * GRID):
            r, c = i // GRID, i % GRID
            crop = img[r*TILE_SIZE:(r+1)*TILE_SIZE,
                       c*TILE_SIZE:(c+1)*TILE_SIZE]
            self.tiles.append(crop)

        self.tiles[self.blank] = None

    def shuffle(self):
        for _ in range(120):
            n = self.neighbors(self.blank)
            s = random.choice(n)
            self.swap(s, self.blank)

    def neighbors(self, idx):
        r, c = idx // GRID, idx % GRID
        out = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < GRID and 0 <= nc < GRID:
                out.append(nr*GRID + nc)
        return out

    def swap(self, a, b):
        self.tiles[a], self.tiles[b] = self.tiles[b], self.tiles[a]
        if self.blank == a:
            self.blank = b
        elif self.blank == b:
            self.blank = a

    def is_adjacent(self, a, b):
        r1, c1 = a//GRID, a%GRID
        r2, c2 = b//GRID, b%GRID
        return abs(r1-r2) + abs(c1-c2) == 1

    def move(self, idx):
        if self.is_adjacent(idx, self.blank):
            self.swap(idx, self.blank)


# ---------------- RENDER ----------------
class Renderer:
    def draw_puzzle(self, frame, puzzle):
        for i in range(GRID * GRID):
            r, c = i // GRID, i % GRID
            x = OFFSET_X + c*(TILE_SIZE+GAP)
            y = OFFSET_Y + r*(TILE_SIZE+GAP)

            tile = puzzle.tiles[i]

            if tile is not None:
                frame[y:y+TILE_SIZE, x:x+TILE_SIZE] = tile

            cv2.rectangle(frame, (x,y),
                          (x+TILE_SIZE,y+TILE_SIZE),
                          (255,255,255),2)

    def draw_cursor(self, frame, cursor, pinch):
        if cursor:
            color = (0,255,255) if pinch else (0,0,255)
            cv2.circle(frame, cursor, 12, color, -1)


# ---------------- UTILS ----------------
def get_tile(x, y):
    x -= OFFSET_X
    y -= OFFSET_Y

    if x < 0 or y < 0:
        return None

    col = x // (TILE_SIZE + GAP)
    row = y // (TILE_SIZE + GAP)

    if row >= GRID or col >= GRID:
        return None

    return row * GRID + col


# ---------------- MAIN APP ----------------
class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.tracker = HandTracker()
        self.renderer = Renderer()

        self.puzzle = None
        self.dragging = None
        self.pinch_prev = False

    def capture(self, frame):
        h, w, _ = frame.shape
        s = min(h, w)
        return frame[0:s, 0:s]

    def run(self):
        while True:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)

            hands = self.tracker.process(frame)

            cursor = None
            pinch = False

            if hands:
                hand = hands[0]
                self.tracker.draw_hand(frame, hand)
                cursor = self.tracker.get_cursor(hand, frame)
                pinch = self.tracker.is_pinch(hand)

            # -------- START / CAPTURE --------
            if self.puzzle is None:
                cv2.putText(frame, "PINCH TO START",
                            (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)

                if pinch and not self.pinch_prev:
                    img = self.capture(frame)
                    self.puzzle = Puzzle(img)

            else:
                tile = None
                if cursor:
                    tile = get_tile(*cursor)

                # pinch start
                if pinch and not self.pinch_prev:
                    if tile is not None:
                        if self.puzzle.is_adjacent(tile, self.puzzle.blank):
                            self.puzzle.move(tile)
                        else:
                            self.dragging = tile

                # release (FIXED)
                elif not pinch and self.pinch_prev:
                    if self.dragging is not None and cursor:
                        drop = get_tile(*cursor)

                        if drop is not None and self.puzzle.is_adjacent(drop, self.puzzle.blank):
                            self.puzzle.move(drop)

                        self.dragging = None

                self.renderer.draw_puzzle(frame, self.puzzle)

            self.renderer.draw_cursor(frame, cursor, pinch)

            self.pinch_prev = pinch

            cv2.imshow("Gesture Puzzle", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


# ---------------- RUN ----------------
if __name__ == "__main__":
    App().run()