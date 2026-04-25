"""
GESTURE PUZZLE - Hand-Controlled Sliding Puzzle
OpenCV + MediaPipe

Controls:
  Both hands visible  -> Draw capture frame (stays on camera)
  Pinch (right hand)  -> Capture region
  Pinch a tile        -> Pick it up and drag
  Drop on any tile    -> Swap instantly
  Both hands (SOLVED) -> Restart

Phases: IDLE -> FRAMING -> CAPTURED -> PLAYING -> SOLVED
"""
#--Puzzle Game using Python-----------------------------------------
import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math

# ── Config ─────────────────────────────────────
GRID          = 3
TILE_GAP      = 6
CORNER_R      = 12
SMOOTH_ALPHA  = 0.28
PINCH_THRESH  = 0.050   # lower = need tighter pinch to trigger

# Colours
ACCENT   = (120, 220, 255)
ACCENT2  = (255, 160, 100)
GLOW_C   = (80,  200, 255)
WIN_C    = (80,  255, 160)
TEXT_C   = (220, 220, 235)
DIM_C    = (100, 105, 120)
FONT     = cv2.FONT_HERSHEY_SIMPLEX


# ── Drawing helpers ────────────────────────────

def rrect(img, x, y, w, h, r, color, thick=-1, alpha=1.0):
    r = max(1, min(r, w//2, h//2))
    if alpha < 1.0:
        ov = img.copy()
        _rr(ov, x, y, w, h, r, color, thick)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    else:
        _rr(img, x, y, w, h, r, color, thick)

def _rr(img, x, y, w, h, r, color, t):
    if t == -1:
        cv2.rectangle(img, (x+r, y),   (x+w-r, y+h),   color, -1)
        cv2.rectangle(img, (x,   y+r), (x+w,   y+h-r), color, -1)
        for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.ellipse(img, (x+r,   y+r),   (r,r), 180, 0, 90, color, t)
        cv2.ellipse(img, (x+w-r, y+r),   (r,r), 270, 0, 90, color, t)
        cv2.ellipse(img, (x+r,   y+h-r), (r,r), 90,  0, 90, color, t)
        cv2.ellipse(img, (x+w-r, y+h-r), (r,r), 0,   0, 90, color, t)
        cv2.line(img, (x+r,   y),   (x+w-r, y),   color, t)
        cv2.line(img, (x+r,   y+h), (x+w-r, y+h), color, t)
        cv2.line(img, (x,     y+r), (x,   y+h-r), color, t)
        cv2.line(img, (x+w,   y+r), (x+w, y+h-r), color, t)

def glow_dot(img, cx, cy, r, color, layers=4):
    for i in range(layers, 0, -1):
        ov = img.copy()
        cv2.circle(ov, (cx, cy), r+(layers-i+1)*4, color, -1)
        cv2.addWeighted(ov, 0.07*i, img, 1-0.07*i, 0, img)
    cv2.circle(img, (cx, cy), r, color, -1)
    cv2.circle(img, (cx, cy), r-2, (255,255,255), 1)

def ttext(img, txt, cx, cy, scale, color, thick=1):
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, thick)
    cv2.putText(img, txt, (cx-tw//2, cy+th//2), FONT, scale, color, thick, cv2.LINE_AA)

def lerp(a, b, t):
    return a + (b-a)*t


# ── Hand Tracker ───────────────────────────────

class HandTracker:
    def __init__(self):
        self._mp   = mp.solutions.hands
        self._h    = self._mp.Hands(
            max_num_hands=2,
            min_detection_confidence=0.72,
            min_tracking_confidence=0.65,
        )
        self._sm = {}

    def process(self, rgb, shape):
        H, W = shape[:2]
        res  = self._h.process(rgb)
        out  = []
        if not res.multi_hand_landmarks:
            self._sm = {}
            return out
        for idx, (lms, hns) in enumerate(
                zip(res.multi_hand_landmarks, res.multi_handedness)):
            label = hns.classification[0].label
            if idx not in self._sm:
                self._sm[idx] = {}
            pts = {}
            for i, lm in enumerate(lms.landmark):
                rx, ry = lm.x*W, lm.y*H
                if i in self._sm[idx]:
                    sx, sy = self._sm[idx][i]
                    sx = lerp(sx, rx, SMOOTH_ALPHA)
                    sy = lerp(sy, ry, SMOOTH_ALPHA)
                else:
                    sx, sy = rx, ry
                self._sm[idx][i] = (sx, sy)
                pts[i] = (int(sx), int(sy))
            pd = math.dist(
                (lms.landmark[4].x, lms.landmark[4].y),
                (lms.landmark[8].x, lms.landmark[8].y),
            )
            out.append({
                'label':    label,
                'pts':      pts,
                'index':    pts[8],
                'pinching': pd < PINCH_THRESH,
            })
        return out

    def draw(self, img, hands):
        for hand in hands:
            pts = hand['pts']
            for a, b in self._mp.HAND_CONNECTIONS:
                if a in pts and b in pts:
                    cv2.line(img, pts[a], pts[b], (50,55,70), 1, cv2.LINE_AA)
            for tip in [4, 8, 12, 16, 20]:
                if tip in pts:
                    col = ACCENT2 if hand['pinching'] else ACCENT
                    glow_dot(img, pts[tip][0], pts[tip][1], 6, col, 3)


# ── Puzzle ─────────────────────────────────────

class Puzzle:
    """
    9 tiles, NO blank.
    Board is rendered on camera feed at the exact frame_rect the user drew.
    Drag any tile -> drop on another tile -> they swap.
    """

    def __init__(self, image, frame_rect):
        x1, y1, x2, y2 = frame_rect
        self.bx = x1
        self.by = y1
        self.bw = x2 - x1
        self.bh = y2 - y1
        self.n  = GRID

        # Tile dimensions that perfectly fill the frame
        gx = TILE_GAP * (self.n + 1)
        gy = TILE_GAP * (self.n + 1)
        self.tw = (self.bw - gx) // self.n
        self.th = (self.bh - gy) // self.n

        self.tiles    = []
        self.drag_idx = None
        self.drag_px  = (0, 0)
        self.hover_idx = None

        self._build(image)
        self._shuffle()

    def _slot_pos(self, slot_idx):
        """Top-left pixel of a slot (0..8)."""
        row, col = divmod(slot_idx, self.n)
        x = self.bx + TILE_GAP + col*(self.tw + TILE_GAP)
        y = self.by + TILE_GAP + row*(self.th + TILE_GAP)
        return (x, y)

    def _build(self, image):
        n  = self.n
        tw, th = self.tw, self.th
        img = cv2.resize(image, (tw*n, th*n))
        for i in range(n*n):
            row, col = divmod(i, n)
            crop = img[row*th:(row+1)*th, col*tw:(col+1)*tw].copy()
            mask = np.zeros((th, tw), dtype=np.uint8)
            _rr(mask, 0, 0, tw, th, CORNER_R, 255, -1)
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            rgba[:,:,3] = mask
            sp = list(self._slot_pos(i))
            self.tiles.append({
                'img':        rgba,
                'correct':    i,
                'slot':       i,
                'anim_pos':   [self.bx + self.bw//2, self.by - th],  # start offscreen above
                'target_pos': sp,
                '_anim_start': time.time() + i*0.04,
                '_anim_dur':   0.5,
            })

    def _shuffle(self, moves=150):
        n = self.n
        for _ in range(moves):
            a = random.randint(0, n*n-1)
            b = random.randint(0, n*n-1)
            if a != b:
                self._do_swap(a, b)

    def _do_swap(self, sa, sb):
        """Swap the tiles that currently occupy slots sa and sb."""
        ta = next(t for t in self.tiles if t['slot'] == sa)
        tb = next(t for t in self.tiles if t['slot'] == sb)
        ta['slot'] = sb;  ta['target_pos'] = list(self._slot_pos(sb))
        tb['slot'] = sa;  tb['target_pos'] = list(self._slot_pos(sa))

    def update(self, dt):
        now = time.time()
        for t in self.tiles:
            start = t.get('_anim_start', 0)
            if now < start:
                continue
            ax, ay = t['anim_pos']
            tx, ty = t['target_pos']
            t['anim_pos'] = [lerp(ax, tx, 0.20), lerp(ay, ty, 0.20)]

    def tile_at(self, px, py):
        """Tile list index under pixel, or None."""
        for i, t in enumerate(self.tiles):
            ax, ay = t['anim_pos']
            if ax <= px <= ax+self.tw and ay <= py <= ay+self.th:
                return i
        return None

    def start_drag(self, tile_idx, px, py):
        self.drag_idx = tile_idx
        self.drag_px  = (px, py)

    def update_drag(self, px, py):
        self.drag_px = (px, py)
        h = self.tile_at(px, py)
        self.hover_idx = h if h != self.drag_idx else None

    def drop(self, px, py):
        if self.drag_idx is None:
            return
        target = self.tile_at(px, py)
        if target is not None and target != self.drag_idx:
            ta = self.tiles[self.drag_idx]
            tb = self.tiles[target]
            sa, sb = ta['slot'], tb['slot']
            now = time.time()
            ta['slot'] = sb;  ta['target_pos'] = list(self._slot_pos(sb))
            ta['_anim_start'] = now;  ta['_anim_dur'] = 0.16
            tb['slot'] = sa;  tb['target_pos'] = list(self._slot_pos(sa))
            tb['_anim_start'] = now;  tb['_anim_dur'] = 0.16
        else:
            # snap back
            t = self.tiles[self.drag_idx]
            t['_anim_start'] = time.time(); t['_anim_dur'] = 0.16
        self.drag_idx  = None
        self.hover_idx = None

    def is_solved(self):
        return all(t['slot'] == t['correct'] for t in self.tiles)

    def draw(self, canvas):
        tw, th = self.tw, self.th

        # Board shadow
        rrect(canvas, self.bx-2, self.by-2, self.bw+4, self.bh+4,
              16, (15,17,24), -1, alpha=0.50)

        # Slot outlines
        for i in range(self.n*self.n):
            sx, sy = self._slot_pos(i)
            rrect(canvas, sx, sy, tw, th, CORNER_R, (45,50,65), -1, alpha=0.35)

        # Normal tiles
        for i, t in enumerate(self.tiles):
            if i == self.drag_idx:
                continue
            ax, ay = int(t['anim_pos'][0]), int(t['anim_pos'][1])
            is_h   = (i == self.hover_idx)
            self._blit(canvas, t['img'], ax, ay, tw, th,
                       highlight=is_h, scale=1.05 if is_h else 1.0)

        # Dragged tile (on top, follows finger)
        if self.drag_idx is not None:
            t  = self.tiles[self.drag_idx]
            dx = int(self.drag_px[0] - tw//2)
            dy = int(self.drag_px[1] - th//2)
            rrect(canvas, dx+7, dy+9, tw, th, CORNER_R, (0,0,0), -1, alpha=0.28)
            self._blit(canvas, t['img'], dx, dy, tw, th,
                       highlight=True, scale=1.08)

    def _blit(self, canvas, rgba, ax, ay, tw, th, highlight=False, scale=1.0):
        ch, cw = canvas.shape[:2]
        if scale != 1.0:
            nw = int(tw*scale); nh = int(th*scale)
            rgba = cv2.resize(rgba, (nw, nh))
            ax -= (nw-tw)//2;  ay -= (nh-th)//2
            tw, th = nw, nh

        x1, y1 = ax, ay
        x2, y2 = ax+tw, ay+th
        cx1 = max(0, x1); cy1 = max(0, y1)
        cx2 = min(cw, x2); cy2 = min(ch, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            return
        sx1 = cx1-x1; sy1 = cy1-y1
        patch = rgba[sy1:sy1+(cy2-cy1), sx1:sx1+(cx2-cx1)]
        roi   = canvas[cy1:cy2, cx1:cx2].astype(np.float32)
        a     = patch[:,:,3:4].astype(np.float32)/255.0
        rgb   = patch[:,:,:3].astype(np.float32)
        canvas[cy1:cy2, cx1:cx2] = (a*rgb + (1-a)*roi).astype(np.uint8)

        if highlight:
            rrect(canvas, cx1, cy1, cx2-cx1, cy2-cy1,
                  CORNER_R, ACCENT2, thick=3, alpha=0.95)


# ── Gesture helper ─────────────────────────────

class Gesture:
    def cursor(self, hands):
        for h in hands:
            if h['label'] == 'Right':
                return h['index']
        return hands[-1]['index'] if hands else None

    def pinch(self, hands):
        for h in hands:
            if h['label'] == 'Right' and h['pinching']:
                return True, h['index']
        return False, None

    def frame_rect(self, hands, shape):
        if len(hands) < 2:
            return None
        pts = [h['index'] for h in hands]
        x1 = min(pts[0][0], pts[1][0])
        y1 = min(pts[0][1], pts[1][1])
        x2 = max(pts[0][0], pts[1][0])
        y2 = max(pts[0][1], pts[1][1])
        if (x2-x1) < 70 or (y2-y1) < 70:
            return None
        return (x1, y1, x2, y2)


# ── UI ─────────────────────────────────────────

def draw_frame_ui(canvas, rect, t):
    x1, y1, x2, y2 = rect
    pulse = 0.5 + 0.5*math.sin(t*4)
    for i in range(4, 0, -1):
        ov = canvas.copy()
        cv2.rectangle(ov, (x1-i*3, y1-i*3), (x2+i*3, y2+i*3), GLOW_C, 2)
        cv2.addWeighted(ov, 0.05*i*pulse, canvas, 1-0.05*i*pulse, 0, canvas)
    col = tuple(int(c*(0.65+0.35*pulse)) for c in ACCENT)
    cv2.rectangle(canvas, (x1,y1), (x2,y2), col, 2, cv2.LINE_AA)
    L = 22
    for cx, cy, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(canvas, (cx,cy), (cx+sx*L, cy), ACCENT2, 3, cv2.LINE_AA)
        cv2.line(canvas, (cx,cy), (cx, cy+sy*L), ACCENT2, 3, cv2.LINE_AA)
    ttext(canvas, "PINCH TO CAPTURE", (x1+x2)//2, (y1+y2)//2, 0.5, ACCENT)

def draw_hud(canvas, phase, fps, n_hands):
    h, w = canvas.shape[:2]
    rrect(canvas, w-84, 8, 72, 26, 7, (28,30,40), -1)
    cv2.putText(canvas, f"{int(fps)} fps", (w-78, 26),
                FONT, 0.45, DIM_C, 1, cv2.LINE_AA)
    labels = {
        'IDLE':     ('IDLE',     DIM_C),
        'FRAMING':  ('FRAMING',  ACCENT),
        'CAPTURED': ('READY',    ACCENT2),
        'PLAYING':  ('PLAYING',  WIN_C),
        'SOLVED':   ('SOLVED',   WIN_C),
    }
    lbl, col = labels.get(phase, ('', TEXT_C))
    cv2.putText(canvas, lbl, (14, 30), FONT, 0.55, col, 1, cv2.LINE_AA)
    for i in range(2):
        cv2.circle(canvas, (14+i*18, 48), 5,
                   ACCENT if i < n_hands else (40,44,58), -1, cv2.LINE_AA)

def draw_instructions(canvas, phase):
    h, w = canvas.shape[:2]
    msgs = {
        'IDLE':     ["Raise both hands to start framing"],
        'FRAMING':  ["Pinch to capture puzzle"],
        'CAPTURED': ["Loading..."],
        'PLAYING':  ["Pinch tile, drag to another tile, release to swap"],
        'SOLVED':   ["Raise both hands to restart"],
    }.get(phase, [])
    yb = h - 14 - len(msgs)*22
    for i, m in enumerate(msgs):
        ttext(canvas, m, w//2, yb+i*22, 0.47, DIM_C)

def draw_win_popup(canvas, t, solve_time):
    h, w = canvas.shape[:2]
    ov = canvas.copy()
    cv2.rectangle(ov, (0,0), (w,h), (8,10,16), -1)
    cv2.addWeighted(ov, 0.58, canvas, 0.42, 0, canvas)

    pulse = 0.5 + 0.5*math.sin(t*3)
    scale = 1.0 + 0.025*math.sin(t*2.5)
    fade  = min(1.0, t*2.8)

    cw, ch2 = 560, 250
    cx, cy = w//2, h//2
    rrect(canvas, cx-cw//2, cy-ch2//2, cw, ch2, 28, (18,22,30), -1, alpha=fade*0.94)
    bc = tuple(int(c*(0.5+0.5*pulse)) for c in WIN_C)
    rrect(canvas, cx-cw//2, cy-ch2//2, cw, ch2, 28, bc, thick=3, alpha=fade)

    ttext(canvas, "PUZZLE  SOLVED!", cx, cy-68, 1.5*scale, WIN_C, 3)

    mins = int(solve_time)//60
    secs = int(solve_time)%60
    ms   = int((solve_time-int(solve_time))*100)
    ts   = (f"Time: {mins}m {secs:02d}.{ms:02d}s" if mins
            else f"Solved in  {secs}.{ms:02d}  seconds!")
    ttext(canvas, ts, cx, cy+12, 0.95, ACCENT2, 2)
    ttext(canvas, "Raise both hands to play again", cx, cy+72, 0.54, TEXT_C)


# ── FPS Counter ────────────────────────────────

class FPS:
    def __init__(self, n=30):
        self._t = []; self._n = n
    def tick(self):
        self._t.append(time.time())
        self._t = self._t[-self._n:]
        if len(self._t) < 2: return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t)-1)/span if span else 0.0


# ── Main App ───────────────────────────────────

class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.tracker     = HandTracker()
        self.gesture     = Gesture()
        self.fps         = FPS()
        self.phase       = 'IDLE'
        self.puzzle      = None
        self.t           = 0.0
        self._frame_rect = None
        self._last_fr    = None
        self._flash      = 0.0
        self._pinch_prev = False
        self._play_start = 0.0
        self._solve_time = 0.0
        self._solved_t   = 0.0
        self._cap_start  = 0.0

    def run(self):
        prev = time.time()
        while True:
            ok, raw = self.cap.read()
            if not ok: break
            raw = cv2.flip(raw, 1)
            now = time.time()
            dt  = now - prev; prev = now
            self.t += dt
            fps = self.fps.tick()

            rgb   = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            hands = self.tracker.process(rgb, raw.shape)

            # Canvas = camera feed always (puzzle overlaid on it)
            canvas = raw.copy()

            if   self.phase == 'IDLE':     self._idle(canvas, hands)
            elif self.phase == 'FRAMING':  self._framing(canvas, hands, raw)
            elif self.phase == 'CAPTURED': self._captured(canvas, hands, dt)
            elif self.phase == 'PLAYING':  self._playing(canvas, hands, dt)
            elif self.phase == 'SOLVED':   self._solved_phase(canvas, hands, dt)

            self.tracker.draw(canvas, hands)

            if self._flash > 0:
                ov = np.ones_like(canvas, dtype=np.uint8)*255
                cv2.addWeighted(ov, self._flash, canvas, 1-self._flash, 0, canvas)
                self._flash = max(0.0, self._flash - dt*3.5)

            draw_hud(canvas, self.phase, fps, len(hands))
            draw_instructions(canvas, self.phase)

            cv2.imshow("Gesture Puzzle", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def _idle(self, canvas, hands):
        if len(hands) >= 2:
            self.phase = 'FRAMING'

    def _framing(self, canvas, hands, raw):
        if len(hands) < 2:
            self.phase = 'IDLE'; self._frame_rect = None; return
        rect = self.gesture.frame_rect(hands, raw.shape)
        if rect:
            if self._last_fr is None:
                self._last_fr = rect
            else:
                lr = self._last_fr
                self._last_fr = (
                    int(lerp(lr[0], rect[0], 0.22)),
                    int(lerp(lr[1], rect[1], 0.22)),
                    int(lerp(lr[2], rect[2], 0.22)),
                    int(lerp(lr[3], rect[3], 0.22)),
                )
            self._frame_rect = self._last_fr
            draw_frame_ui(canvas, self._frame_rect, self.t)
            pinching, _ = self.gesture.pinch(hands)
            if pinching and self._frame_rect:
                self._capture(raw, self._frame_rect)

    def _capture(self, raw, rect):
        x1, y1, x2, y2 = rect
        H, W = raw.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        if x2-x1 < 60 or y2-y1 < 60:
            return
        crop = raw[y1:y2, x1:x2]

        # Make puzzle square using shorter side, centred on frame
        side = min(x2-x1, y2-y1)
        cx   = (x1+x2)//2
        cy   = (y1+y2)//2
        fr   = (cx-side//2, cy-side//2, cx+side//2, cy+side//2)

        square = cv2.resize(crop, (side, side))
        self.puzzle      = Puzzle(square, fr)
        self._flash      = 0.80
        self._cap_start  = time.time()
        self.phase       = 'CAPTURED'
        self._pinch_prev = False

    def _captured(self, canvas, hands, dt):
        if self.puzzle:
            self.puzzle.update(dt)
            self.puzzle.draw(canvas)
        if time.time() - self._cap_start > 1.5:
            self.phase       = 'PLAYING'
            self._play_start = time.time()

    def _playing(self, canvas, hands, dt):
        if not self.puzzle: return
        self.puzzle.update(dt)

        cursor   = self.gesture.cursor(hands)
        pinching, _ = self.gesture.pinch(hands)

        if pinching and not self._pinch_prev:
            if cursor:
                ti = self.puzzle.tile_at(*cursor)
                if ti is not None:
                    self.puzzle.start_drag(ti, *cursor)
        elif pinching and self._pinch_prev:
            if cursor and self.puzzle.drag_idx is not None:
                self.puzzle.update_drag(*cursor)
        elif not pinching and self._pinch_prev:
            if cursor:
                self.puzzle.drop(*cursor)

        self._pinch_prev = pinching
        self.puzzle.draw(canvas)

        if cursor:
            col = ACCENT2 if pinching else ACCENT
            glow_dot(canvas, cursor[0], cursor[1], 8, col, 3)
            if pinching:
                cv2.circle(canvas, cursor, 15, ACCENT2, 1, cv2.LINE_AA)

        # Live timer top-center
        elapsed = time.time() - self._play_start
        m, s = int(elapsed)//60, int(elapsed)%60
        ts = f"{m}:{s:02d}" if m else f"{s}s"
        h2, w2 = canvas.shape[:2]
        ttext(canvas, ts, w2//2, 30, 0.85, ACCENT, 2)

        if self.puzzle.is_solved():
            self._solve_time = time.time() - self._play_start
            self.phase       = 'SOLVED'
            self._solved_t   = 0.0

    def _solved_phase(self, canvas, hands, dt):
        self._solved_t += dt
        if self.puzzle:
            self.puzzle.update(dt)
            self.puzzle.draw(canvas)
        draw_win_popup(canvas, self._solved_t, self._solve_time)
        if len(hands) >= 2:
            self.phase = 'IDLE'
            self.puzzle = None
            self._frame_rect = None
            self._last_fr    = None


# ── Entry ──────────────────────────────────────

if __name__ == "__main__":
    print("""
+------------------------------------------+
|          GESTURE PUZZLE                  |
|                                          |
|  Both hands  -> Frame a region           |
|  Pinch       -> Capture puzzle           |
|  Pinch tile  -> Pick up                  |
|  Drag + drop -> Swap with another tile   |
|  Q           -> Quit                     |
+------------------------------------------+
""")
    App().run()
