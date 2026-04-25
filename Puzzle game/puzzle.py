"""
╔══════════════════════════════════════════════════════════╗
║        GESTURE PUZZLE — Hand-Controlled Sliding Puzzle    ║
║        OpenCV + MediaPipe · Minimal Dark Aesthetic        ║
╚══════════════════════════════════════════════════════════╝
 
Controls:
  ✌️  Both hands visible  → Draw capture frame
  🤏  Pinch (right hand)  → Capture region / Pick+Drop tile
  ☝️  Index finger        → Move cursor
 
Phases:
  IDLE      → Show camera, detect hands
  FRAMING   → Rectangle between both hands
  CAPTURED  → Show puzzle, shuffle animation
  PLAYING   → Gesture-drag tiles to solve
  SOLVED    → Win overlay
"""
 
import cv2
import mediapipe as mp
import numpy as np
import time

import random
import math
 
# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
GRID          = 3          # 3×3 puzzle
TILE_SIZE     = 150        # px per tile (will be overridden dynamically)
TILE_GAP      = 8          # gap between tiles
CORNER_RADIUS = 14         # rounded corners
PUZZLE_OFFSET = (80, 90)   # top-left of puzzle board (will be overridden)
SMOOTH_ALPHA  = 0.25       # landmark smoothing (lower = smoother)
PINCH_THRESH  = 0.055      # normalised distance for pinch
 
# Colour palette  ── dark theme ──────────────────
BG            = (18,  18,  22)
ACCENT        = (120, 220, 255)   # cyan
ACCENT2       = (255, 160, 100)   # amber
TILE_BG       = (38,  40,  50)
TILE_BORDER   = (65,  70,  90)
GLOW_COLOR    = (80, 200, 255)
WIN_COLOR     = (80, 255, 160)
TEXT_COLOR    = (220, 220, 235)
DIM_TEXT      = (100, 105, 120)
 
FONT          = cv2.FONT_HERSHEY_SIMPLEX
 
# ─────────────────────────────────────────────
#  UTILITIES — drawing helpers
# ─────────────────────────────────────────────
 
def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1, alpha=1.0):
    """Draw a filled or outlined rounded rectangle."""
    if alpha < 1.0:
        overlay = img.copy()
        _draw_rounded_rect_on(overlay, x, y, w, h, r, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        _draw_rounded_rect_on(img, x, y, w, h, r, color, thickness)
 
def _draw_rounded_rect_on(img, x, y, w, h, r, color, thickness):
    r = min(r, w // 2, h // 2)
    if thickness == -1:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        cv2.circle(img, (x + r,     y + r),     r, color, -1)
        cv2.circle(img, (x + w - r, y + r),     r, color, -1)
        cv2.circle(img, (x + r,     y + h - r), r, color, -1)
        cv2.circle(img, (x + w - r, y + h - r), r, color, -1)
    else:
        cv2.rectangle(img, (x + r, y), (x + w - r, y), color, thickness)
        cv2.rectangle(img, (x + r, y + h), (x + w - r, y + h), color, thickness)
        cv2.rectangle(img, (x, y + r), (x, y + h - r), color, thickness)
        cv2.rectangle(img, (x + w, y + r), (x + w, y + h - r), color, thickness)
        cv2.ellipse(img, (x + r,     y + r),     (r, r), 180, 0, 90,  color, thickness)
        cv2.ellipse(img, (x + w - r, y + r),     (r, r), 270, 0, 90,  color, thickness)
        cv2.ellipse(img, (x + r,     y + h - r), (r, r), 90,  0, 90,  color, thickness)
        cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0,   0, 90,  color, thickness)
 
def glow_circle(img, cx, cy, r, color, layers=4):
    """Draw a soft glowing circle."""
    for i in range(layers, 0, -1):
        alpha = 0.08 * i
        rad   = r + (layers - i + 1) * 4
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), rad, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.circle(img, (cx, cy), r, color, -1)
    cv2.circle(img, (cx, cy), r - 2, (255, 255, 255), 1)
 
def lerp(a, b, t):
    return a + (b - a) * t
 
def lerp_pt(pa, pb, t):
    return (lerp(pa[0], pb[0], t), lerp(pa[1], pb[1], t))
 
def ease_out(t):
    return 1 - (1 - t) ** 3
 
def put_text_centered(img, text, cx, cy, scale, color, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2), FONT, scale, color, thickness, cv2.LINE_AA)
 
# ─────────────────────────────────────────────
#  HAND TRACKING MODULE
# ─────────────────────────────────────────────
 
class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        # Smoothed landmark positions  {hand_idx: {lm_idx: [x,y]}}
        self._smooth = {}
 
    def process(self, frame_rgb, frame_shape):
        """Returns list of hand dicts with smoothed landmarks."""
        h, w = frame_shape[:2]
        results = self.hands.process(frame_rgb)
        hands_out = []
 
        if not results.multi_hand_landmarks:
            self._smooth = {}
            return hands_out
 
        for idx, (lm_list, handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            label = handedness.classification[0].label  # 'Left' / 'Right'
            if idx not in self._smooth:
                self._smooth[idx] = {}
 
            pts = {}
            for i, lm in enumerate(lm_list.landmark):
                raw_x = lm.x * w
                raw_y = lm.y * h
                if i in self._smooth[idx]:
                    sx, sy = self._smooth[idx][i]
                    sx = lerp(sx, raw_x, SMOOTH_ALPHA)
                    sy = lerp(sy, raw_y, SMOOTH_ALPHA)
                else:
                    sx, sy = raw_x, raw_y
                self._smooth[idx][i] = (sx, sy)
                pts[i] = (int(sx), int(sy))
 
            # Derived metrics
            thumb_tip  = pts[4]
            index_tip  = pts[8]
            pinch_dist = math.dist(
                (lm_list.landmark[4].x, lm_list.landmark[4].y),
                (lm_list.landmark[8].x, lm_list.landmark[8].y)
            )
            is_pinching = pinch_dist < PINCH_THRESH
 
            hands_out.append({
                'label':       label,
                'pts':         pts,
                'thumb_tip':   thumb_tip,
                'index_tip':   index_tip,
                'pinch_dist':  pinch_dist,
                'is_pinching': is_pinching,
                'wrist':       pts[0],
            })
 
        return hands_out
 
    def draw_landmarks(self, img, hands):
        """Render glowing fingertips and subtle skeleton."""
        for hand in hands:
            pts = hand['pts']
            # Skeleton — dim lines
            connections = self.mp_hands.HAND_CONNECTIONS
            for a, b in connections:
                if a in pts and b in pts:
                    cv2.line(img, pts[a], pts[b], (50, 55, 70), 1, cv2.LINE_AA)
            # Fingertips glow
            for tip_id in [4, 8, 12, 16, 20]:
                if tip_id in pts:
                    color = ACCENT if not hand['is_pinching'] else ACCENT2
                    glow_circle(img, pts[tip_id][0], pts[tip_id][1], 6, color, layers=3)
 
# ─────────────────────────────────────────────
#  PUZZLE LOGIC MODULE
# ─────────────────────────────────────────────
 
class Puzzle:
    def __init__(self, image, grid=GRID, canvas_size=(1280, 720)):
        self.grid       = grid
        self.gap        = TILE_GAP

        # Dynamically compute tile size to fill ~82% of canvas
        cw, ch = canvas_size
        usable  = int(min(cw, ch) * 0.82)
        self.tile_size = (usable - TILE_GAP * (grid + 1)) // grid

        # Center puzzle on canvas
        board_px = grid * (self.tile_size + self.gap) + self.gap
        ox = (cw - board_px) // 2
        oy = (ch - board_px) // 2
        self.offset = (ox, oy)

        self.tiles      = []
        self.blank      = None
        self._build(image)
        self.shuffle()
        # Animation state
        self.anim_tiles = {}
        self.shuffle_anim_done = False
        self.dragging   = None
        self.drag_pos   = (0, 0)
        self.hover_tile = None
        self._start_shuffle_anim()
 
    def _build(self, image):
        n    = self.grid
        ts   = self.tile_size
        img  = cv2.resize(image, (n * ts, n * ts))
        self.tiles = []
        for row in range(n):
            for col in range(n):
                idx = row * n + col
                crop = img[row*ts:(row+1)*ts, col*ts:(col+1)*ts].copy()
                # Carve rounded corners mask
                mask = np.zeros((ts, ts), dtype=np.uint8)
                _draw_rounded_rect_on(mask, 0, 0, ts, ts, CORNER_RADIUS, 255, -1)
                crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
                crop_rgba[:, :, 3] = mask
                self.tiles.append({
                    'img':       crop_rgba,
                    'correct':   idx,
                    'current':   idx,
                    'anim_pos':  self._tile_pixel_pos(row, col),
                    'target_pos':self._tile_pixel_pos(row, col),
                })
        # Last tile = blank
        self.blank = n * n - 1
        self.tiles[self.blank]['img'] = None
 
    def _tile_pixel_pos(self, row, col):
        ox, oy = self.offset
        return (ox + col * (self.tile_size + self.gap),
                oy + row * (self.tile_size + self.gap))
 
    def idx_to_rc(self, idx):
        return divmod(idx, self.grid)
 
    def shuffle(self, moves=80):
        """Perform random valid moves to shuffle."""
        for _ in range(moves):
            neighbors = self._blank_neighbors()
            swap = random.choice(neighbors)
            self._swap(swap, self.blank)
 
    def _blank_neighbors(self):
        br, bc = self.idx_to_rc(self.blank)
        nbrs = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            r2, c2 = br + dr, bc + dc
            if 0 <= r2 < self.grid and 0 <= c2 < self.grid:
                nbrs.append(r2 * self.grid + c2)
        return nbrs
 
    def _swap(self, a, b):
        self.tiles[a]['current'], self.tiles[b]['current'] = \
            self.tiles[b]['current'], self.tiles[a]['current']
        # Physically swap tile data
        self.tiles[a], self.tiles[b] = self.tiles[b], self.tiles[a]
        self.blank = b if self.blank == a else a
 
    def _start_shuffle_anim(self):
        """Animate all tiles to their current board positions."""
        n = self.grid
        for i in range(n * n):
            row, col = self.idx_to_rc(i)
            target = self._tile_pixel_pos(row, col)
            self.tiles[i]['target_pos'] = target
            # stagger
            delay_offset = (i * 0.04)
            self.tiles[i]['_anim_start'] = time.time() + delay_offset
            self.tiles[i]['_anim_dur']   = 0.45
 
    def update(self, dt):
        """Advance tile animations."""
        now = time.time()
        for tile in self.tiles:
            start = tile.get('_anim_start', 0)
            dur   = tile.get('_anim_dur', 0.3)
            if now < start:
                continue
            t = min((now - start) / dur, 1.0)
            t = ease_out(t)
            ax, ay = tile['anim_pos']
            tx, ty = tile['target_pos']
            tile['anim_pos'] = (lerp(ax, tx, 0.18), lerp(ay, ty, 0.18))
 
    def move_to(self, tile_idx, animate=True):
        """Slide a tile into blank if adjacent."""
        if not self._is_adjacent(tile_idx, self.blank):
            return False
        blank_target = self.tiles[tile_idx]['target_pos']
        tile_target  = self.tiles[self.blank]['target_pos']
        self._swap(tile_idx, self.blank)
        self.tiles[self.blank]['target_pos']    = blank_target
        other = tile_idx
        self.tiles[other]['target_pos'] = tile_target
        self.tiles[other]['_anim_start'] = time.time()
        self.tiles[other]['_anim_dur']   = 0.22
        return True

    def swap_with(self, src_idx, dst_idx):
        """Swap any two tiles (drag-drop to any slot)."""
        if src_idx == dst_idx:
            return False
        src_target = self.tiles[dst_idx]['target_pos']
        dst_target = self.tiles[src_idx]['target_pos']
        self.tiles[src_idx], self.tiles[dst_idx] = self.tiles[dst_idx], self.tiles[src_idx]
        if self.blank == src_idx:
            self.blank = dst_idx
        elif self.blank == dst_idx:
            self.blank = src_idx
        self.tiles[src_idx]['target_pos']  = src_target
        self.tiles[dst_idx]['target_pos']  = dst_target
        self.tiles[src_idx]['_anim_start'] = time.time()
        self.tiles[src_idx]['_anim_dur']   = 0.28
        self.tiles[dst_idx]['_anim_start'] = time.time()
        self.tiles[dst_idx]['_anim_dur']   = 0.28
        return True

    def nearest_slot(self, px, py):
        """Return board slot index nearest to pixel (px, py)."""
        best_idx, best_dist = None, float('inf')
        for i in range(self.grid * self.grid):
            r, c = self.idx_to_rc(i)
            tx, ty = self._tile_pixel_pos(r, c)
            cx = tx + self.tile_size // 2
            cy = ty + self.tile_size // 2
            d = math.dist((px, py), (cx, cy))
            if d < best_dist:
                best_dist, best_idx = d, i
        return best_idx
 
    def _is_adjacent(self, a, b):
        ar, ac = self.idx_to_rc(a)
        br, bc = self.idx_to_rc(b)
        return abs(ar - br) + abs(ac - bc) == 1
 
    def is_solved(self):
        return all(t['correct'] == t['current'] for t in self.tiles)
 
    def tile_at_pixel(self, px, py):
        """Return tile index under pixel (px,py), or None."""
        n  = self.grid
        ts = self.tile_size
        for i, tile in enumerate(self.tiles):
            if tile['img'] is None:
                continue
            ax, ay = tile['anim_pos']
            if ax <= px <= ax + ts and ay <= py <= ay + ts:
                return i
        return None
 
    def draw(self, canvas, hover_tile=None, drag_idx=None, drag_pos=None, drop_target=None):
        """Render all tiles onto canvas."""
        n  = self.grid
        ts = self.tile_size

        # Board background — exact board size, no extra padding
        bw = n * (ts + self.gap) + self.gap
        bh = bw
        ox, oy = self.offset
        draw_rounded_rect(canvas, ox - 4, oy - 4, bw + 8, bh + 8,
                          24, (25, 27, 35), -1, alpha=0.92)

        # Draw slot placeholders
        for i in range(n * n):
            r, c = self.idx_to_rc(i)
            sx, sy = self._tile_pixel_pos(r, c)
            is_drop = (i == drop_target and drag_idx is not None)
            slot_color = (30, 70, 45) if is_drop else (40, 44, 58)
            draw_rounded_rect(canvas, sx, sy, ts, ts, CORNER_RADIUS, slot_color, -1)
            if is_drop:
                draw_rounded_rect(canvas, sx, sy, ts, ts, CORNER_RADIUS, ACCENT2, thickness=2)

        # Draw all non-dragged tiles
        for i, tile in enumerate(self.tiles):
            if tile['img'] is None:
                continue
            if i == drag_idx:
                continue
            ax, ay = int(tile['anim_pos'][0]), int(tile['anim_pos'][1])
            self._blit_tile(canvas, tile, ax, ay, hover=(i == hover_tile), ts=ts)


    def _blit_tile(self, canvas, tile, ax, ay, hover=False, ts=TILE_SIZE, scale=1.0):
        img_rgba = tile['img']
        h, w = canvas.shape[:2]
 
        if scale != 1.0:
            new_s = int(ts * scale)
            img_rgba = cv2.resize(img_rgba, (new_s, new_s))
            offset   = (new_s - ts) // 2
            ax -= offset
            ay -= offset
            ts_draw = new_s
        else:
            ts_draw = ts
 
        # Clamp to canvas
        x1, y1 = ax, ay
        x2, y2 = ax + ts_draw, ay + ts_draw
        cx1 = max(0, x1); cy1 = max(0, y1)
        cx2 = min(w, x2); cy2 = min(h, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            return
 
        sx1 = cx1 - x1; sy1 = cy1 - y1
        sx2 = sx1 + (cx2 - cx1); sy2 = sy1 + (cy2 - cy1)
 
        roi   = canvas[cy1:cy2, cx1:cx2]
        patch = img_rgba[sy1:sy2, sx1:sx2]
 
        alpha_ch = patch[:, :, 3:4].astype(np.float32) / 255.0
        rgb_src  = patch[:, :, :3].astype(np.float32)
        rgb_dst  = roi.astype(np.float32)
        blended  = (alpha_ch * rgb_src + (1 - alpha_ch) * rgb_dst).astype(np.uint8)
        canvas[cy1:cy2, cx1:cx2] = blended
 
        # Hover border
        if hover:
            draw_rounded_rect(canvas, cx1, cy1, cx2 - cx1, cy2 - cy1,
                              CORNER_RADIUS, ACCENT, thickness=2, alpha=0.9)
 
# ─────────────────────────────────────────────
#  GESTURE CONTROLLER
# ─────────────────────────────────────────────
 
class GestureController:
    """Maps hand data → game actions with debouncing."""
 
    def __init__(self):
        self._pinch_held  = False
        self._pinch_start = 0
        self.PINCH_HOLD   = 0.12   # seconds to confirm pinch
 
    def get_cursor(self, hands):
        """Primary cursor = right-hand index tip."""
        for h in hands:
            if h['label'] == 'Right':
                return h['index_tip']
        if hands:
            return hands[-1]['index_tip']
        return None
 
    def get_pinch(self, hands):
        """Is any hand pinching right now?"""
        for h in hands:
            if h['label'] == 'Right' and h['is_pinching']:
                return True, h['index_tip']
        return False, None
 
    def get_frame_rect(self, hands, frame_shape):
        """Return (x1,y1,x2,y2) rectangle between both hands, or None."""
        if len(hands) < 2:
            return None
        h, w = frame_shape[:2]
        pts = [h_['index_tip'] for h_ in hands]
        x1 = min(pts[0][0], pts[1][0])
        y1 = min(pts[0][1], pts[1][1])
        x2 = max(pts[0][0], pts[1][0])
        y2 = max(pts[0][1], pts[1][1])
        min_size = 80
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return None
        return (x1, y1, x2, y2)
 
# ─────────────────────────────────────────────
#  OVERLAY / UI RENDERER
# ─────────────────────────────────────────────
 
def draw_frame_rect(canvas, rect, t):
    """Draw an animated selection rectangle."""
    x1, y1, x2, y2 = rect
    pulse = 0.5 + 0.5 * math.sin(t * 4)
 
    # Glow layers
    for i in range(4, 0, -1):
        alpha = 0.05 * i * pulse
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1 - i*3, y1 - i*3), (x2 + i*3, y2 + i*3),
                      GLOW_COLOR, 2)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
 
    # Main border
    color = tuple(int(c * (0.75 + 0.25 * pulse)) for c in ACCENT)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
 
    # Corner accents
    length = 20
    thick  = 3
    for cx, cy, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(canvas, (cx, cy), (cx + sx*length, cy), ACCENT2, thick, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy + sy*length), ACCENT2, thick, cv2.LINE_AA)
 
    # Centre label
    cx_mid = (x1 + x2) // 2
    cy_mid = (y1 + y2) // 2
    put_text_centered(canvas, "PINCH TO CAPTURE", cx_mid, cy_mid,
                      0.45, ACCENT, 1)
 
def draw_flash(canvas, alpha):
    """White flash effect on capture."""
    overlay = np.ones_like(canvas, dtype=np.uint8) * 255
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
 
def draw_hud(canvas, phase, fps, hands_count):
    h, w = canvas.shape[:2]
 
    # FPS chip
    fps_text = f"{int(fps)} fps"
    draw_rounded_rect(canvas, w - 80, 12, 68, 26, 8, (30, 32, 42), -1)
    cv2.putText(canvas, fps_text, (w - 72, 30), FONT, 0.45,
                DIM_TEXT, 1, cv2.LINE_AA)
 
    # Phase indicator
    phase_labels = {
        'IDLE':     ('● IDLE',        DIM_TEXT),
        'FRAMING':  ('◈ FRAMING',     ACCENT),
        'CAPTURED': ('✦ PUZZLE READY',ACCENT2),
        'PLAYING':  ('▶ PLAYING',     WIN_COLOR),
        'SOLVED':   ('★ SOLVED',      WIN_COLOR),
    }
    label, color = phase_labels.get(phase, ('', TEXT_COLOR))
    cv2.putText(canvas, label, (16, 34), FONT, 0.55, color, 1, cv2.LINE_AA)
 
    # Hand count dots
    for i in range(2):
        col = ACCENT if i < hands_count else (45, 48, 60)
        cv2.circle(canvas, (16 + i * 18, 52), 5, col, -1, cv2.LINE_AA)
 
def draw_instructions(canvas, phase):
    h, w = canvas.shape[:2]
    lines = {
        'IDLE':     ["Raise both hands", "to start framing"],
        'FRAMING':  ["🤏 Pinch to capture"],
        'CAPTURED': ["Puzzle loading…"],
        'PLAYING':  ["☝️  Cursor  |  🤏 Pick & Drop"],
        'SOLVED':   ["🎉 Puzzle Complete!"],
    }
    msgs = lines.get(phase, [])
    y_base = h - 20 - len(msgs) * 24
    for i, msg in enumerate(msgs):
        put_text_centered(canvas, msg, w // 2, y_base + i * 24,
                          0.50, DIM_TEXT, 1)
 
def draw_solved_overlay(canvas, t, solve_time=0.0):
    h, w = canvas.shape[:2]
    # Dark vignette
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    pulse  = 0.5 + 0.5 * math.sin(t * 3)
    scale  = 1.0 + 0.03 * math.sin(t * 2.5)

    # Popup card
    card_w, card_h = 520, 230
    cx = w // 2
    cy = h // 2
    cx1 = cx - card_w // 2
    cy1 = cy - card_h // 2
    alpha_card = min(1.0, t * 3)
    draw_rounded_rect(canvas, cx1, cy1, card_w, card_h, 28,
                      (20, 24, 32), -1, alpha=alpha_card * 0.92)
    border_col = tuple(int(c * (0.6 + 0.4 * pulse)) for c in WIN_COLOR)
    draw_rounded_rect(canvas, cx1, cy1, card_w, card_h, 28,
                      border_col, thickness=3, alpha=alpha_card)

    put_text_centered(canvas, "PUZZLE  SOLVED!", cx, cy - 60,
                      1.5 * scale, WIN_COLOR, 3)

    mins = int(solve_time) // 60
    secs = int(solve_time) % 60
    ms   = int((solve_time - int(solve_time)) * 100)
    if mins > 0:
        time_str = f"Time: {mins}m {secs:02d}.{ms:02d}s"
    else:
        time_str = f"Solved in  {secs}.{ms:02d}  seconds!"
    put_text_centered(canvas, time_str, cx, cy + 10, 0.9, ACCENT2, 2)

    put_text_centered(canvas, "Raise both hands to play again",
                      cx, cy + 72, 0.55, TEXT_COLOR, 1)


#  GESTURE CONTROLLER
# ─────────────────────────────────────────────
 
class GestureController:
    """Maps hand data → game actions with debouncing."""
 
    def __init__(self):
        self._pinch_held  = False
        self._pinch_start = 0
        self.PINCH_HOLD   = 0.12   # seconds to confirm pinch
 
    def get_cursor(self, hands):
        """Primary cursor = right-hand index tip."""
        for h in hands:
            if h['label'] == 'Right':
                return h['index_tip']
        if hands:
            return hands[-1]['index_tip']
        return None
 
    def get_pinch(self, hands):
        """Is any hand pinching right now?"""
        for h in hands:
            if h['label'] == 'Right' and h['is_pinching']:
                return True, h['index_tip']
        return False, None
 
    def get_frame_rect(self, hands, frame_shape):
        """Return (x1,y1,x2,y2) rectangle between both hands, or None."""
        if len(hands) < 2:
            return None
        h, w = frame_shape[:2]
        pts = [h_['index_tip'] for h_ in hands]
        x1 = min(pts[0][0], pts[1][0])
        y1 = min(pts[0][1], pts[1][1])
        x2 = max(pts[0][0], pts[1][0])
        y2 = max(pts[0][1], pts[1][1])
        min_size = 80
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return None
        return (x1, y1, x2, y2)
 
# ─────────────────────────────────────────────
#  OVERLAY / UI RENDERER
# ─────────────────────────────────────────────
 
def draw_frame_rect(canvas, rect, t):
    """Draw an animated selection rectangle."""
    x1, y1, x2, y2 = rect
    pulse = 0.5 + 0.5 * math.sin(t * 4)
 
    # Glow layers
    for i in range(4, 0, -1):
        alpha = 0.05 * i * pulse
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1 - i*3, y1 - i*3), (x2 + i*3, y2 + i*3),
                      GLOW_COLOR, 2)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
 
    # Main border
    color = tuple(int(c * (0.75 + 0.25 * pulse)) for c in ACCENT)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
 
    # Corner accents
    length = 20
    thick  = 3
    for cx, cy, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(canvas, (cx, cy), (cx + sx*length, cy), ACCENT2, thick, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy + sy*length), ACCENT2, thick, cv2.LINE_AA)
 
    # Centre label
    cx_mid = (x1 + x2) // 2
    cy_mid = (y1 + y2) // 2
    put_text_centered(canvas, "PINCH TO CAPTURE", cx_mid, cy_mid,
                      0.45, ACCENT, 1)
 
def draw_flash(canvas, alpha):
    """White flash effect on capture."""
    overlay = np.ones_like(canvas, dtype=np.uint8) * 255
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
 
def draw_hud(canvas, phase, fps, hands_count):
    h, w = canvas.shape[:2]
 
    # FPS chip
    fps_text = f"{int(fps)} fps"
    draw_rounded_rect(canvas, w - 80, 12, 68, 26, 8, (30, 32, 42), -1)
    cv2.putText(canvas, fps_text, (w - 72, 30), FONT, 0.45,
                DIM_TEXT, 1, cv2.LINE_AA)
 
    # Phase indicator
    phase_labels = {
        'IDLE':     ('● IDLE',        DIM_TEXT),
        'FRAMING':  ('◈ FRAMING',     ACCENT),
        'CAPTURED': ('✦ PUZZLE READY',ACCENT2),
        'PLAYING':  ('▶ PLAYING',     WIN_COLOR),
        'SOLVED':   ('★ SOLVED',      WIN_COLOR),
    }
    label, color = phase_labels.get(phase, ('', TEXT_COLOR))
    cv2.putText(canvas, label, (16, 34), FONT, 0.55, color, 1, cv2.LINE_AA)
 
    # Hand count dots
    for i in range(2):
        col = ACCENT if i < hands_count else (45, 48, 60)
        cv2.circle(canvas, (16 + i * 18, 52), 5, col, -1, cv2.LINE_AA)
 
def draw_instructions(canvas, phase):
    h, w = canvas.shape[:2]
    lines = {
        'IDLE':     ["Raise both hands", "to start framing"],
        'FRAMING':  ["🤏 Pinch to capture"],
        'CAPTURED': ["Puzzle loading…"],
        'PLAYING':  ["☝️  Cursor  |  🤏 Pick & Drop"],
        'SOLVED':   ["🎉 Puzzle Complete!"],
    }
    msgs = lines.get(phase, [])
    y_base = h - 20 - len(msgs) * 24
    for i, msg in enumerate(msgs):
        put_text_centered(canvas, msg, w // 2, y_base + i * 24,
                          0.50, DIM_TEXT, 1)
 
def draw_solved_overlay(canvas, t, solve_time=0.0):
    h, w = canvas.shape[:2]
    # Dark vignette
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    pulse  = 0.5 + 0.5 * math.sin(t * 3)
    scale  = 1.0 + 0.03 * math.sin(t * 2.5)

    # Popup card
    card_w, card_h = 520, 230
    cx = w // 2
    cy = h // 2
    cx1 = cx - card_w // 2
    cy1 = cy - card_h // 2
    alpha_card = min(1.0, t * 3)
    draw_rounded_rect(canvas, cx1, cy1, card_w, card_h, 28,
                      (20, 24, 32), -1, alpha=alpha_card * 0.92)
    border_col = tuple(int(c * (0.6 + 0.4 * pulse)) for c in WIN_COLOR)
    draw_rounded_rect(canvas, cx1, cy1, card_w, card_h, 28,
                      border_col, thickness=3, alpha=alpha_card)

    put_text_centered(canvas, "PUZZLE  SOLVED!", cx, cy - 60,
                      1.5 * scale, WIN_COLOR, 3)

    mins = int(solve_time) // 60
    secs = int(solve_time) % 60
    ms   = int((solve_time - int(solve_time)) * 100)
    if mins > 0:
        time_str = f"Time: {mins}m {secs:02d}.{ms:02d}s"
    else:
        time_str = f"Solved in  {secs}.{ms:02d}  seconds!"
    put_text_centered(canvas, time_str, cx, cy + 10, 0.9, ACCENT2, 2)

    put_text_centered(canvas, "Raise both hands to play again",
                      cx, cy + 72, 0.55, TEXT_COLOR, 1)

#  MAIN APPLICATION
# ─────────────────────────────────────────────
 
class GesturePuzzleApp:
    def __init__(self):
        self.cap     = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
        self.tracker    = HandTracker()
        self.gesture    = GestureController()
        self.puzzle     = None
        self.phase      = 'IDLE'   # IDLE | FRAMING | CAPTURED | PLAYING | SOLVED
 
        self.frame_rect  = None
        self.flash_alpha = 0.0
        self.t           = 0.0
        self.fps_counter = FPSCounter()
 
        # Drag state
        self.dragging_idx = None
        self.drag_pos     = (0, 0)
        self._pinch_was   = False
        self._last_frame_rect = None
 
        # Solved animation
        self._solved_t    = 0.0
        self._play_start  = 0.0   # when PLAYING phase begins
        self._solve_time  = 0.0   # seconds taken to solve
 
    def run(self):
        print("🎮  Gesture Puzzle started. Press Q to quit.")
        prev_time = time.time()
 
        while True:
            ret, raw_frame = self.cap.read()
            if not ret:
                break
 
            raw_frame = cv2.flip(raw_frame, 1)
            now  = time.time()
            dt   = now - prev_time
            prev_time = now
            self.t += dt
            fps = self.fps_counter.tick()
 
            # ── Hand tracking ────────────────────────────────
            rgb    = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            fshape = raw_frame.shape
            hands  = self.tracker.process(rgb, fshape)
 
            # ── Build canvas ─────────────────────────────────
            if self.phase in ('PLAYING', 'SOLVED', 'CAPTURED'):
                # Dark panel background
                canvas = np.full(raw_frame.shape, BG, dtype=np.uint8)
                # Small camera preview top-right
                ph, pw = raw_frame.shape[:2]
                prev_w, prev_h = 260, 146
                small = cv2.resize(raw_frame, (prev_w, prev_h))
                px = pw - prev_w - 20
                py = 70
                draw_rounded_rect(canvas, px - 3, py - 3, prev_w + 6, prev_h + 6,
                                  10, TILE_BORDER, -1)
                canvas[py:py+prev_h, px:px+prev_w] = small
                self.tracker.draw_landmarks(canvas[py:py+prev_h, px:px+prev_w], hands)
            else:
                canvas = raw_frame.copy()
                # Slight dark overlay for readability
                dark = np.zeros_like(canvas)
                cv2.addWeighted(dark, 0.3, canvas, 0.7, 0, canvas)
 
            # ── State machine ─────────────────────────────────
            if self.phase == 'IDLE':
                self._update_idle(canvas, hands, fshape)
 
            elif self.phase == 'FRAMING':
                self._update_framing(canvas, hands, fshape, raw_frame)
 
            elif self.phase == 'CAPTURED':
                self._update_captured(canvas, dt)
 
            elif self.phase == 'PLAYING':
                self._update_playing(canvas, hands, dt)
 
            elif self.phase == 'SOLVED':
                self._update_solved(canvas, hands, dt)
 
            # ── Hand landmarks (camera view phases) ──────────
            if self.phase in ('IDLE', 'FRAMING'):
                self.tracker.draw_landmarks(canvas, hands)
 
            # ── Flash effect ──────────────────────────────────
            if self.flash_alpha > 0:
                draw_flash(canvas, self.flash_alpha)
                self.flash_alpha = max(0.0, self.flash_alpha - dt * 3.5)
 
            # ── HUD ───────────────────────────────────────────
            draw_hud(canvas, self.phase, fps, len(hands))
            draw_instructions(canvas, self.phase)
 
            cv2.imshow("✦ Gesture Puzzle", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
        self.cap.release()
        cv2.destroyAllWindows()
 
    # ── Phase handlers ────────────────────────────────────────
 
    def _update_idle(self, canvas, hands, fshape):
        if len(hands) >= 2:
            self.phase = 'FRAMING'
 
    def _update_framing(self, canvas, hands, fshape, raw_frame):
        if len(hands) < 2:
            self.phase = 'IDLE'
            self.frame_rect = None
            return
 
        rect = self.gesture.get_frame_rect(hands, fshape)
        if rect:
            # Smooth rect transition
            if self._last_frame_rect is None:
                self._last_frame_rect = rect
            else:
                lr = self._last_frame_rect
                self._last_frame_rect = (
                    int(lerp(lr[0], rect[0], 0.3)),
                    int(lerp(lr[1], rect[1], 0.3)),
                    int(lerp(lr[2], rect[2], 0.3)),
                    int(lerp(lr[3], rect[3], 0.3)),
                )
            self.frame_rect = self._last_frame_rect
            draw_frame_rect(canvas, self.frame_rect, self.t)
 
            # Check pinch to capture
            is_pinching, _ = self.gesture.get_pinch(hands)
            if is_pinching and self.frame_rect:
                self._capture_region(raw_frame, self.frame_rect)
 
    def _capture_region(self, raw_frame, rect):
        x1, y1, x2, y2 = rect
        h, w = raw_frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 - x1 < 60 or y2 - y1 < 60:
            return
        cropped = raw_frame[y1:y2, x1:x2]
        side    = min(x2-x1, y2-y1)
        square  = cv2.resize(cropped, (side, side))
        fh, fw = raw_frame.shape[:2]
        self.puzzle     = Puzzle(square, grid=GRID, canvas_size=(fw, fh))
        self.flash_alpha = 0.9
        self.phase       = 'CAPTURED'
        self._captured_start = time.time()
 
    def _update_captured(self, canvas, dt):
        if self.puzzle:
            self.puzzle.update(dt)
            self.puzzle.draw(canvas)
            # Auto-advance to PLAYING after brief pause
            if time.time() - self._captured_start > 1.5:
                self.phase = 'PLAYING'
                self._play_start = time.time()
 
    def _update_playing(self, canvas, hands, dt):
        if not self.puzzle:
            return
        self.puzzle.update(dt)
 
        cursor = self.gesture.get_cursor(hands)
        is_pinching, pinch_pos = self.gesture.get_pinch(hands)
 
        hover = None
        if cursor:
            hover = self.puzzle.tile_at_pixel(*cursor)
 
        # Pinch down - pick tile
        if is_pinching and not self._pinch_was:
            if cursor:
                pick = self.puzzle.tile_at_pixel(*cursor)
                if pick is not None and self.puzzle.tiles[pick]['img'] is not None:
                    self.dragging_idx = pick
                    self.drag_pos = cursor

        elif is_pinching and self._pinch_was:
            # Continue drag
            if self.dragging_idx is not None and cursor:
                self.drag_pos = cursor

        elif not is_pinching and self._pinch_was:
            # Release - swap with nearest slot
            if self.dragging_idx is not None and cursor:
                target_slot = self.puzzle.nearest_slot(*cursor)
                if target_slot is not None and target_slot != self.dragging_idx:
                    self.puzzle.swap_with(self.dragging_idx, target_slot)
                self.dragging_idx = None

        self._pinch_was = is_pinching
 
        # Draw puzzle
        drop_tgt = None
        if self.dragging_idx is not None and cursor:
            drop_tgt = self.puzzle.nearest_slot(*cursor)
        self.puzzle.draw(canvas, hover_tile=hover,
                         drag_idx=self.dragging_idx,
                         drag_pos=self.drag_pos if self.dragging_idx is not None else None,
                         drop_target=drop_tgt)
 
        # Draw elapsed timer top-center
        elapsed = time.time() - self._play_start
        mins_e = int(elapsed) // 60
        secs_e = int(elapsed) % 60
        timer_str = f"{mins_e}:{secs_e:02d}" if mins_e > 0 else f"{secs_e}s"
        ch, cw = canvas.shape[:2]
        put_text_centered(canvas, timer_str, cw // 2, 32, 0.8, ACCENT, 2)

        # Draw cursor
        if cursor:
            color = ACCENT2 if is_pinching else ACCENT
            glow_circle(canvas, cursor[0], cursor[1], 8, color, layers=3)
            if is_pinching:
                cv2.circle(canvas, cursor, 14, ACCENT2, 1, cv2.LINE_AA)
 
        # Check win
        if self.puzzle.is_solved():
            self.phase      = 'SOLVED'
            self._solved_t  = 0.0
            self._solve_time = time.time() - self._play_start
            # Final draw with completed puzzle
            self.puzzle.draw(canvas)
 
    def _update_solved(self, canvas, hands, dt):
        self._solved_t += dt
        if self.puzzle:
            self.puzzle.update(dt)
            self.puzzle.draw(canvas)
        draw_solved_overlay(canvas, self._solved_t, self._solve_time)
        # Restart on two-hand raise
        if len(hands) >= 2:
            self.phase   = 'IDLE'
            self.puzzle  = None
            self.frame_rect = None
            self._last_frame_rect = None
 
 
# ─────────────────────────────────────────────
#  FPS COUNTER
# ─────────────────────────────────────────────
 
class FPSCounter:
    def __init__(self, window=30):
        self._times  = []
        self._window = window
 
    def tick(self):
        self._times.append(time.time())
        self._times = self._times[-self._window:]
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0
 
 
# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════╗
║           ✦  GESTURE PUZZLE  ✦               ║
║                                              ║
║  ✌️  Both hands  → Frame a region            ║
║  🤏  Pinch       → Capture & start puzzle    ║
║  ☝️  Index       → Cursor                    ║
║  🤏  Pinch tile  → Slide it                  ║
║  Q              → Quit                       ║
╚══════════════════════════════════════════════╝
""")
    app = GesturePuzzleApp()
    app.run()
