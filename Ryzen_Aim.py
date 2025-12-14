import os
import time
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mss
import numpy as np
import win32api
import win32con
import win32gui
from ultralytics import YOLO

os.environ["OMP_NUM_THREADS"] = "4"

SKELETON_CONNECTIONS = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7)
]
SKELETON_COLORS = (0, 255, 255)
JOINT_COLOR = (0, 0, 255)

@dataclass
class Config:
    MODEL_PATH: str = "best.pt"
    ONNX_PATH: str = "best.onnx"
    
    MODEL_INPUT_SIZE: int = 224 
    
    FOV_SIZE: int = 272 

    PID_KP: float = 0.35 
    PID_KI: float = 0.0 
    PID_KD: float = 0.2 

    DEADZONE_RADIUS: float = 0.0 
    MAX_MOVE_SPEED: int = 60 

    TARGET_LEAD_TIME: float = 0.05 

    CONFIDENCE_THRESHOLD: float = 0.50
    TARGET_CLASS: int = 0

    MOUSE_PRIORITY_RADIUS: float = 130.0
    MOUSE_PRIORITY_WEIGHT: float = 2.0
    BOX_RANDOMNESS: float = 0.0

    SHOW_WINDOW: bool = True
    WINDOW_NAME: str = "Ryzen Tracker v5 - With Speed Prediction"

class MouseAccumulator:
    """Accumulates fractional mouse movements for sub-pixel accuracy."""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0

    def add(self, dx: float, dy: float) -> Tuple[int, int]:
        self.x += dx
        self.y += dy
        ix = int(self.x)
        iy = int(self.y)
        self.x -= ix
        self.y -= iy
        return ix, iy

class RyzenTracker:
    def __init__(self):
        print("[+] Starting Ryzen Tracker v5 - With Speed Prediction")
        self.model = YOLO(Config.ONNX_PATH if os.path.exists(Config.ONNX_PATH) else Config.MODEL_PATH)
        self.sct = mss.mss()
        self.acc = MouseAccumulator()

        w = win32api.GetSystemMetrics(0)
        h = win32api.GetSystemMetrics(1)
        half = Config.FOV_SIZE // 2
        self.area = {"top": h//2 - half, "left": w//2 - half, "width": Config.FOV_SIZE, "height": Config.FOV_SIZE}
        self.cx = Config.FOV_SIZE // 2
        self.cy = Config.FOV_SIZE // 2

        self.target = None
        self.prev_time = time.perf_counter()
        
        self.prev_target_center: Optional[Tuple[float, float]] = None
        self.target_velocity: Tuple[float, float] = (0.0, 0.0)
        
        self.scale_factor = Config.FOV_SIZE / Config.MODEL_INPUT_SIZE
        
    def get_target(self, results):
        """Selects the best target based on proximity to the center and mouse."""
        if not results[0].boxes: return None
        
        boxes_scaled = results[0].boxes.xyxy.cpu().numpy() * self.scale_factor
        confs = results[0].boxes.conf.cpu().numpy()

        mx_global, my_global = win32api.GetCursorPos()

        mx_rel = mx_global - self.area["left"]
        my_rel = my_global - self.area["top"]
        
        is_mouse_in_fov = (0 <= mx_rel < Config.FOV_SIZE) and (0 <= my_rel < Config.FOV_SIZE)
        
        best = None
        best_score = 99999

        for (x1,y1,x2,y2), conf in zip(boxes_scaled, confs):
            if conf < Config.CONFIDENCE_THRESHOLD: continue
            
            cx = (x1 + x2) / 2
            cy = y1 + (y2 - y1) * 0.3 

            dist_to_mouse = math.hypot(cx - mx_rel, cy - my_rel)
            
            score = math.hypot(cx - self.cx, cy - self.cy) 

            if is_mouse_in_fov and dist_to_mouse < Config.MOUSE_PRIORITY_RADIUS:
                weight_factor = (1 - (dist_to_mouse / Config.MOUSE_PRIORITY_RADIUS)) * Config.MOUSE_PRIORITY_WEIGHT
                score = score * (1 - min(weight_factor, 1.0))

            if is_mouse_in_fov and dist_to_mouse < Config.DEADZONE_RADIUS * 2:
                score = 0.0

            if score < best_score:
                best_score = score
                best = (cx, cy, x1, y1, x2, y2)

        return best
    
    def draw_keypoints(self, frame, keypoints_data):
        """Draws the skeleton keypoints and connections on the frame."""
        keypoints = keypoints_data.xy.cpu().numpy() * self.scale_factor 
        
        for kps in keypoints:
            for connection in SKELETON_CONNECTIONS:
                p1_index, p2_index = connection
                p1 = tuple(map(int, kps[p1_index - 1]))
                p2 = tuple(map(int, kps[p2_index - 1]))
                
                if p1[0] > 0 and p2[0] > 0:
                    cv2.line(frame, p1, p2, SKELETON_COLORS, 2)

            for kp in kps:
                x, y = map(int, kp)
                if x > 0 and y > 0:
                    cv2.circle(frame, (x, y), 3, JOINT_COLOR, -1)
        
        return True

    def run(self):
        print("HOLD RIGHT MOUSE BUTTON → AIMBOT ACTIVE")
        print("THIS WORKS IN VALORANT, CS2, APEX, WARZONE, ETC.")
        
        if Config.SHOW_WINDOW:
            cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)
            
            time.sleep(0.1) 
            hwnd = win32gui.FindWindow(None, Config.WINDOW_NAME)
            
            if hwnd:
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        while True:
            img = np.array(self.sct.grab(self.area))
            
            frame = img[:, :, :3].copy() 

            now = time.perf_counter()
            dt = min(now - self.prev_time, 0.1) 
            self.prev_time = now

            resized_frame = cv2.resize(frame, (Config.MODEL_INPUT_SIZE, Config.MODEL_INPUT_SIZE))
            
            results = self.model(resized_frame, verbose=False, imgsz=Config.MODEL_INPUT_SIZE, classes=[Config.TARGET_CLASS])
            
            self.target = self.get_target(results)
            
            keypoints_detected = hasattr(results[0], 'keypoints') and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0

            mx, my = win32api.GetCursorPos()
            rel_mx = mx - self.area["left"]
            rel_my = my - self.area["top"]

            if self.target is not None:
                tx, ty, _, _, _, _ = self.target
                current_target_center = (tx, ty)

                if self.prev_target_center is not None and dt > 0:
                    prev_tx, prev_ty = self.prev_target_center
                    vx = (tx - prev_tx) / dt
                    vy = (ty - prev_ty) / dt
                    self.target_velocity = (vx, vy)
                
                self.prev_target_center = current_target_center
            else:
                self.prev_target_center = None
                self.target_velocity = (0.0, 0.0)

            if win32api.GetKeyState(0x02) < 0 and self.target is not None:
                tx, ty, x1, y1, x2, y2 = self.target

                vx, vy = self.target_velocity
                pred_tx = tx + vx * Config.TARGET_LEAD_TIME 
                pred_ty = ty + vy * Config.TARGET_LEAD_TIME
                
                rand_x = random.uniform(-Config.BOX_RANDOMNESS, Config.BOX_RANDOMNESS) * (x2 - x1)
                rand_y = random.uniform(-Config.BOX_RANDOMNESS, Config.BOX_RANDOMNESS) * (y2 - y1)
                aim_x = pred_tx + rand_x
                aim_y = pred_ty + rand_y

                dx = (aim_x - self.cx) * 1.05
                dy = (aim_y - self.cy) * 1.05

                dist = math.hypot(dx, dy)
                if dist > Config.DEADZONE_RADIUS:
                    
                    move_x = dx * Config.PID_KP 
                    move_y = dy * Config.PID_KP 

                    speed = Config.MAX_MOVE_SPEED
                    if dist < 100: 
                         speed *= (dist / 100) 

                    move_x = np.clip(move_x, -speed, speed)
                    move_y = np.clip(move_y, -speed, speed)

                    mx_move, my_move = self.acc.add(move_x, move_y)
                    if mx_move or my_move:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, mx_move, my_move) 

            if Config.SHOW_WINDOW:
                
                if keypoints_detected:
                    self.draw_keypoints(frame, results[0].keypoints)
                
                if self.target:
                    tx, ty, x1, y1, x2, y2 = self.target
                    vx, vy = self.target_velocity
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    
                    cv2.circle(frame, (int(tx), int(ty)), 8, (0,0,255), -1) 
                    
                    pred_tx = tx + vx * Config.TARGET_LEAD_TIME 
                    pred_ty = ty + vy * Config.TARGET_LEAD_TIME
                    cv2.circle(frame, (int(pred_tx), int(pred_ty)), 8, (0,255,255), 2) 
                    
                    cv2.line(frame, (int(tx), int(ty)), (int(rel_mx), int(rel_my)), (255,0,0), 2)

                cv2.circle(frame, (int(rel_mx), int(rel_my)), 14, (255,255,255), 3)
                cv2.line(frame, (int(rel_mx-18), int(rel_my)), (int(rel_mx+18), int(rel_my)), (255,255,255), 3)
                cv2.line(frame, (int(rel_mx), int(rel_my-18)), (int(rel_mx), int(rel_my+18)), (255,255,255), 3)

                vx, vy = self.target_velocity
                speed_mag = math.hypot(vx, vy)
                
                cv2.putText(frame, "AIMBOT ACTIVE" if win32api.GetKeyState(0x02) < 0 else "HOLD RMB",
                            (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.putText(frame, f"FPS: {int(1/dt)}", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"TGT Speed: {int(speed_mag)} px/s", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
                
                cv2.imshow(Config.WINDOW_NAME, frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    RyzenTracker().run()
