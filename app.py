import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime

class CyberpunkHandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.animation_counter = 0
        
    def create_glow_effect(self, img, intensity=0.5):
        blur = cv2.GaussianBlur(img, (21, 21), 0)
        return cv2.addWeighted(img, 1, blur, intensity, 0)

    # Dalga çizimi
    def draw_wave_line(self, img, start_point, end_point):
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        points = []
        steps = 80
        for i in range(steps):
            t = i / (steps-1)
            x = start_point[0] + dx*t
            y = start_point[1] + dy*t
            
            offset = math.sin(t * math.pi * 2 + self.animation_counter * 0.1) * 4
            angle = math.atan2(dy, dx) + math.pi/2
            x += math.cos(angle) * offset
            y += math.sin(angle) * offset
            points.append((int(x), int(y)))
        
        # Noktaları birleştir
        if len(points) > 1:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
        
    def draw_tech_circles(self, img, center, radius, color):
        cv2.circle(img, center, radius, color, 1)
        cv2.circle(img, center, radius + 5, color, 1)
        
        for i in range(0, 360, 45):
            start_angle = i
            end_angle = (i + 25) % 360
            
            start_point = (
                int(center[0] + (radius + 10) * math.cos(math.radians(start_angle))),
                int(center[1] + (radius + 10) * math.sin(math.radians(start_angle)))
            )
            end_point = (
                int(center[0] + (radius + 10) * math.cos(math.radians(end_angle))),
                int(center[1] + (radius + 10) * math.sin(math.radians(end_angle)))
            )
            
            cv2.line(img, start_point, end_point, color, 1)
    
    def draw_tech_rectangle(self, img, center, size, color):
        half_size = size // 2
        pt1 = (center[0] - half_size, center[1] - half_size)
        pt2 = (center[0] + half_size, center[1] + half_size)
        cv2.rectangle(img, pt1, pt2, color, 1)
        
        corner_length = size // 3
        cv2.line(img, (pt1[0] - 5, pt1[1]), (pt1[0] + corner_length, pt1[1]), color, 1)
        cv2.line(img, (pt1[0], pt1[1] - 5), (pt1[0], pt1[1] + corner_length), color, 1)
        cv2.line(img, (pt2[0] + 5, pt1[1]), (pt2[0] - corner_length, pt1[1]), color, 1)
        cv2.line(img, (pt2[0], pt1[1] - 5), (pt2[0], pt1[1] + corner_length), color, 1)
        cv2.line(img, (pt1[0] - 5, pt2[1]), (pt1[0] + corner_length, pt2[1]), color, 1)
        cv2.line(img, (pt1[0], pt2[1] + 5), (pt1[0], pt2[1] - corner_length), color, 1)
        cv2.line(img, (pt2[0] + 5, pt2[1]), (pt2[0] - corner_length, pt2[1]), color, 1)
        cv2.line(img, (pt2[0], pt2[1] + 5), (pt2[0], pt2[1] - corner_length), color, 1)
    
    def draw_data_text(self, img, hand_landmarks, text_color=(0, 255, 255)):
        h, w, _ = img.shape
        current_time = datetime.now()
        fps = int(1000000 / (current_time - self.prev_frame_time).microseconds if hasattr(self, 'prev_frame_time') else 0)
        self.prev_frame_time = current_time
        
        index_tip = hand_landmarks.landmark[8]
        x, y = int(index_tip.x * w), int(index_tip.y * h)
        
        cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
        cv2.putText(img, f"X: {x}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
        cv2.putText(img, f"Y: {y}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
    
    def process_frame(self, frame):
        self.animation_counter += 1
        h, w, _ = frame.shape
        output = np.zeros_like(frame)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_coord = []
                for landmark in hand_landmarks.landmark:
                    x_px = min(math.floor(landmark.x * w), w - 1)
                    y_px = min(math.floor(landmark.y * h), h - 1)
                    landmarks_coord.append((x_px, y_px))
                
                # Başparmak-işaret parmağı arası dinamik daire
                thumb_tip = landmarks_coord[4]   # Başparmak ucu
                index_tip = landmarks_coord[8]   # İşaret parmağı ucu
                center_x = (thumb_tip[0] + index_tip[0]) // 2
                center_y = (thumb_tip[1] + index_tip[1]) // 2
                radius = int(math.sqrt((thumb_tip[0] - index_tip[0])**2 + 
                                     (thumb_tip[1] - index_tip[1])**2) / 2)
                cv2.circle(output, (center_x, center_y), radius, (255, 255, 255), 1)
                
                # Parmak bağlantılarını dalgalı çiz
                connections = self.mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if not (start_idx == 4 and end_idx == 8) and not (start_idx == 8 and end_idx == 4):
                        start_point = landmarks_coord[start_idx]
                        end_point = landmarks_coord[end_idx]
                        self.draw_wave_line(output, start_point, end_point)
                
                # Eklem noktaları
                for idx, coord in enumerate(landmarks_coord):
                    cv2.circle(output, coord, 12, (255, 255, 255), 1)  # Dış daire
                
                self.draw_data_text(output, hand_landmarks)
                
                # Diğer bağlantı çizgileri
                for i in range(0, len(landmarks_coord), 4):
                    if i + 3 < len(landmarks_coord):
                        start = landmarks_coord[i]
                        end = landmarks_coord[i + 3]
                        
                        phase = (self.animation_counter * 0.1) % (2 * math.pi)
                        dash_length = int(10 + 5 * math.sin(phase))
                        
                        distance = int(math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2))
                        if distance > 0:
                            dx = (end[0] - start[0]) / distance
                            dy = (end[1] - start[1]) / distance
                            
                            for d in range(0, distance, dash_length * 2):
                                x1 = int(start[0] + d * dx)
                                y1 = int(start[1] + d * dy)
                                x2 = int(min(x1 + dash_length * dx, end[0]))
                                y2 = int(min(y1 + dash_length * dy, end[1]))
                                
                                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        output = self.create_glow_effect(output)
        final_output = cv2.addWeighted(frame, 0.5, output, 0.9, 0)
        
        return final_output

def main():
    cap = cv2.VideoCapture(0)
    tracker = CyberpunkHandTracker()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        processed_frame = tracker.process_frame(frame)
        cv2.imshow('Cyberpunk Hand Tracking', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()