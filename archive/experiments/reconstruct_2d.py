import cv2
import numpy as np
import os
from pathlib import Path

class ChamberReconstructor:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = sorted(Path(image_folder).glob('*.jpg'))
        
    def load_image(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load {path}")
        return img
    
    def detect_green_floor(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return mask
    
    def detect_chamber_corners(self, img):
        mask = self.detect_green_floor(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 10000:
            return None
            
        hull = cv2.convexHull(largest_contour)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) >= 4:
            corners = self.order_corners(approx.reshape(-1, 2))
            return corners
        return None
    
    def order_corners(self, corners):
        if len(corners) < 4:
            return corners
        
        corners = corners.astype(np.float32)
        center = corners.mean(axis=0)
        
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        corners = corners[sorted_indices]
        
        rect = np.zeros((4, 2), dtype=np.float32)
        
        tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]
        
        if tl[1] > bl[1]:
            tl, bl = bl, tl
        if tr[1] > br[1]:
            tr, br = br, tr
            
        rect[0] = tl
        rect[1] = tr
        rect[2] = br
        rect[3] = bl
        
        return rect
    
    def apply_perspective_transform(self, img, corners, target_size=(1000, 1000)):
        if corners is None or len(corners) != 4:
            return None
            
        src_pts = corners.astype(np.float32)
        
        width = int(max(
            np.linalg.norm(src_pts[1] - src_pts[0]),
            np.linalg.norm(src_pts[3] - src_pts[2])
        ))
        height = int(max(
            np.linalg.norm(src_pts[2] - src_pts[1]),
            np.linalg.norm(src_pts[0] - src_pts[3])
        ))
        
        dst_size = (width, height)
        
        dst_pts = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, dst_size)
        
        return warped
    
    def find_best_image(self):
        best_score = 0
        best_img_path = None
        best_corners = None
        
        for img_path in self.images[:20]:
            try:
                img = self.load_image(img_path)
                corners = self.detect_chamber_corners(img)
                
                if corners is not None:
                    area = cv2.contourArea(corners.astype(np.int32))
                    aspect_ratio = self.get_aspect_ratio(corners)
                    score = area / (1 + abs(aspect_ratio - 1))
                    
                    if score > best_score:
                        best_score = score
                        best_img_path = img_path
                        best_corners = corners
                        
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        return best_img_path, best_corners, best_score
    
    def get_aspect_ratio(self, corners):
        width1 = np.linalg.norm(corners[1] - corners[0])
        width2 = np.linalg.norm(corners[2] - corners[3])
        height1 = np.linalg.norm(corners[2] - corners[1])
        height2 = np.linalg.norm(corners[3] - corners[0])
        
        width = max(width1, width2)
        height = max(height1, height2)
        
        return width / height if height > 0 else 1
    
    def create_top_down_view(self, output_path='top_down_view.png'):
        best_img_path, best_corners, score = self.find_best_image()
        
        if best_img_path is None:
            print("Could not detect chamber in any image")
            return None
        
        print(f"Best image: {best_img_path.name}")
        print(f"Detection score: {score:.0f}")
        
        img = self.load_image(best_img_path)
        top_down = self.apply_perspective_transform(img, best_corners)
        
        if top_down is not None:
            cv2.imwrite(output_path, top_down)
            print(f"Saved top-down view to {output_path}")
            
            debug_img = img.copy()
            cv2.drawContours(debug_img, [best_corners.astype(np.int32)], -1, (0, 255, 0), 3)
            cv2.imwrite('detection_debug.png', debug_img)
            
        return top_down
    
    def visualize_comparison(self, top_down):
        if top_down is None:
            return
            
        cv2.imshow('Top-Down View', top_down)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create 2D top-down view of chamber')
    parser.add_argument('--folder', default='splat-files', help='Image folder path')
    parser.add_argument('--output', default='top_down_view.png', help='Output image path')
    
    args = parser.parse_args()
    
    recon = ChamberReconstructor(args.folder)
    top_down = recon.create_top_down_view(args.output)
    
    if top_down is not None:
        print(f"\nTop-down view dimensions: {top_down.shape[1]}x{top_down.shape[0]} pixels")
