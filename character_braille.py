import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def image_to_dots(image, spacing=3, dot_size=3):
    h, w = image.shape
    dot_image = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h, spacing):
        for x in range(0, w, spacing):
            if image[y, x] > 0:
                cv2.circle(dot_image, (x, y), dot_size, 255, -1)
    return dot_image

class CharacterContourExtractor:
    def preprocess_image(self, image_path):
        # 이미지를 로드하고 그레이스케일로 변환
        image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Cannot open image file {image_path}")
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 이진화 (Thresholding)하여 누끼 따기
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 누끼를 따기 위해 윤곽선을 찾음
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 최대 윤곽선 (가장 큰 객체) 선택
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        # 마스크 적용하여 배경 제거
        result = cv2.bitwise_and(image, image, mask=mask)

        return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    def extract_contour(self, image):
        # 에지를 찾기 위해 Canny edge detection 사용
        edged = cv2.Canny(image, 50, 150)
        return edged

    def visualize_contour(self, edged, spacing=3, dot_size=3):
        dot_image = image_to_dots(edged, spacing, dot_size)
        plt.figure(figsize=(12, 6))
        plt.imshow(dot_image, cmap='gray')
        plt.title('Contour with Dots')
        plt.show()

if __name__ == "__main__":
    contour_extractor = CharacterContourExtractor()
    data_folder = os.path.join(os.getcwd(), 'data')

    # data 폴더 내의 모든 이미지 파일 처리
    with os.scandir(data_folder) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(".jpg"):
                image_path = os.fsdecode(entry.path)
                image = contour_extractor.preprocess_image(image_path)
                if image is not None:  # 이미지가 올바르게 로드되었는지 확인
                    edged = contour_extractor.extract_contour(image)
                    contour_extractor.visualize_contour(edged)
