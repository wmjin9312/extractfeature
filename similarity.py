import cv2
import os
import numpy as np
import torch
import time
from torchvision import models, transforms
from skimage.feature import hog
from tqdm import tqdm
from skimage import exposure
from torchvision.models.efficientnet import EfficientNet_B7_Weights

# EfficientNet B2 모델 로드
weights = EfficientNet_B7_Weights.DEFAULT
model = models.efficientnet_b7(weights=weights)
model.eval()

# 이미지 전처리 함수, 입력 크기를 EfficientNet B2에 맞게 조정
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 240)),  # EfficientNet B2의 입력 크기로 변경
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = transform(image)
    tensor_image = tensor_image.unsqueeze(0)

    return image, tensor_image

# HOG 특징 추출 함수
def extract_hog_features(image):
    fd, _ = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=2)
    return fd

# 색상 히스토그램 추출 함수
def color_histogram(image):
    hist = [cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(3)]
    hist = np.concatenate(hist)
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# 복합 특징 추출 함수
def extract_combined_features(image_path, standard_length=9168):
    image, tensor_image = preprocess_image(image_path)
    with torch.no_grad():
        deep_features = model(tensor_image).cpu().numpy().flatten()

    hog_features = extract_hog_features(image)
    color_hist_features = color_histogram(image)

    combined_features = np.concatenate((deep_features, hog_features, color_hist_features))
    if len(combined_features) < standard_length:
        combined_features = np.pad(combined_features, (0, standard_length - len(combined_features)), 'constant')

    combined_features_tensor = torch.tensor(combined_features, dtype=torch.float32)
    return combined_features_tensor

# 유사도 계산 함수
def calculate_similarity(feature1, feature2):
    if feature1.ndim == 1:
        feature1 = feature1.unsqueeze(0)
    if feature2.ndim == 1:
        feature2 = feature2.unsqueeze(0)
    return torch.nn.functional.cosine_similarity(feature1, feature2, dim=1)

# 이미지 비교 함수
def compare_with_folder(data_folder, upload_folder, standard_length=9168):
    start_time = time.time()  # 함수 실행 시작 시간을 기록합니다.
    upload_image_name = []
    upload_image_paths = []
    similarities = []

    for filename in os.listdir(upload_folder):
        if filename.lower().endswith(".jpg"):
            upload_image_name.append(filename)  # 이미지 이름 리스트에 추가
            upload_image_paths.append(os.path.join(upload_folder, filename))  # 경로 리스트에 추가

    upload_features_list = [extract_combined_features(upload_path, standard_length) for upload_path in
                            upload_image_paths]

    for data_image_path in tqdm(os.listdir(data_folder), desc="Comparing images..."):
        if data_image_path.lower().endswith(".jpg"):
            data_features = extract_combined_features(os.path.join(data_folder, data_image_path), standard_length)
            for upload_features in upload_features_list:
                if data_features.shape[0] == upload_features.shape[0]:
                    similarity = calculate_similarity(data_features, upload_features)
                    similarities.append((data_image_path, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    print("--------------------------------------------------------")
    print("Top 5 similarities file & similarity")

    for i in range(5):
        filename, similarity = similarities[i]
        top5_similarity_percentage = round(similarity.item(), 2) * 100
        print(f"Image {filename} similarity: {top5_similarity_percentage}%")

    most_similar_image = max(similarities, key=lambda x: x[1])
    print("--------------------------------------------------------")
    print("Compared image:", upload_image_name[0])
    print("Most similar image:", most_similar_image[0])
    most_similarity_percentage = round(most_similar_image[1].item(), 2) * 100
    print(f"Similarity: {most_similarity_percentage}%")
    print("--------------------------------------------------------")

    end_time = time.time()  # 함수 실행 종료 시간을 기록합니다.
    print("Execution time: {:.2f} seconds".format(end_time - start_time))  # 실행에 걸린 시간을 출력합니다.
    print("--------------------------------------------------------")

data_folder = "data"
upload_folder = "upload"

compare_with_folder(data_folder, upload_folder)