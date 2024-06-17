import cv2
import os
import numpy as np
import torch
import time

from matplotlib import font_manager, rc
from torchvision import models, transforms
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog
from skimage import exposure
import matplotlib.pyplot as plt
from skimage import feature
from torchvision.models.efficientnet import EfficientNet_B7_Weights

# EfficientNet B7 모델 로드
weights = EfficientNet_B7_Weights.DEFAULT
model = models.efficientnet_b7(weights=weights)
model.eval()

# 한글 폰트 설정
font_path = "C:\Windows\Fonts\malgun.ttf"  # Windows 경로 예시
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 이미지 전처리 함수, 입력 크기를 EfficientNet B7에 맞게 조정
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = transform(image)
    tensor_image = tensor_image.unsqueeze(0)

    # PIL 이미지 시각화
    # pil_image = transforms.ToPILImage()(tensor_image.squeeze(0))
    # pil_image.show()

    return image, tensor_image

# HOG 특징 추출 함수
def extract_hog_features(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)

    # HOG 알고리즘 시각화
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    # ax1.axis('off')
    # ax1.imshow(image)
    # ax1.set_title('Original Image')
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    #
    # plt.show()

    return fd, hog_image

# 색상 히스토그램 추출 함수
def color_histogram(image, image_path):

    color_hist = [cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(3)]
    color_hist = np.concatenate(color_hist)
    color_hist = cv2.normalize(color_hist, color_hist).flatten()

    # 파일명 추출
    filename = os.path.basename(image_path)

    # 새 figure 생성
    plt.figure(figsize=(10, 6))

    # RGB 채널별 히스토그램 계산 및 시각화
    # color = ('b', 'g', 'r')
    # for i, col in enumerate(color):
    #     histr = cv2.calcHist([image], [i], None, [256], [0, 256])
    #     plt.plot(histr, color=col)
    #     plt.xlim([0, 256])
    #
    # plt.title(f'Color Histogram - {filename}')
    # plt.xlabel('Pixel Values')
    # plt.ylabel('Frequency')
    # plt.legend(['Blue Channel', 'Green Channel', 'Red Channel'])
    #
    # plt.show()

    return color_hist

# LBP 특성 추출 함수
def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

# Canny edge detector를 사용한 형태 기반 특성 추출 함수
def extract_edge_features(image):
    edges = feature.canny(image, sigma=3)
    edges = edges.astype(np.uint8)  # edges를 uint8 타입으로 변환
    edge_hist, _ = np.histogram(edges.ravel(), bins=2, range=(0, 2), density=True)
    return edge_hist

# 복합 특징 추출 함수
def extract_combined_features(image_path, standard_length=9168):
    image, tensor_image = preprocess_image(image_path)
    with torch.no_grad():
        deep_features = model(tensor_image).cpu().numpy().flatten()

    hog_features, _ = extract_hog_features(image)
    color_hist_features = color_histogram(image, image_path)

    # 그레이스케일로 변환 후 LBP와 Canny edge 특성 추출
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp_features = extract_lbp_features(gray_image)
    edge_features = extract_edge_features(gray_image)

    # 모든 특징 벡터들을 하나의 긴 벡터로 결합
    combined_features = np.concatenate((deep_features, hog_features, color_hist_features, lbp_features, edge_features))

    # 특징 벡터의 길이를 표준화
    if len(combined_features) < standard_length:
        combined_features = np.pad(combined_features, (0, standard_length - len(combined_features)), 'constant')

    combined_features_tensor = torch.tensor(combined_features, dtype=torch.float32)
    return combined_features_tensor

# 유사도 계산 함수
def calculate_similarity(feature1, feature2, weights_tensor):
    if feature1.ndim == 1:
        feature1 = feature1.unsqueeze(0)
    if feature2.ndim == 1:
        feature2 = feature2.unsqueeze(0)

    # Adjusted_weights를 직접 계산하지 않고, 전달받은 weights_tensor를 사용
    weighted_feature1 = feature1 * weights_tensor[:feature1.size(1)]
    weighted_feature2 = feature2 * weights_tensor[:feature2.size(1)]

    return torch.nn.functional.cosine_similarity(weighted_feature1, weighted_feature2, dim=1)

# 이미지 비교 함수
def compare_with_folder(data_folder, upload_folder, standard_length=9168):
    actual_deep_features_dim = 1280
    actual_hog_features_dim = 8192
    actual_color_hist_features_dim = 768
    actual_lbp_features_dim = 10
    actual_edge_features_dim = 2

    deep_feature_weight = 1.0
    hog_feature_weight = 0.3
    color_hist_feature_weight = 0.8
    lbp_feature_weight = 2.0
    edge_feature_weight = 2.0

    weights = np.concatenate([
        np.full(actual_deep_features_dim, deep_feature_weight),
        np.full(actual_hog_features_dim, hog_feature_weight),
        np.full(actual_color_hist_features_dim, color_hist_feature_weight),
        np.full(actual_lbp_features_dim, lbp_feature_weight),
        np.full(actual_edge_features_dim, edge_feature_weight),
    ])

    # Convert numpy array to torch tensor
    weights_tensor = torch.from_numpy(weights).float()

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
                    similarity = calculate_similarity(data_features, upload_features, weights_tensor)
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
