import cv2
import os
import numpy as np
import torch
import time
import json
from torchvision import models, transforms
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog
from skimage import feature
from torchvision.models.efficientnet import EfficientNet_B7_Weights
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm

# 한글 폰트 설정
# font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows의 경우
# font_prop = fm.FontProperties(fname=font_path)
# plt.rc('font', family=font_prop.get_name())

class ImageProcessor:
    def __init__(self):
        weights = EfficientNet_B7_Weights.DEFAULT
        self.model = models.efficientnet_b7(weights=weights)
        self.model.eval()

    def preprocess_image(self, image_path):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        encoded_path = image_path.encode('utf-8')
        image = cv2.imdecode(np.fromfile(encoded_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Image not found or unable to read: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor_image = transform(image)
        tensor_image = tensor_image.unsqueeze(0)

        # PIL 이미지 시각화
        # pil_image = transforms.ToPILImage()(tensor_image.squeeze(0))
        # pil_image.show()

        return image, tensor_image

    def extract_hog_features(self, image):
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

    def color_histogram(self, image, image_path):
        color_hist = [cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(3)]
        color_hist = np.concatenate(color_hist)
        color_hist = cv2.normalize(color_hist, color_hist).flatten()

        # RGB 채널별 히스토그램 계산 및 시각화
        #
        # # 파일명 추출
        # filename = os.path.basename(image_path)
        #
        # # 새 figure 생성
        # plt.figure(figsize=(10, 6))
        # color = ('b', 'g', 'r')
        #
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

    def extract_lbp_features(self, image, P=8, R=1):
        lbp = local_binary_pattern(image, P, R, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        return hist

    def extract_edge_features(self, image):
        edges = feature.canny(image, sigma=3)
        edges = edges.astype(np.uint8)
        edge_hist, _ = np.histogram(edges.ravel(), bins=2, range=(0, 2), density=True)

        # 시각화 추가
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        # ax1.axis('off')
        # ax1.imshow(image, cmap=plt.cm.gray)
        # ax1.set_title('Original Image')
        # ax2.axis('off')
        # ax2.imshow(edges, cmap=plt.cm.gray)
        # ax2.set_title('Edges')
        # plt.show()

        return edge_hist

class FeatureExtractor:
    def __init__(self):
        self.processor = ImageProcessor()
        self.standard_length = 9168

    def extract_combined_features(self, image_path):
        image, tensor_image = self.processor.preprocess_image(image_path)
        with torch.no_grad():
            deep_features = self.processor.model(tensor_image).cpu().numpy().flatten()

        hog_features, _ = self.processor.extract_hog_features(image)
        color_hist_features = self.processor.color_histogram(image, image_path)

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp_features = self.processor.extract_lbp_features(gray_image)
        edge_features = self.processor.extract_edge_features(gray_image)

        combined_features = np.concatenate((deep_features, hog_features, color_hist_features, lbp_features, edge_features))

        # 벡터 표준화
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(combined_features.reshape(-1, 1)).flatten()

        if len(combined_features) < self.standard_length:
            combined_features = np.pad(combined_features, (0, self.standard_length - len(combined_features)), 'constant')
            standardized_features = np.pad(standardized_features, (0, self.standard_length - len(standardized_features)), 'constant')

        # 시각화 추가
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # ax1.plot(combined_features)
        # ax1.set_title('Combined Features')
        # ax2.plot(standardized_features)
        # ax2.set_title('Standardized Features')
        # plt.show()

        combined_features_tensor = torch.tensor(combined_features, dtype=torch.float32)
        return combined_features_tensor

class ImageComparer:
    def __init__(self, data_folder, upload_folder):
        self.data_folder = data_folder
        self.upload_folder = upload_folder
        self.extractor = FeatureExtractor()

    def calculate_similarity(self, feature1, feature2, weights_tensor):
        if feature1.ndim == 1:
            feature1 = feature1.unsqueeze(0)
        if feature2.ndim == 1:
            feature2 = feature2.unsqueeze(0)

        weighted_feature1 = feature1 * weights_tensor[:feature1.size(1)]
        weighted_feature2 = feature2 * weights_tensor[:feature2.size(1)]

        return torch.nn.functional.cosine_similarity(weighted_feature1, weighted_feature2, dim=1)

    def compare_with_folder(self):
        weights_tensor = self._prepare_weights()

        start_time = time.time()
        upload_image_paths, upload_image_names = self._get_image_paths(self.upload_folder)
        upload_features_list = [self.extractor.extract_combined_features(upload_path) for upload_path in upload_image_paths]

        data_image_paths, _ = self._get_image_paths(self.data_folder)
        similarities = self._calculate_similarities(data_image_paths, upload_features_list, weights_tensor)

        similarities.sort(key=lambda x: x[1], reverse=True)
        self._save_results(similarities, upload_image_names)
        self._print_results(similarities, upload_image_names, start_time)

        # 벡터 비교 시각화
        self._visualize_feature_comparison(data_image_paths, upload_image_paths, similarities, upload_features_list, weights_tensor)

    def _prepare_weights(self):
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

        return torch.from_numpy(weights).float()

    def _get_image_paths(self, folder):
        image_names = []
        image_paths = []
        for filename in os.listdir(folder):
            if filename.lower().endswith(".jpg"):
                image_names.append(filename)
                image_paths.append(os.path.join(folder, filename))
        return image_paths, image_names

    def _calculate_similarities(self, data_image_paths, upload_features_list, weights_tensor):
        similarities = []
        for data_image_path in tqdm(data_image_paths, desc="Comparing images..."):
            try:
                data_features = self.extractor.extract_combined_features(data_image_path)
                for upload_features in upload_features_list:
                    if data_features.shape[0] == upload_features.shape[0]:
                        similarity = self.calculate_similarity(data_features, upload_features, weights_tensor)
                        similarities.append((os.path.basename(data_image_path), similarity))
            except Exception as e:
                print(f"Error processing {data_image_path}: {e}")
        return similarities

    def _save_results(self, similarities, upload_image_names):
        results = {
            'compared_image': upload_image_names[0] if upload_image_names else None,
            'similarity_top5': [{
                'image': filename,
                'similarity': f"{round(similarity.item(), 2) * 100}%"
            } for filename, similarity in similarities[:5]]
        }

        with open('image_similarity_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def _print_results(self, similarities, upload_image_names, start_time):
        print("Results saved to image_similarity_results.json")

        print("--------------------------------------------------------")
        print("Top 5 similarities file & similarity")

        for i in range(min(5, len(similarities))):
            filename, similarity = similarities[i]
            top5_similarity_percentage = round(similarity.item(), 2) * 100
            print(f"Image {filename} similarity: {top5_similarity_percentage}%")

        if similarities:
            most_similar_image = max(similarities, key=lambda x: x[1])
            print("--------------------------------------------------------")
            print("Compared image:", upload_image_names[0])
            print("Most similar image:", most_similar_image[0])
            most_similarity_percentage = round(most_similar_image[1].item(), 2) * 100
            print("Similarity:", f"{most_similarity_percentage}%")
            print("--------------------------------------------------------")

        end_time = time.time()
        print("Execution time: {:.2f} seconds".format(end_time - start_time))
        print("--------------------------------------------------------")

    def _visualize_feature_comparison(self, data_image_paths, upload_image_paths, similarities, upload_features_list, weights_tensor):
        for upload_path, upload_features in zip(upload_image_paths, upload_features_list):
            data_features_list = []
            for data_image_path in data_image_paths:
                data_features = self.extractor.extract_combined_features(data_image_path)
                data_features_list.append(data_features.numpy())

            fig, ax = plt.subplots(len(data_image_paths), 2, figsize=(12, len(data_image_paths) * 3))

            for i, (data_features, similarity) in enumerate(zip(data_features_list, similarities)):
                ax[i, 0].plot(upload_features.numpy())
                ax[i, 0].set_title(f"Upload Image Features (Similarity: {round(similarity[1].item(), 2) * 100}%)")
                ax[i, 1].plot(data_features)
                ax[i, 1].set_title(f"Data Image Features: {similarity[0]}")

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    data_folder = "data"
    upload_folder = "upload"
    comparer = ImageComparer(data_folder, upload_folder)
    comparer.compare_with_folder()