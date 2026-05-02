import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.random import sample_without_replacement
import scipy.ndimage
import os
from config import SUPERVISED_MODEL_WEIGHTS
from models import LSTMAutoencoder, ScorePredictor

class BaseSummarizer:
    def __init__(self, data_loader, feature_extractor):
        self.data_loader = data_loader
        self.feature_extractor = feature_extractor
        self.all_features = None

    def _extract_all_features(self, video_id):
        features_list = []
        for batch in self.data_loader.get_video_frames(video_id, batch_size=32):
            batch = self.feature_extractor.preprocess(batch)
            with torch.no_grad():
                features = self.feature_extractor.extract_features(batch)
            features_list.append(features.cpu().numpy())
        all_features = np.concatenate(features_list, axis=0)
        return all_features

class KMeansSummarizer(BaseSummarizer):
    def __init__(self, data_loader, feature_extractor):
        super().__init__(data_loader, feature_extractor)
        self.final_labels = None
        self.final_kmeans = None

    def summarize(self, video_id):
        all_features = self._extract_all_features(video_id)
        self.all_features = all_features

        k_range = range(5, min(21, len(all_features) // 10))
        sample_size = min(1000, len(all_features))
        sample_indices = sample_without_replacement(len(all_features), sample_size, random_state=42)
        sample_features = all_features[sample_indices]

        best_score = -1
        best_k = 5
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(sample_features)
            score = silhouette_score(sample_features, labels)
            if score > best_score:
                best_score = score
                best_k = k

        final_kmeans = KMeans(n_clusters=best_k, random_state=42)
        final_labels = final_kmeans.fit_predict(all_features)
        self.final_labels = final_labels
        self.final_kmeans = final_kmeans

        representative_indices = []
        for cluster_id in range(best_k):
            centroid = final_kmeans.cluster_centers_[cluster_id]
            cluster_indices = np.where(final_labels == cluster_id)[0]
            cluster_features = all_features[cluster_indices]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest_idx = np.argmin(distances)
            representative_indices.append(cluster_indices[closest_idx])

        return sorted(representative_indices), final_labels

class LSTMSummarizer(BaseSummarizer):
    def __init__(self, data_loader, feature_extractor):
        super().__init__(data_loader, feature_extractor)
        self.lstm_model = None
        self.final_labels = None
        self.final_kmeans = None
        self.lstm_features = None

    def _train_lstm(self, features_tensor, epochs=40):
        device = features_tensor.device
        model = LSTMAutoencoder(input_dim=2048).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon, _ = model(features_tensor)
            loss = criterion(recon, features_tensor)
            loss.backward()
            optimizer.step()

        self.lstm_model = model

    def summarize(self, video_id, num_frames=5):
        all_features = self._extract_all_features(video_id)
        self.all_features = all_features
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0).to(device)

        self._train_lstm(features_tensor, epochs=20)
        self.lstm_model.eval()
        with torch.no_grad():
            _, enc_out = self.lstm_model(features_tensor)
        
        lstm_features_np = enc_out.squeeze(0).cpu().numpy()
        self.lstm_features = lstm_features_np

        k_range = range(5, min(21, len(lstm_features_np) // 10))
        sample_size = min(1000, len(lstm_features_np))
        sample_indices = sample_without_replacement(len(lstm_features_np), sample_size, random_state=42)
        sample_features = lstm_features_np[sample_indices]

        best_score = -1
        best_k = 5
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(sample_features)
            score = silhouette_score(sample_features, labels)
            if score > best_score:
                best_score = score
                best_k = k

        final_kmeans = KMeans(n_clusters=best_k, random_state=42)
        final_labels = final_kmeans.fit_predict(lstm_features_np)
        self.final_labels = final_labels
        self.final_kmeans = final_kmeans

        representative_indices = []
        for cluster_id in range(best_k):
            centroid = final_kmeans.cluster_centers_[cluster_id]
            cluster_indices = np.where(final_labels == cluster_id)[0]
            cluster_features = lstm_features_np[cluster_indices]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest_idx = np.argmin(distances)
            representative_indices.append(cluster_indices[closest_idx])

        return sorted(representative_indices), final_labels

class SupervisedSummarizer(BaseSummarizer):
    def __init__(self, data_loader, feature_extractor):
        super().__init__(data_loader, feature_extractor)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ScorePredictor().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.feature_cache = {}
        self.score_cache = {}
        self.device = device

    def precompute_features(self):
        video_ids = self.data_loader.video_names
        for i, vid_id in enumerate(video_ids):
            try:
                frames_iter = self.data_loader.get_video_frames(vid_id, batch_size=32)
                features = []
                for batch in frames_iter:
                    batch = self.feature_extractor.preprocess(batch)
                    with torch.no_grad():
                        feat = self.feature_extractor.extract_features(batch)
                    features.append(feat.cpu())

                if not features: continue
                video_tensor = torch.cat(features)
                self.feature_cache[vid_id] = video_tensor
                raw_anno = self.data_loader.get_video_annotation(vid_id)
                avg_scores = np.mean(raw_anno, axis=0)
                avg_scores = (avg_scores - avg_scores.min()) / (avg_scores.max() - avg_scores.min() + 1e-6)

                if len(avg_scores) != len(video_tensor):
                    zoom_factor = len(video_tensor) / len(avg_scores)
                    avg_scores = scipy.ndimage.zoom(avg_scores, zoom_factor)

                self.score_cache[vid_id] = torch.tensor(avg_scores, dtype=torch.float32)

            except Exception as e:
                print(f"   Error processing {vid_id}: {e}")
                continue

    def train_on_dataset(self, epochs=20, force_retrain=False):
        if os.path.exists(SUPERVISED_MODEL_WEIGHTS) and not force_retrain:
            print(f"Loading saved supervised model weights from {SUPERVISED_MODEL_WEIGHTS}...")
            self.model.load_state_dict(torch.load(SUPERVISED_MODEL_WEIGHTS, map_location=self.device, weights_only=True))
            return

        if not self.feature_cache:
            self.precompute_features()

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            count = 0
            for vid_id in self.feature_cache:
                features = self.feature_cache[vid_id].to(self.device).unsqueeze(0)
                targets = self.score_cache[vid_id].to(self.device).unsqueeze(0)

                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                count += 1
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/max(1, count):.5f}")

        print(f"Saving supervised model weights to {SUPERVISED_MODEL_WEIGHTS}...")
        torch.save(self.model.state_dict(), SUPERVISED_MODEL_WEIGHTS)

    def summarize(self, video_id, num_frames=5):
        self.model.eval()
        if video_id in self.feature_cache:
            video_features = self.feature_cache[video_id].to(self.device).unsqueeze(0)
            self.all_features = self.feature_cache[video_id].numpy()
        else:
            all_features = self._extract_all_features(video_id)
            self.all_features = all_features
            video_features = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scores = self.model(video_features).squeeze().cpu().numpy()

        top_indices = np.argsort(scores)[::-1]
        selected_indices = []
        min_distance = 15
        for idx in top_indices:
            if len(selected_indices) >= num_frames: break
            if all(abs(idx - s) > min_distance for s in selected_indices):
                selected_indices.append(idx)

        return sorted(selected_indices), None
