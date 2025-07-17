import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
import os
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class ImageSimilaritySearch:
    def __init__(self, image_dir, index_dir="index_data"):
        self.image_dir = image_dir
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.index_file = os.path.join(index_dir, "faiss.index")
        self.paths_file = os.path.join(index_dir, "image_paths.pkl")

        if os.path.exists(self.index_file) and os.path.exists(self.paths_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.paths_file, "rb") as f:
                self.image_paths = pickle.load(f)
            print(f"Loaded existing index with {len(self.image_paths)} images.")
        else:
            self.index, self.image_paths = self._build_index_from_folder()

    def _extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image)
        return features.squeeze().numpy()

    def _build_index_from_folder(self):
        image_paths = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        features = []
        valid_paths = []

        for path in image_paths:
            try:
                features.append(self._extract_features(path))
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")

        features = np.array(features).astype('float32')
        index = faiss.IndexFlatL2(features.shape[1])
        index.add(features)

        faiss.write_index(index, self.index_file)
        with open(self.paths_file, "wb") as f:
            pickle.dump(valid_paths, f)

        print(f"Indexed {len(valid_paths)} images.")
        return index, valid_paths

    def find_similar(self, query_path, k=5):
        query_vec = self._extract_features(query_path).astype('float32').reshape(1, -1)
        k = min(k, len(self.image_paths))
        distances, indices = self.index.search(query_vec, k)

        return [
            {"path": self.image_paths[idx], "score": 1 / (1 + distances[0][i])}
            for i, idx in enumerate(indices[0])
        ]

    def add_image_to_index(self, image_path):
        if image_path in self.image_paths:
            print(f"Image already indexed: {image_path}")
            return

        features = self._extract_features(image_path).astype('float32').reshape(1, -1)
        self.index.add(features)
        self.image_paths.append(image_path)

        faiss.write_index(self.index, self.index_file)
        with open(self.paths_file, "wb") as f:
            pickle.dump(self.image_paths, f)

        print(f"Added and indexed: {image_path}")


if __name__ == "__main__":
    # Dataset directory
    image_dir = "images"
    searcher = ImageSimilaritySearch(image_dir)

    # 🔍 Query image 
    query_image_path = "my_image.jpg"  

    if not os.path.exists(query_image_path):
        print(f"Query image not found: {query_image_path}")
        exit(1)

    print(f"\nTop 5 similar images to {query_image_path}:\n")
    results = searcher.find_similar(query_image_path, k=5)
    for i, r in enumerate(results):
        print(f"{i+1}. {r['path']} (score: {r['score']:.4f})")

    # Optionally add the query image to the index
    # searcher.add_image_to_index(query_image_path)
