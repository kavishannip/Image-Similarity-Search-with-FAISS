# Image Similarity Search

A Python-based image similarity search engine that uses deep learning features and efficient vector search to find visually similar images in a dataset.

## Features

- **Deep Learning Feature Extraction**: Uses ResNet-50 pre-trained on ImageNet to extract high-quality image features
- **Efficient Vector Search**: Leverages FAISS (Facebook AI Similarity Search) for fast similarity searches
- **Persistent Index**: Automatically saves and loads the search index for quick startup
- **Incremental Updates**: Add new images to the index without rebuilding from scratch
- **Similarity Scoring**: Provides normalized similarity scores for intuitive ranking

## Requirements

### Dependencies

```bash
pip install torch torchvision pillow numpy faiss-cpu pickle-mixin
```

**Note**: For GPU acceleration, install `faiss-gpu` instead of `faiss-cpu`.

### System Requirements

- Python 3.7+
- At least 4GB RAM (depends on dataset size)
- Storage space for index files (typically 10-20% of image dataset size)

## Installation

1. Clone or download the project files
2. Install required dependencies:
   ```bash
   pip install torch torchvision pillow numpy faiss-cpu
   ```
3. Create your image dataset directory structure:
   ```
   project/
   ├── images/          # Your image dataset
   ├── index_data/      # Will be created automatically
   └── similarity_search.py          # The similarity search script
   ```

## Usage

### Basic Usage

1. **Prepare your dataset**: Place all images in the `images/` directory
2. **Set up query image**: Place your query image (e.g., `my_image.jpg`) in the project root
3. **Run the search**:
   ```bash
   python similarity_search.py
   ```

### Code Example

```python
from image_similarity_search import ImageSimilaritySearch

# Initialize the search engine
searcher = ImageSimilaritySearch("path/to/image/dataset")

# Find similar images
results = searcher.find_similar("query_image.jpg", k=5)

# Print results
for i, result in enumerate(results):
    print(f"{i+1}. {result['path']} (score: {result['score']:.4f})")

# Add a new image to the index
searcher.add_image_to_index("new_image.jpg")
```

### Configuration Options

#### Constructor Parameters

- `image_dir` (str): Path to the directory containing your image dataset
- `index_dir` (str, optional): Directory to store index files (default: "index_data")

#### Search Parameters

- `k` (int): Number of similar images to return (default: 5)

## How It Works

### 1. Feature Extraction
The system uses a pre-trained ResNet-50 model with the final classification layer removed to extract 2048-dimensional feature vectors from images. Images are preprocessed with:
- Resize to 256x256
- Center crop to 224x224
- Normalization with ImageNet statistics

### 2. Index Building
- Features are extracted from all images in the dataset
- A FAISS L2 (Euclidean distance) index is built for efficient similarity search
- The index and image paths are saved for future use

### 3. Similarity Search
- Query image features are extracted using the same process
- FAISS performs efficient k-nearest neighbor search
- Results are ranked by similarity score (higher = more similar)

### 4. Similarity Scoring
The similarity score is calculated as:
```
score = 1 / (1 + distance)
```
This produces scores between 0 and 1, where 1 indicates identical images.

## File Structure

```
project/
├── similarity_search.py                    # Main similarity search script
├── index_data/                # Index storage directory
│   ├── faiss.index           # FAISS vector index
│   └── image_paths.pkl       # Image file paths
├── images/                   # Your image dataset
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── my_image.jpg             # Query image
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)

## Performance Notes

### Initial Index Building
- **Time**: Depends on dataset size and hardware (typically 1-5 seconds per image)
- **Memory**: Requires loading one image at a time plus storing all feature vectors
- **Storage**: Index files are typically 10-20% of the original dataset size

### Search Performance
- **Speed**: Sub-second search times for datasets up to 100K images
- **Memory**: Constant memory usage regardless of dataset size
- **Accuracy**: High-quality results due to deep learning features

## Troubleshooting

### Common Issues

1. **"KMP_DUPLICATE_LIB_OK" Error**:
   - Already handled in the code with `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`

2. **Out of Memory Error**:
   - Reduce batch size or use CPU-only mode
   - Ensure sufficient RAM for your dataset size

3. **No Images Found**:
   - Check that image files are in the correct directory
   - Verify image file extensions are supported

4. **Poor Search Results**:
   - Ensure query image is similar in style/content to dataset images
   - ResNet-50 works best with natural images (not abstract art, etc.)

### Performance Optimization

1. **Use GPU**: Install `faiss-gpu` for faster index building and search
2. **Batch Processing**: For large datasets, consider processing images in batches
3. **Index Tuning**: For very large datasets, consider using approximate indices (IndexIVFFlat)

## Advanced Usage

### Custom Feature Extraction
You can modify the feature extraction by changing the model or preprocessing:

```python
# Use a different model
self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

# Adjust preprocessing
self.transform = transforms.Compose([
    transforms.Resize(512),  # Higher resolution
    transforms.CenterCrop(448),
    # ... rest of transforms
])
```

### Different Distance Metrics
Change the FAISS index type for different similarity measures:

```python
# Cosine similarity
index = faiss.IndexFlatIP(features.shape[1])  # Inner product

# Custom distance
index = faiss.IndexFlatL1(features.shape[1])  # L1 distance
```

