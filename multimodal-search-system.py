import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
from tqdm import tqdm
import faiss
import pickle
import json
from typing import List, Dict, Union, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultimodalSearchSystem:
    """
    A scalable multimodal search and retrieval system that supports 
    text-to-image and image-to-text search functionality.
    """
    
    def __init__(
        self, 
        model_name: str = "ViT-B/32", 
        device: str = None, 
        index_path: str = None
    ):
        """
        Initialize the multimodal search system.
        
        Args:
            model_name: CLIP model variant to use
            device: Device to run the model on ('cuda', 'cpu')
            index_path: Path to load pre-built indexes from
        """
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load the CLIP model
        logger.info(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.embed_dim = self.model.visual.output_dim
        
        # Initialize indexes
        self.image_index = None
        self.text_index = None
        self.image_paths = []
        self.text_documents = []
        
        # Load existing indexes if provided
        if index_path and os.path.exists(index_path):
            self.load_indexes(index_path)
    
    def build_image_index(self, image_directory: str, batch_size: int = 32) -> None:
        """
        Build a searchable index from a directory of images.
        
        Args:
            image_directory: Path to directory containing images
            batch_size: Batch size for processing images
        """
        image_paths = []
        for root, _, files in os.walk(image_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            logger.warning(f"No images found in {image_directory}")
            return
        
        logger.info(f"Building image index with {len(image_paths)} images")
        
        # Create dataset and dataloader
        dataset = ImageDataset(image_paths, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Process images in batches
        image_features = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing images"):
                batch = batch.to(self.device)
                features = self.model.encode_image(batch)
                image_features.append(features.cpu().numpy())
        
        # Concatenate all features
        image_features = np.vstack(image_features)
        
        # Normalize features
        image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
        
        # Build FAISS index
        self.image_index = faiss.IndexFlatIP(self.embed_dim)
        self.image_index.add(image_features.astype('float32'))
        self.image_paths = image_paths
        
        logger.info(f"Image index built successfully with {self.image_index.ntotal} vectors")

    def build_text_index(self, text_data: Union[str, List[str]], batch_size: int = 128) -> None:
        """
        Build a searchable index from text data.
        
        Args:
            text_data: Either a list of text documents or path to a JSON file
            batch_size: Batch size for processing text
        """
        # Handle text data input
        if isinstance(text_data, str) and os.path.exists(text_data):
            with open(text_data, 'r', encoding='utf-8') as f:
                if text_data.endswith('.json'):
                    text_documents = json.load(f)
                else:
                    text_documents = f.read().splitlines()
        elif isinstance(text_data, list):
            text_documents = text_data
        else:
            raise ValueError("text_data must be either a file path or a list of strings")
        
        if not text_documents:
            logger.warning("No text documents provided")
            return
        
        logger.info(f"Building text index with {len(text_documents)} documents")
        
        # Process text in batches
        text_features = []
        for i in tqdm(range(0, len(text_documents), batch_size), desc="Processing text"):
            batch = text_documents[i:i+batch_size]
            text_tokens = clip.tokenize(batch).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_text(text_tokens)
                text_features.append(features.cpu().numpy())
        
        # Concatenate all features
        text_features = np.vstack(text_features)
        
        # Normalize features
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        
        # Build FAISS index
        self.text_index = faiss.IndexFlatIP(self.embed_dim)
        self.text_index.add(text_features.astype('float32'))
        self.text_documents = text_documents
        
        logger.info(f"Text index built successfully with {self.text_index.ntotal} vectors")

    def text_to_image_search(
        self, 
        query_text: str, 
        k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Search for images matching the given text query.
        
        Args:
            query_text: The text query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing image paths and similarity scores
        """
        if self.image_index is None or not self.image_paths:
            raise RuntimeError("Image index not built. Call build_image_index first.")
        
        logger.info(f"Performing text-to-image search for: '{query_text}'")
        
        # Encode the query text
        with torch.no_grad():
            text_features = self.model.encode_text(clip.tokenize([query_text]).to(self.device))
            text_features = text_features.cpu().numpy()
        
        # Normalize features
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        
        # Search the index
        k = min(k, len(self.image_paths))
        scores, indices = self.image_index.search(text_features.astype('float32'), k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            results.append({
                "image_path": self.image_paths[idx],
                "similarity": float(score),
                "rank": i + 1
            })
        
        return results
    
    def image_to_text_search(
        self, 
        image_path: str, 
        k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Search for text documents matching the given image.
        
        Args:
            image_path: Path to the query image
            k: Number of results to return
            
        Returns:
            List of dictionaries containing text documents and similarity scores
        """
        if self.text_index is None or not self.text_documents:
            raise RuntimeError("Text index not built. Call build_text_index first.")
        
        logger.info(f"Performing image-to-text search for image: {image_path}")
        
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode the image
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features.cpu().numpy()
        
        # Normalize features
        image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
        
        # Search the index
        k = min(k, len(self.text_documents))
        scores, indices = self.text_index.search(image_features.astype('float32'), k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            results.append({
                "text": self.text_documents[idx],
                "similarity": float(score),
                "rank": i + 1
            })
        
        return results
    
    def semantic_search(
        self, 
        query: str, 
        mode: str = "text_to_image", 
        k: int = 5
    ) -> List[Dict]:
        """
        Unified search interface that dispatches to the appropriate search method.
        
        Args:
            query: The search query (text or image path)
            mode: Search mode ('text_to_image' or 'image_to_text')
            k: Number of results to return
            
        Returns:
            List of search results
        """
        if mode == "text_to_image":
            return self.text_to_image_search(query, k)
        elif mode == "image_to_text":
            return self.image_to_text_search(query, k)
        else:
            raise ValueError("Mode must be either 'text_to_image' or 'image_to_text'")
    
    def save_indexes(self, output_dir: str) -> None:
        """
        Save the built indexes to disk.
        
        Args:
            output_dir: Directory to save indexes
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save image index
        if self.image_index is not None:
            image_index_path = os.path.join(output_dir, "image_index.bin")
            faiss.write_index(self.image_index, image_index_path)
            
            # Save image paths
            with open(os.path.join(output_dir, "image_paths.pkl"), 'wb') as f:
                pickle.dump(self.image_paths, f)
                
            logger.info(f"Image index saved to {image_index_path}")
        
        # Save text index
        if self.text_index is not None:
            text_index_path = os.path.join(output_dir, "text_index.bin")
            faiss.write_index(self.text_index, text_index_path)
            
            # Save text documents
            with open(os.path.join(output_dir, "text_documents.pkl"), 'wb') as f:
                pickle.dump(self.text_documents, f)
                
            logger.info(f"Text index saved to {text_index_path}")
    
    def load_indexes(self, index_dir: str) -> None:
        """
        Load indexes from disk.
        
        Args:
            index_dir: Directory containing saved indexes
        """
        # Load image index
        image_index_path = os.path.join(index_dir, "image_index.bin")
        if os.path.exists(image_index_path):
            self.image_index = faiss.read_index(image_index_path)
            
            # Load image paths
            with open(os.path.join(index_dir, "image_paths.pkl"), 'rb') as f:
                self.image_paths = pickle.load(f)
                
            logger.info(f"Image index loaded with {self.image_index.ntotal} vectors")
        
        # Load text index
        text_index_path = os.path.join(index_dir, "text_index.bin")
        if os.path.exists(text_index_path):
            self.text_index = faiss.read_index(text_index_path)
            
            # Load text documents
            with open(os.path.join(index_dir, "text_documents.pkl"), 'rb') as f:
                self.text_documents = pickle.load(f)
                
            logger.info(f"Text index loaded with {self.text_index.ntotal} vectors")


class ImageDataset(Dataset):
    """
    Dataset for loading and preprocessing images.
    """
    
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            return self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a blank image in case of error
            blank = Image.new("RGB", (224, 224), (0, 0, 0))
            return self.transform(blank)


class MultimodalSearchAPI:
    """
    API wrapper for the multimodal search system.
    """
    
    def __init__(self, search_system=None, index_dir=None):
        """
        Initialize the API with either an existing search system or by creating a new one.
        
        Args:
            search_system: Existing MultimodalSearchSystem instance
            index_dir: Directory containing saved indexes
        """
        if search_system:
            self.search_system = search_system
        else:
            self.search_system = MultimodalSearchSystem(index_path=index_dir)
    
    def initialize_indexes(self, image_dir=None, text_data=None):
        """
        Initialize search indexes.
        
        Args:
            image_dir: Directory containing images
            text_data: Text data for indexing
        """
        if image_dir:
            self.search_system.build_image_index(image_dir)
        
        if text_data:
            self.search_system.build_text_index(text_data)
    
    def search(self, query, mode="text_to_image", k=5):
        """
        Perform a search.
        
        Args:
            query: Search query (text or image path)
            mode: Search mode ('text_to_image' or 'image_to_text')
            k: Number of results to return
            
        Returns:
            List of search results
        """
        return self.search_system.semantic_search(query, mode, k)


def main():
    """
    Example usage of the multimodal search system.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal Search System")
    parser.add_argument("--action", choices=["index", "search"], required=True,
                      help="Action to perform: index or search")
    parser.add_argument("--image_dir", type=str, help="Directory containing images to index")
    parser.add_argument("--text_data", type=str, help="Path to text data file")
    parser.add_argument("--index_dir", type=str, default="./indexes",
                      help="Directory to save/load indexes")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--mode", choices=["text_to_image", "image_to_text"], 
                      default="text_to_image", help="Search mode")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    # Create API
    api = MultimodalSearchAPI(index_dir=args.index_dir if os.path.exists(args.index_dir) else None)
    
    if args.action == "index":
        # Build indexes
        api.initialize_indexes(image_dir=args.image_dir, text_data=args.text_data)
        
        # Save indexes
        os.makedirs(args.index_dir, exist_ok=True)
        api.search_system.save_indexes(args.index_dir)
        
    elif args.action == "search":
        if not args.query:
            parser.error("--query is required for search action")
        
        # Perform search
        results = api.search(args.query, args.mode, args.k)
        
        # Print results
        print(f"\nSearch results for '{args.query}':")
        for result in results:
            if args.mode == "text_to_image":
                print(f"Rank {result['rank']}: {result['image_path']} (Score: {result['similarity']:.4f})")
            else:
                print(f"Rank {result['rank']}: '{result['text'][:100]}...' (Score: {result['similarity']:.4f})")


if __name__ == "__main__":
    main()
