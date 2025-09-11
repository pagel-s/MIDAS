#!/usr/bin/env python3
"""
Script to precompute text embeddings for the dataset.
This avoids the overhead of computing embeddings on-the-fly during training.
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List
import torch
from tqdm import tqdm

# Add repo to path for imports
import sys
sys.path.append(str(Path(__file__).parent))

from text_embedder import TextEmbeddingModel


def load_text_descriptions(csv_path: Path) -> Dict[str, str]:
    """Load text descriptions from CSV file."""
    descriptions = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            descriptions[row['name']] = row['text']
    return descriptions


def precompute_embeddings(
    text_descriptions: Dict[str, str],
    model_name: str,
    output_path: Path,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """Precompute embeddings for all text descriptions."""
    
    print(f"Loading text embedding model: {model_name}")
    print(f"Using device: {device}")
    
    # Initialize text embedder
    device_obj = torch.device(device)
    embedder = TextEmbeddingModel(model_name, device_obj)
    
    print(f"Embedding dimension: {embedder.embedding_dim}")
    print(f"Number of descriptions: {len(text_descriptions)}")
    
    # Prepare data
    names = list(text_descriptions.keys())
    texts = list(text_descriptions.values())
    
    # Compute embeddings in batches
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        with torch.no_grad():
            batch_embeddings = embedder.encode(batch_texts)
            all_embeddings.append(batch_embeddings.cpu())
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    print(f"Final embeddings shape: {all_embeddings.shape}")
    
    # Save embeddings and mapping
    output_data = {
        'embeddings': all_embeddings.numpy(),
        'names': names,
        'model_name': model_name,
        'embedding_dim': embedder.embedding_dim
    }
    
    np.savez(output_path, **output_data)
    print(f"Saved embeddings to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Precompute text embeddings for training")
    parser.add_argument("--text_csv", type=Path, required=True, 
                       help="CSV file with name,text columns")
    parser.add_argument("--model_name", type=str, 
                       default="GT4SD/multitask-text-and-chemistry-t5-base-standard",
                       help="HuggingFace model name for text embeddings")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output .npz file to save embeddings")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for embedding computation")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu). Auto-detects if not specified")
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load text descriptions
    print(f"Loading text descriptions from: {args.text_csv}")
    text_descriptions = load_text_descriptions(args.text_csv)
    
    # Precompute embeddings
    precompute_embeddings(
        text_descriptions=text_descriptions,
        model_name=args.model_name,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
