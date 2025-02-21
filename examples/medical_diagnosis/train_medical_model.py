"""
Script to train a medical diagnosis model using GRPO.
"""

import os
import logging
from healthcare_examples import create_dataset, setup_trainer, diagnosis_reward_function
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create output directory
    output_dir = "medical_model_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    logger.info("Creating dataset...")
    train_dataset = create_dataset()
    eval_dataset = create_dataset()[:2]  # Use first two examples for evaluation
    
    # Initialize trainer
    logger.info("Setting up trainer...")
    trainer = setup_trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_pretrained(output_dir)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 