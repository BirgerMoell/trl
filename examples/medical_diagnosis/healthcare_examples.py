"""
Example healthcare data and reward functions for GRPO training.
This file contains dummy data and reward functions for training a medical diagnosis model.
"""

from typing import List, Dict, Union, Any
import json
from dataclasses import dataclass
from enum import Enum
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
import torch

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MedicalExample:
    """Single medical example with patient information and diagnosis."""
    prompt: str
    ground_truth: str
    severity: Severity
    age: int
    medical_history: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for the trainer."""
        return {
            "prompt": [
                {"role": "system", "content": "You are a medical diagnostic assistant. Based on the patient information provided, suggest a single most likely diagnosis."},
                {"role": "user", "content": self.prompt}
            ],
            "ground_truth": self.ground_truth,
            "severity": self.severity,
            "age": self.age,
            "medical_history": self.medical_history
        }

# Example medical cases with varying complexity and severity
MEDICAL_EXAMPLES = [
    MedicalExample(
        prompt="Patient presents with fever of 101°F, severe headache, neck stiffness, and sensitivity to light. Symptoms developed over 24 hours.",
        ground_truth="Bacterial Meningitis",
        severity=Severity.CRITICAL,
        age=45,
        medical_history="Generally healthy, no chronic conditions"
    ),
    MedicalExample(
        prompt="Patient reports chronic joint pain in fingers and wrists, morning stiffness lasting >1 hour, and fatigue. Symptoms gradually worsened over 6 months.",
        ground_truth="Rheumatoid Arthritis",
        severity=Severity.MEDIUM,
        age=52,
        medical_history="Family history of autoimmune conditions"
    ),
    MedicalExample(
        prompt="Patient experiencing chest tightness, wheezing, and shortness of breath. Symptoms worsen with exercise and cold weather.",
        ground_truth="Asthma",
        severity=Severity.MEDIUM,
        age=23,
        medical_history="Allergic rhinitis"
    ),
    MedicalExample(
        prompt="Patient presents with sudden onset severe chest pain radiating to left arm, sweating, and nausea. Pain described as crushing.",
        ground_truth="Myocardial Infarction",
        severity=Severity.CRITICAL,
        age=68,
        medical_history="Hypertension, Type 2 Diabetes"
    ),
    MedicalExample(
        prompt="Patient reports persistent cough for 3 weeks, low-grade fever, and mild fatigue. No improvement with over-the-counter medications.",
        ground_truth="Upper Respiratory Tract Infection",
        severity=Severity.LOW,
        age=35,
        medical_history="No significant medical history"
    )
]

def create_dataset() -> List[Dict[str, Any]]:
    """Create a dataset in the format expected by the GRPO trainer."""
    return [example.to_dict() for example in MEDICAL_EXAMPLES]

def calculate_diagnosis_similarity(predicted: str, ground_truth: str) -> float:
    """
    Calculate similarity between predicted and ground truth diagnoses.
    This is a simplified version - in practice, you'd want to use medical ontologies.
    """
    # Convert to lowercase and remove extra spaces
    pred_clean = predicted.lower().strip()
    truth_clean = ground_truth.lower().strip()
    
    # Exact match
    if pred_clean == truth_clean:
        return 1.0
    
    # Simple word overlap similarity
    pred_words = set(pred_clean.split())
    truth_words = set(truth_clean.split())
    
    if not truth_words:
        return 0.0
    
    overlap = len(pred_words.intersection(truth_words))
    similarity = overlap / len(truth_words)
    
    return similarity

def diagnosis_reward_function(
    prompts: List[Any],
    completions: List[Any],
    **kwargs
) -> List[float]:
    """
    Reward function that evaluates the accuracy of medical diagnoses.
    
    Args:
        prompts: List of prompts (each containing patient information)
        completions: List of model completions (predicted diagnoses)
        **kwargs: Additional keyword arguments from the dataset
    
    Returns:
        List of reward scores between 0 and 1
    """
    rewards = []
    
    # Get ground truths and severities from kwargs
    ground_truths = kwargs.get('ground_truth', [])
    severities = kwargs.get('severity', [])
    
    for completion, ground_truth, severity in zip(completions, ground_truths, severities):
        # Extract prediction from completion
        if isinstance(completion, list):  # Conversational format
            predicted_diagnosis = completion[0]["content"]
        else:  # Standard format
            predicted_diagnosis = completion
        
        # Calculate base similarity score
        similarity = calculate_diagnosis_similarity(predicted_diagnosis, ground_truth)
        
        # Apply severity multiplier
        severity_multiplier = {
            Severity.LOW: 1.0,
            Severity.MEDIUM: 1.5,
            Severity.HIGH: 2.0,
            Severity.CRITICAL: 3.0
        }[severity]
        
        # Calculate final reward
        reward = similarity * severity_multiplier
        rewards.append(float(reward))
    
    return rewards

def setup_trainer(
    model_name: str = "Qwen/Qwen2-0.5B-Instruct",
    train_dataset: List[Dict[str, Any]] = None,
    eval_dataset: List[Dict[str, Any]] = None,
    output_dir: str = "medical_model_output"
) -> GRPOTrainer:
    """
    Set up the GRPO trainer with medical diagnosis configuration using Qwen2-0.5B-Instruct.
    
    Args:
        model_name: Name or path of the pretrained model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Directory to save the model and training outputs
    
    Returns:
        Configured GRPOTrainer instance
    """
    # Default to example dataset if none provided
    if train_dataset is None:
        train_dataset = create_dataset()
    if eval_dataset is None:
        eval_dataset = create_dataset()[:2]  # Use first two examples for eval
    
    # Configuration for medical diagnosis task with GRPO
    config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        num_generations=4,  # Number of generations per prompt
        max_prompt_length=512,
        max_completion_length=128,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        max_steps=10000,
        logging_steps=10,
        save_steps=100,
        beta=0.04,  # KL coefficient
        num_iterations=2,  # Number of iterations per batch
        epsilon=0.2,  # Clipping parameter
        temperature=0.7,  # Temperature for generation
        use_vllm=False,  # Not using vLLM for now
        remove_unused_columns=False,  # Keep all columns for reward computation
        seed=42
    )
    
    # Initialize tokenizer with Qwen2 specific settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model with recommended settings for Qwen2
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",  # Qwen2 recommended setting
        device_map="auto",
        trust_remote_code=True
    )
    
    # Initialize trainer with reward function
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=diagnosis_reward_function,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )
    
    return trainer

if __name__ == "__main__":
    # Example usage
    dataset = create_dataset()
    print(f"Created dataset with {len(dataset)} examples")
    print("\nExample data point:")
    print(json.dumps(dataset[0], indent=2))
    
    # Setup trainer
    trainer = setup_trainer(train_dataset=dataset)
    print("\nTrainer setup complete. Ready for training.") 