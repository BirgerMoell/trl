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

def create_medical_dataset():
    """
    Create a dataset of medical cases for training.
    
    Returns:
        List[Dict]: List of medical cases with prompts and expected diagnoses
    """
    medical_cases = [
        {
            "prompt": "Patient presents with fever, cough, and difficulty breathing for the past 3 days. Chest X-ray shows patchy infiltrates.",
            "diagnosis": "Pneumonia",
            "symptoms": ["fever", "cough", "dyspnea"],
            "test_results": ["patchy infiltrates on chest X-ray"]
        },
        {
            "prompt": "65-year-old patient with sudden onset chest pain, radiating to left arm, sweating, and nausea.",
            "diagnosis": "Myocardial Infarction",
            "symptoms": ["chest pain", "radiation to left arm", "sweating", "nausea"],
            "test_results": []
        },
        {
            "prompt": "Patient reports severe headache, neck stiffness, and sensitivity to light. Temperature 39.5°C.",
            "diagnosis": "Meningitis",
            "symptoms": ["severe headache", "neck stiffness", "photophobia", "fever"],
            "test_results": ["elevated temperature"]
        }
    ]
    
    # Convert to dataset format
    dataset = []
    for case in medical_cases:
        dataset.append({
            "prompt": f"Medical Case:\n{case['prompt']}\n\nBased on the above information, what is the most likely diagnosis? Please explain your reasoning.",
            "completion": f"After analyzing the symptoms and test results, the most likely diagnosis is {case['diagnosis']}.",
            "diagnosis": case["diagnosis"]
        })
    
    return dataset

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

def setup_trainer(output_dir: str = "medical-diagnosis-model"):
    """
    Set up the GRPO trainer for medical diagnosis.
    
    Args:
        output_dir: Directory to save the model outputs
        
    Returns:
        GRPOTrainer: Configured trainer ready for medical diagnosis tasks
    """
    # Configuration for quick test run
    config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1.41e-5,
        per_device_train_batch_size=2,  # Small batch size
        gradient_accumulation_steps=1,   # No gradient accumulation
        max_steps=2,                     # Only run 2 steps
        logging_steps=1,                 # Log every step
        save_steps=2,                    # Save at the end
        max_prompt_length=128,           # Shorter sequences
        max_completion_length=64,        # Shorter completions
        num_generations=2,               # Minimal generations
        beta=0.04,
        temperature=0.6,
        use_vllm=False,
        remove_unused_columns=False,
        seed=42,
        report_to="none"
    )

    # Initialize tokenizer and model
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=diagnosis_reward_function,
        args=config,
        train_dataset=create_medical_dataset(),
        processing_class=tokenizer
    )
    
    return trainer

if __name__ == "__main__":
    # Example usage
    dataset = create_medical_dataset()
    print(f"Created dataset with {len(dataset)} examples")
    print("\nExample data point:")
    print(json.dumps(dataset[0], indent=2))
    
    # Setup trainer
    trainer = setup_trainer()
    print("\nTrainer setup complete. Ready for training.") 