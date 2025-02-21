"""Unit tests for the medical diagnosis trainer."""

import unittest
import tempfile
import shutil
import os
import subprocess
import logging
from healthcare_examples import create_medical_dataset, setup_trainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestNotebook(unittest.TestCase):
    """Test cases for verifying the notebook cells can run."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.test_dir}")
    
    def tearDown(self):
        """Clean up test environment."""
        logger.info(f"Cleaning up temporary directory: {self.test_dir}")
        shutil.rmtree(self.test_dir)
    
    def test_notebook_cell1_dependencies(self):
        """Test that the first cell's dependencies can be imported."""
        logger.info("Testing notebook cell 1 - Checking dependencies...")
        try:
            logger.info("Importing torch...")
            import torch
            logger.info("Importing transformers...")
            import transformers
            logger.info("Importing accelerate...")
            import accelerate
            logger.info("Importing trl...")
            import trl
            logger.info("All dependencies imported successfully!")
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import dependencies: {str(e)}")
    
    def test_notebook_cell2_execution(self):
        """Test that the second cell's code can run."""
        try:
            logger.info("Testing notebook cell 2 - Testing model setup...")
            
            # Configure for minimal test
            output_dir = os.path.join(self.test_dir, "test_output")
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
            
            # Create minimal dataset
            logger.info("Creating medical dataset...")
            dataset = create_medical_dataset()
            logger.info(f"Created dataset with {len(dataset)} examples")
            self.assertTrue(len(dataset) > 0, "Dataset should not be empty")
            
            # Setup trainer with minimal config
            logger.info("Setting up trainer with minimal configuration...")
            trainer = setup_trainer(output_dir=output_dir)
            self.assertIsNotNone(trainer, "Trainer should be initialized")
            
            # Test that trainer components are properly initialized
            logger.info("Verifying trainer components...")
            self.assertIsNotNone(trainer.model, "Model should be initialized")
            self.assertIsNotNone(trainer.args, "Training arguments should be initialized")
            self.assertIsNotNone(trainer.train_dataset, "Training dataset should be initialized")
            logger.info("All trainer components verified successfully!")
            
        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}")
            self.fail(f"Failed to execute notebook cell 2: {str(e)}")

if __name__ == '__main__':
    unittest.main() 