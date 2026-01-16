import unittest
import torch
from unittest.mock import MagicMock, patch
from src.data.dataset import LazySupervisedDataset
from src.data.entities import InstructionSample, ConversationTurn

class MockTokenizer:
    def __init__(self):
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        
    def __call__(self, text, return_tensors="pt", add_special_tokens=True, **kwargs):
        # Mock tokenization: 1 char = 1 token (simple ASCII)
        # + BOS if add_special_tokens
        ids = [ord(c) for c in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return MagicMock(input_ids=torch.tensor([ids]))

class TestMaskingLogic(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.dataset = LazySupervisedDataset(
            data_path="dummy.json",
            tokenizer=self.tokenizer,
            image_folder="dummy_folder",
            data_args=None
        )
        # Mock loading data to avoid file I/O
        self.dataset.list_data_dict = []

    def test_masking_user_turn(self):
        # Create a sample
        sample = InstructionSample(
            id="1",
            image="test.jpg",
            conversations=[
                ConversationTurn(from_role="human", value="Hello"),
                ConversationTurn(from_role="gpt", value="World")
            ]
        )
        self.dataset.list_data_dict = [sample]
        
        # We patch Image.open to avoid error
        with patch("PIL.Image.open") as mock_open:
            mock_open.return_value.convert.return_value = MagicMock()
            
            # We assume image processor is None for this test or mocked
            self.dataset.image_processor = None
            
            # Execute __getitem__
            item = self.dataset[0]
            
            labels = item["labels"]
            input_ids = item["input_ids"]
            
            # Verify structure
            # Text: "User: <image>\nHello\nAssistant: World\n"
            # (Note: Logic in dataset adds <image> if missing)
            
            # Check User part is -100
            # "User: <image>\nHello\n" should be masked.
            # "Assistant: World\n" should be unmasked.
            
            # With MockTokenizer:
            # User part length
            # Assistant part length
            
            # Let's check if there are ANY unmasked tokens
            unmasked = labels[labels != -100]
            self.assertTrue(len(unmasked) > 0, "Should have unmasked tokens for Assistant")
            
            # Check expansion
            # Text "User: <image>\n..."
            # Expect roughly 576 + small text length tokens
            self.assertGreater(len(input_ids), 500, "Should have expanded image tokens")
            
            # Check if User tokens are masked (first few tokens)
            # The BOS is 0, then USER: ...
            self.assertEqual(labels[0].item(), -100)
            
            # Check unmasked Assistant
            unmasked = labels[labels != -100]
            self.assertTrue(len(unmasked) > 0, "Should have unmasked tokens for Assistant")
            
            # Check consistency
            self.assertEqual(len(input_ids), len(labels))

if __name__ == "__main__":
    unittest.main()
