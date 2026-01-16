from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ConversationTurn:
    from_role: str  # "human" or "gpt"
    value: str

@dataclass
class InstructionSample:
    id: str
    image: Optional[str]  # Image filename
    conversations: List[ConversationTurn] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'InstructionSample':
        """
        Create an InstructionSample from a dictionary (e.g., from JSON).
        """
        # Handle different ID formats if necessary, convert to string
        sample_id = str(data.get("id", "unknown"))
        
        image_path = data.get("image")
        
        conversations_data = data.get("conversations", [])
        conversations = []
        
        for turn in conversations_data:
            # LLaVA dataset uses 'from' and 'value'
            role = turn.get("from")
            value = turn.get("value")
            
            if role and value:
                conversations.append(ConversationTurn(from_role=role, value=value))
                
        return cls(id=sample_id, image=image_path, conversations=conversations)
