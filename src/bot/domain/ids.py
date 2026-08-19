"""Semantic identifiers for distinct domain concepts.

Every semantically different identifier receives its own ``NewType`` so that
IDs cannot be mixed up accidentally across application/domain contracts.
"""

from typing import NewType

# Telegram identifiers
TelegramUserId = NewType("TelegramUserId", int)
TelegramChatId = NewType("TelegramChatId", int)
TelegramMessageId = NewType("TelegramMessageId", int)

# Inference identifiers
RequestId = NewType("RequestId", str)
ModelId = NewType("ModelId", str)

# Future extension points (not used in the first version)
AgentId = NewType("AgentId", str)
ToolId = NewType("ToolId", str)
