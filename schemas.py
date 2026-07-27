# schemas.py
from enum import Enum
from pydantic import BaseModel, Field


class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    other = "other"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class SupportTicket(BaseModel):
    # Field descriptions ship INTO the JSON schema and steer the model — treat them as a contract.
    summary: str = Field(description="One-sentence summary of the customer's problem")
    category: Category = Field(description="Which area the request belongs to")
    priority: Priority = Field(description="Urgency inferred from tone and content")
    sentiment: Sentiment = Field(description="Overall emotional tone of the message")
    customer_name: str | None = Field(default=None, description="Customer name if stated, else null")
    action_items: list[str] = Field(default_factory=list, description="Concrete next steps to resolve it")

# schemas.py (add)
class SupportReply(BaseModel):
    reply: str = Field(description="Message shown to the customer. Plain text, under 80 words.")
    orders_checked: list[str] = Field(default_factory=list,
                                      description="Order IDs actually looked up via tools, not guessed")
    resolved: bool = Field(description="True only if the customer's question was fully answered")
    escalate: bool = Field(description="True if a human agent must take over")
    escalation_reason: str | None = Field(default=None, description="Why escalation is needed, or null")
    sentiment: Sentiment = Field(description="Customer's emotional tone across this conversation")