from pydantic import BaseModel, Field
from typing import Literal, get_origin

class AgentOutput(BaseModel):
    title: str = Field(description="A short 5-8 word title summarising the output")
    summary: str = Field(description="A single paragraph overview of the findings")
    domain: str = Field(description="The domain this output covers e.g. technical, market, security, product, competitive")
    key_points: list[str] = Field(description="The main findings or insights, each as a concise sentence")
    recommendations: list[str] = Field(description="Strategic recommendations based on the findings")
    next_steps: list[str] = Field(description="Immediate actionable steps, specific and time-bound")
    observations: list[str] = Field(description="Things noticed that may be worth tracking or investigating further")
    follow_up_questions: list[str] = Field(description="Questions the user should explore next, phrased as questions")
    tags: list[str] = Field(description="3-6 short keywords extracted from the output for filtering and search")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level — must be exactly: high, medium, or low")


    @classmethod
    def prompt_template(cls) -> str:
        fields = []
        for name, field in cls.model_fields.items():
            description = field.description or name
            if get_origin(field.annotation) is list or field.annotation == list[str]:
                fields.append(f'    "{name}": ["{description}"]')
            else:
                fields.append(f'    "{name}": "{description}"')

        inner = ",\n".join(fields)
        return (
            "You must respond with valid JSON only. "
            "No preamble, no explanation, no markdown code blocks.\n"
            "Respond with exactly this structure:\n"
            "{\n" + inner + "\n}"
        )
