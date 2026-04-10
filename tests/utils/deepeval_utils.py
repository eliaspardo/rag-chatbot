from deepeval.models import DeepEvalBaseLLM
from tests.utils.evals_utils import build_provider_llm
import logging
import re

logger = logging.getLogger(__name__)


class DeepEvalLLMAdapter(DeepEvalBaseLLM):
    def __init__(self):
        self.model = build_provider_llm()

    def load_model(self):
        return self.model

    def _sanitize_json_output(self, content: str) -> str:
        """
        Sanitize LLM output to ensure valid JSON by escaping control characters.

        Many LLMs output JSON with unescaped newlines, tabs, and other control
        characters within string values, which breaks JSON parsing. This method
        fixes those issues while preserving the JSON structure.

        Additionally, some LLMs (like Llama-3.3-70B) output valid JSON followed by
        hallucinated text/code. This method extracts ONLY the first complete JSON object.
        """
        # Extract JSON from markdown code blocks if present
        if "```json" in content or "```" in content:
            json_match = re.search(r"```(?:json)?\s*\n(.*?)```", content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

        # Find the first JSON object by tracking brace depth
        start = content.find("{")
        if start == -1:
            # No JSON found, return as-is
            return content

        # Track brace depth to find the matching closing brace
        depth = 0
        in_string = False
        escaped = False
        end = start

        for i in range(start, len(content)):
            char = content[i]

            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        # Found the matching closing brace
                        end = i + 1
                        break

        if end == start:
            # No matching closing brace found
            return content

        # Extract just the first complete JSON object
        json_part = content[start:end]

        # Now sanitize control characters within string values
        result = []
        in_string = False
        escaped = False

        for char in json_part:
            if escaped:
                result.append(char)
                escaped = False
                continue

            if char == "\\":
                result.append(char)
                escaped = True
                continue

            if char == '"':
                result.append(char)
                in_string = not in_string
                continue

            # If we're inside a string, escape control characters
            if in_string:
                if char == "\n":
                    result.append("\\n")
                elif char == "\r":
                    result.append("\\r")
                elif char == "\t":
                    result.append("\\t")
                elif ord(char) < 32:  # Other control characters
                    result.append(f"\\u{ord(char):04x}")
                else:
                    result.append(char)
            else:
                result.append(char)

        return "".join(result)

    def generate(self, prompt: str) -> str:
        result = self.model.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)
        sanitized = self._sanitize_json_output(content)
        self.print_llm_exchange(prompt, sanitized)
        return sanitized

    async def a_generate(self, prompt: str) -> str:
        result = await self.model.ainvoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)
        sanitized = self._sanitize_json_output(content)
        self.print_llm_exchange(prompt, sanitized)
        return sanitized

    def get_model_name(self) -> str:
        return getattr(self.model, "model_name", None) or getattr(
            self.model, "model", "unknown"
        )

    def print_llm_exchange(self, prompt: str, result: str) -> None:
        logger.debug("\n--- Deepeval EVAL LLM prompt ---")
        logger.debug(prompt)
        logger.debug("\n--- Deepeval EVAL LLM output ---")
        logger.debug(result)
        logger.debug("\n-------------------------------\n")
