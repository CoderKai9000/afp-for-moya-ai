"""
AzureOpenAIAgent for Moya.

An Agent that uses Azure OpenAI's ChatCompletion or Completion API
to generate responses.
"""

import os
from openai import AzureOpenAI
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from moya.agents.openai_agent import OpenAIAgent, OpenAIAgentConfig


@dataclass
class AzureOpenAIAgentConfig(OpenAIAgentConfig):
    """
    Configuration data for an AzureOpenAIAgent.
    """
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None
    deployment_name: Optional[str] = None


class AzureOpenAIAgent(OpenAIAgent):
    """
    A simple AzureOpenAI-based agent that uses the ChatCompletion API.
    """

    def __init__(self, config: AzureOpenAIAgentConfig):
        """
        Initialize the AzureOpenAIAgent.

        :param config: Configuration for the agent.
        """
        # Call the parent class constructor with the necessary parameters
        super().__init__(
            agent_name=config.agent_name,
            description=config.description,
            agent_config=config,
            tool_registry=config.tool_registry
        )
        
        # Validate Azure-specific configurations
        if not config.api_base:
            raise ValueError("Azure OpenAI API base is required for AzureOpenAIAgent.")
        
        if not config.api_version:
            raise ValueError("Azure OpenAI API version is required for AzureOpenAIAgent.")
            
        if not config.deployment_name:
            raise ValueError("Deployment name is required for AzureOpenAIAgent.")

        # Ensure the API base doesn't have a trailing slash
        api_base = config.api_base.rstrip('/')

        # Print configuration info for debugging
        print(f"Initializing AzureOpenAIAgent with:")
        print(f"  - Deployment name: {config.deployment_name}")
        print(f"  - API base: {api_base}")
        print(f"  - API version: {config.api_version}")
        print(f"  - Agent name: {config.agent_name}")

        # Create Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=config.api_key, 
            azure_endpoint=api_base, 
            api_version=config.api_version,
            organization=config.organization
        )
        
        # Store the deployment name for use in API calls
        self.deployment_name = config.deployment_name

    def setup(self) -> None:
        """
        Override the setup method since we already initialized the client in __init__.
        """
        # No need to create a new client as we did that in __init__
        pass
        
    def handle_message(self, message: str, **kwargs) -> str:
        """
        Calls Azure OpenAI ChatCompletion to handle the user's message.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message},
                ],
                **self.agent_config.llm_config
            )
            
            # Check if choices list is empty before accessing it
            if not response.choices:
                return "[AzureOpenAIAgent error: No response choices returned from API]"
                
            return response.choices[0].message.content or "[Empty response]"
        except Exception as e:
            print(f"AzureOpenAI API error: {str(e)}")
            return f"[AzureOpenAIAgent error: {str(e)}]"

    def handle_message_stream(self, message: str, **kwargs):
        """
        Calls Azure OpenAI ChatCompletion to handle the user's message with streaming support.
        """
        # Starting streaming response from AzureOpenAIAgent
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message},
                ],
                stream=True,
                **self.agent_config.llm_config
            )
            
            response_started = False
            
            for chunk in response:
                if not chunk.choices:
                    continue  # Skip this chunk if choices is empty
                    
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    if chunk.choices[0].delta.content is not None:
                        response_started = True
                        content = chunk.choices[0].delta.content
                        yield content
            
            # If we never yielded anything, yield an empty response message
            if not response_started:
                yield "[No response content]"
                
        except Exception as e:
            error_message = f"[AzureOpenAIAgent error: {str(e)}]"
            print(error_message)
            yield error_message
