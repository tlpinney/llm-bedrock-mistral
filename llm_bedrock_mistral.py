import boto3
import llm
import json
from pydantic import Field
from typing import Optional

@llm.hookimpl
def register_models(register):
    register(
        BedrockMistral("mistral.mistral-7b-instruct-v0:2", BedrockMistral7Options),
        aliases=("bedrock-mistral-7b-instruct", "bm7"),
    )
    register(
        BedrockMistral("mistral.mixtral-8x7b-instruct-v0:1", BedrockMixtral8Options),
        aliases=("bedrock-mixtral-8x7b-instruct", "bm8"),
    )
# XXX Fix issue with invocation
#    register(
#        BedrockMistral("mistral.mistral-large-2402-v1:0", BedrockMistralLargeOptions),
#        aliases=("mistral.mistral-large-2402-v1:0", "bml")
#    )


# Parameters, defaults and descriptions taken from
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
# XXX DRY
# XXX Implement stop
class BedrockMistral7Options(llm.Options):
    max_tokens: Optional[int] = Field(
        description=(
            "Specify the maximum number of tokens to use in the generated response. "
            "The model truncates the response once the generated text exceeds max_tokens."
        ),
        ge=1,
        le=8192,
        default=512,
    )
    temperature: Optional[float] = Field(
        description="Controls the randomness of predictions made by the model.",
        ge=0,
        le=1,
        default=0.5,
    )
    top_p: Optional[float] = Field(
        description=(
            "Controls the diversity of text that the model generates by setting the percentage "
            "of most-likely candidates that the model considers for the next token."
        ),
        ge=0,
        le=1,
        default=0.9,
    )
    top_k: Optional[int] = Field(
        description=(
            "Controls the number of most-likely candidates that the model considers for the next token."
        ),
        ge=1,
        le=200,
        default=50,
    )


class BedrockMixtral8Options(llm.Options):
    max_tokens: Optional[int] = Field(
        description=(
            "Specify the maximum number of tokens to use in the generated response. "
            "The model truncates the response once the generated text exceeds max_tokens."
        ),
        ge=1,
        le=4096,
        default=512,
    )
    temperature: Optional[float] = Field(
        description="Controls the randomness of predictions made by the model.",
        ge=0,
        le=1,
        default=0.5,
    )
    top_p: Optional[float] = Field(
        description=(
            "Controls the diversity of text that the model generates by setting the percentage "
            "of most-likely candidates that the model considers for the next token."
        ),
        ge=0,
        le=1,
        default=0.9,
    )
    top_k: Optional[int] = Field(
        description=(
            "Controls the number of most-likely candidates that the model considers for the next token."
        ),
        ge=1,
        le=200,
        default=50,
    )


class BedrockMistralLargeOptions(llm.Options):
    max_tokens: Optional[int] = Field(
        description=(
            "Specify the maximum number of tokens to use in the generated response. "
            "The model truncates the response once the generated text exceeds max_tokens."
        ),
        ge=1,
        le=8192,
        default=8192,
    )
    temperature: Optional[float] = Field(
        description="Controls the randomness of predictions made by the model.",
        ge=0,
        le=1,
        default=0.7,
    )
    top_p: Optional[float] = Field(
        description=(
            "Controls the diversity of text that the model generates by setting the percentage "
            "of most-likely candidates that the model considers for the next token."
        ),
        ge=0,
        le=1,
        default=1,
    )


class BedrockMistral(llm.Model):
    can_stream: bool = True

    def __init__(self, model_id, options):
        self.model_id = model_id
        self.default_system_prompt = None
        self.Options = options

    # based off of https://github.com/flabat/llm-bedrock-meta
    # modified to work with Mistral's prompt
    # XXX needs more testing with conversation and system prompts
    def build_messages(self, prompt, conversation):
        prompt_bits = []

        # Now build the prompt pieces
        if conversation is not None:
            for prev_response in conversation.responses:
                prompt_bits.append("<s>[INST] ")
                prompt_bits.append(
                    f"{prev_response.prompt.prompt} [/INST] ",
                )

        # Add the latest prompt
        if not prompt_bits:
            # Start with the system prompt
            prompt_bits.append("<s>[INST] ")
        else:
            prompt_bits.append("<s>[INST] ")
        prompt_bits.append(f"{prompt.prompt} [/INST] ")

        return "".join(prompt_bits).rstrip()

    # based off of https://github.com/flabat/llm-bedrock-meta
    def execute(self, prompt, stream, response, conversation):
        client = boto3.client("bedrock-runtime")

        prompt_str = self.build_messages(prompt, conversation)

        prompt_json = {
            "prompt": prompt_str,
            "max_tokens": prompt.options.max_tokens,
            "temperature": prompt.options.temperature,
            "top_p": prompt.options.top_p,
        }

        # top_k is disabled in mistral large
        if "top_k" in prompt.options:
            prompt_json["top_k"] = prompt.options.top_k

        prompt.prompt_json = prompt_json

        if stream:
            bedrock_response = client.invoke_model_with_response_stream(
                modelId=self.model_id, body=json.dumps(prompt_json)
            )

            chunks = bedrock_response.get("body")

            for event in chunks:
                chunk = event.get("chunk")
                if chunk:
                    response = json.loads(chunk.get("bytes").decode())
                    completion = response["outputs"][0]["text"]
                    yield completion

        else:
            # based on https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeMixtral8x7B_section.html
            # and  https://github.com/flabat/llm-bedrock-meta
            try:
                bedrock_response = client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(prompt_json),
                )
            except boto3.exceptions.ValidationException as e:
                print(e.message)

            body = bedrock_response["body"].read()

            response.response_json = json.loads(body)

            outputs = response.response_json.get("outputs")

            completions = [output["text"] for output in outputs]

            for completion in completions:
                yield completion
