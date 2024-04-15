# llm-bedrock-mistral

[![PyPI](https://img.shields.io/pypi/v/llm-bedrock-mistral.svg)](https://pypi.org/project/llm-bedrock-mistral/)
[![Changelog](https://img.shields.io/github/v/release/tlpinney/llm-bedrock-mistral?include_prereleases&label=changelog)](https://github.com/tlpinney/llm-bedrock-mistral/releases)
[![Tests](https://github.com/tlpinney/llm-bedrock-mistral/workflows/Test/badge.svg)](https://github.com/tlpinney/llm-bedrock-mistral/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/tlpinney/llm-bedrock-mistral/blob/main/LICENSE)

Plugin for [LLM](https://llm.datasette.io/) adding support for Mistral models in Amazon Bedrock

## Installation

Install this plugin in the same environment as LLM. From the current directory
```bash
llm install llm-bedrock-mistral
```
## Configuration

You will need to specify AWS Configuration with the normal boto3 and environment variables.

For example, to use the region `us-east-1` and AWS credentials under the `personal` profile, set the environment variables

```bash
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=personal
```

## Usage

This plugin adds model called `bedrock-mistral-7b-instruct` and `bedrock-mixtral-8x7b-instruct`. You can also use it with the alias `bm7` or `bm8`.

You can query them like this:

```bash
llm -m bedrock-mistral-7b-instruct "Ten great names for a new space station"
```

```bash
llm -m bm7 "Ten great names for a new space station"
```

You can also chat with the model:

```bash
llm chat -m bm8
```

## Options

- `-o max_tokens`, Specify the maximum number of tokens to use in the generated response.
- `-o temperature`, Controls the randomness of predictions made by the model.
- `-o top_p`, Controls the diversity of text that the model generates by setting the percentage.
- `-o top_k`, Controls the number of most-likely candidates that the model considers for the next token.

Use like this:
```bash
llm -m bm7 -o max_tokens 20 -o temperature 0 "Return the alphabet. Be succint."
 Here is the alphabet in English: A, B, C, D, E, F,
```

