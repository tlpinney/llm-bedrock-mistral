import llm
from llm.plugins import pm
#from llm_bedrock_mistral import BedrockMistral
#from llm_bedrock_mistral import BedrockMistral7Options


# based off of https://github.com/flabat/llm-bedrock-meta
def test_plugin_is_installed():
    plugins = pm.get_plugins()
    assert "llm_bedrock_mistral" in {mod.__name__ for mod in plugins}
