import pytest
from unittest.mock import Mock
import jinja2
from returns.result import Success, Failure
from tgutil.prompting import ContextPromptInfo
from tgutil.prompting_utils import MinichainPrompterWrapper
from tgutil.configs import APIConfig, TextGenerationConfig

def test_get_dict_with_generated_text_success():
    # Setup
    mock_generate_text_fn = Mock()
    template = jinja2.Template("Test template {{ repo }}")
    wrapper = MinichainPrompterWrapper(
        generate_text_fn=mock_generate_text_fn,
        prompt_template=template
    )
    
    # Mock the generation results to return Success
    wrapper.get_generation_results = Mock(return_value=Success("Generated text"))
    
    # Create test context info
    context_info = ContextPromptInfo(
        repo="test-repo",
        dependencies=["dep1", "dep2"],
        tasks=["task1"]
    )
    
    # Execute
    result = wrapper.get_dict_with_generated_text(context_info)
    
    # Assert
    assert result.is_success()
    value = result.unwrap()
    assert value["generated_text"] == "Generated text"
    assert value["input_text"] == "Test template test-repo"
    assert value["repo"] == "test-repo"

def test_get_dict_with_generated_text_failure():
    # Setup
    mock_generate_text_fn = Mock()
    template = jinja2.Template("Test template {{ repo }}")
    wrapper = MinichainPrompterWrapper(
        generate_text_fn=mock_generate_text_fn,
        prompt_template=template
    )
    
    # Mock the generation results to return Failure
    test_exception = Exception("Generation failed")
    wrapper.get_generation_results = Mock(return_value=Failure(test_exception))
    
    # Create test context info
    context_info = ContextPromptInfo(
        repo="test-repo",
        dependencies=["dep1", "dep2"],
        tasks=["task1"]
    )
    
    # Execute
    result = wrapper.get_dict_with_generated_text(context_info)
    
    # Assert
    assert result.is_failure()
    assert isinstance(result.failure(), Exception)
    assert str(result.failure()) == "Generation failed"

def test_minichain_prompter_wrapper_with_openai():
    # Setup
    api_config = APIConfig(
        endpoint_url="https://api.openai.com/v1/completions",
        flavor="openai"
    )
    
    template = "Test template with repo: {{ repo }}"
    wrapper = MinichainPrompterWrapper.create(
        text_generation_config=api_config,
        prompt_template=template,
        prompt_templates_path=None,
        prompt_template_name=None,
        max_new_tokens=50
    )
    
    # Create test context info
    context_info = ContextPromptInfo(
        repo="test-openai-repo",
        dependencies=["dep1", "dep2"],
        tasks=["task1"]
    )
    
    # Mock the generation results
    wrapper.generate_text_fn = Mock()
    wrapper.generate_text_fn.return_value.run = Mock(return_value="OpenAI generated response")
    
    # Execute
    result = wrapper.get_dict_with_generated_text(context_info)
    
    # Assert
    assert result.is_success()
    value = result.unwrap()
    assert value["generated_text"] == "OpenAI generated response"
    assert value["input_text"] == "Test template with repo: test-openai-repo"
    assert value["repo"] == "test-openai-repo"
