import yaml

from jinja2 import Template
from langsmith import Client as LSClient


def from_template_config(yaml_path: str, prompt: str):

    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    template_content = config["prompts"][prompt]

    return Template(template_content)

def  build_prompt_jinja(prompts_file, prompt, preprocessed_context, question):
    
    template = from_template_config(prompts_file, prompt)

    return template.render(preprocessed_context=preprocessed_context, question=question)

def  build_prompt_registry(prompt, ls_client, preprocessed_context, question):
    
    template_text = ls_client.pull_prompt(prompt).messages[0].prompt.template
    
    template = Template(template_text)

    return template.render(preprocessed_context=preprocessed_context, question=question)