import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Step 1: Generate GitHub Actions YAML
prompt_generate = """
Generate a GitHub Actions pipeline in YAML for a Python project.
The workflow should:
- Run on push to the main branch
- Lint the code using flake8
- Run tests using pytest
- Include a mocked deployment step using `echo`
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt_generate}],
    temperature=0.3,
)

generated_yaml = response.choices[0].message.content
print(" Generated GitHub Actions Workflow:\n")
print(generated_yaml)

# Step 2: Introduce a YAML error
broken_yaml = generated_yaml.replace("- name: Lint with flake8", "   - name: Lint with flake8")
print("\n Broken YAML with Indentation Error:\n")
print(broken_yaml)

# Step 3: Ask GPT to fix the YAML
prompt_debug = f"""
I have a GitHub Actions YAML file and I'm getting an error saying:
'YAML file does not conform to schema: Unexpected value'.

Here is the broken YAML:
---
{broken_yaml}
---
Can you help identify and fix the issue?
"""

response_fix = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt_debug}],
    temperature=0.3,
)

import re

# Extract only YAML content between ```yaml and ```
raw_output = response_fix.choices[0].message.content

# Remove triple backticks and language identifiers if present
fixed_yaml = re.sub(r"```(?:yaml)?\n?", "", raw_output)  # Remove ```yaml or ```
fixed_yaml = re.sub(r"\n?```$", "", fixed_yaml.strip())  # Remove trailing ```
fixed_yaml = re.sub(r"^[^\r\n]*[\r\n]+", "", fixed_yaml) #Remove any leading non-YAML text before the first line of YAML
fixed_yaml = re.sub(r'\n.[^\n]*$', '', fixed_yaml) #Remove any trailing non-YAML text before the first line of YAML


print(" Cleaned Fixed YAML by GPT:\n")
print(fixed_yaml)

#fixed_yaml = response_fix.choices[0].message.content
print(" Fixed YAML by GPT:\n")
print(fixed_yaml)

# Step 4: Save the YAML file
os.makedirs(".github/workflows", exist_ok=True)
with open(".github/workflows/python-ci.yml", "w") as f:
    f.write(fixed_yaml)

print(" Saved to .github/workflows/python-ci.yml")
