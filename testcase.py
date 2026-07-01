import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_classic.prompts import PromptTemplate
# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# Initialize Chat Model (GPT-4)
llm = ChatOpenAI(api_key=api_key, model="gpt-4")
#llm = OpenAI(api_key=api_key, model="gpt-4")
# Prompt Template for Generating Test Cases

# few shot prompt (fewshot or 0 shot)
test_case_prompt = PromptTemplate(
    input_variables=["requirement"],
    template="""
    Given the following software requirement:
    "{requirement}"
    Generate a detailed set of test cases covering:
    - Functional test cases
    - Edge cases
    - Negative test cases
    Provide structured output in this format:
    ```
    Test Case ID: TC-001
    Description: [Test Scenario]
    Steps:
    1. [Step 1]
    2. [Step 2]
    Expected Result: [Expected Outcome]
    ```
    """
)
# Prompt Template for Generating Test Artifacts (Test Plan, User Scenarios)
# few short format (fewshot or 0 shot)
artifact_prompt = PromptTemplate(
    input_variables=["requirement"],
    template="""
    Based on the following requirement:
    "{requirement}"
    Generate the following test artifacts:
    1. **Test Plan**: Outline objectives, scope, and responsibilities.
    2. **User Scenarios**: Define realistic user interactions with the system.
    3. **Test Data**: Provide sample test data.
    Structure the output in Markdown format for readability.
    """
)
#  Function to generate test cases
def generate_test_cases(requirement):
    prompt = test_case_prompt.format(requirement=requirement)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content
#  Function to generate test artifacts
def generate_test_artifacts(requirement):
    prompt = artifact_prompt.format(requirement=requirement)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content
# Function to save generated documentation
def save_to_file(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Documentation saved as {filename}")
#  Main Execution
if __name__ == "__main__":
    # Example Requirement
    requirement_text = """
    Implement a login functionality where users enter their email and password. 
    If the credentials are correct, they are redirected to the dashboard. 
    If incorrect, an error message is displayed.
    """
    # Generate Documentation
    test_cases = generate_test_cases(requirement_text)
    test_artifacts = generate_test_artifacts(requirement_text)
    # Save Outputs
    save_to_file("test_cases.txt", test_cases)
    save_to_file("test_artifacts.md", test_artifacts)
    # Print Output
    print("\n Generated Test Cases:\n", test_cases)
    print("\n Generated Test Artifacts:\n", test_artifacts)
 