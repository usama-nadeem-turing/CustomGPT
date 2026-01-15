# CustomGPT - Resume Evaluation Tool

An AI-powered resume evaluation system that uses OpenAI's GPT models to analyze and evaluate developer resumes based on customizable evaluation criteria. The tool processes resumes in parallel, extracts structured evaluation data, and outputs results in both JSON and CSV formats.

## Features

- 🤖 **AI-Powered Evaluation**: Uses OpenAI GPT-4o to evaluate resumes based on custom prompts
- 🎨 **Fully Customizable**: Easily tailor both the evaluation prompt and script to match any evaluation criteria
- ⚡ **Parallel Processing**: Processes multiple resumes concurrently for faster evaluation
- 📊 **Structured Output**: Generates both JSON and CSV outputs for easy analysis
- 💾 **Incremental Saving**: Saves results incrementally to prevent data loss
- 🎯 **Flexible Filtering**: Filter by developer ID, limit processing count, or specific CSV files
- 📈 **Progress Tracking**: Real-time progress bar and cost/time estimates

## Requirements

- Python 3.7+
- OpenAI API key
- Required packages (see `requirements.txt`)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd CustomGPT
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key:**
   
   Create a `.env` file in the project directory:
   ```env
   OPENAI_API_KEY=sk-your-api-key-here
   ```
   
   Or set it as an environment variable:
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="sk-your-api-key-here"
   
   # Windows Command Prompt
   set OPENAI_API_KEY=sk-your-api-key-here
   
   # Linux/Mac
   export OPENAI_API_KEY="sk-your-api-key-here"
   ```
   
   For detailed setup instructions, see [SETUP.md](SETUP.md).

## Input Data Format

The tool expects a CSV file with at least the following columns:
- `developer_id`: Unique identifier for each developer
- `resume_plain`: Plain text resume content

Example CSV structure:
```csv
developer_id,resume_plain
1,"John Doe - Software Engineer with 5 years of experience..."
2,"Jane Smith - Data Scientist specializing in machine learning..."
```

## Usage

### Basic Usage

Evaluate all developers in the default CSV file (`json_list.csv`):
```bash
python main.py
```

### Command-Line Options

```bash
python main.py [OPTIONS]
```

**Options:**
- `--csv PATH`: Specify path to input CSV file (default: `json_list.csv`)
- `--limit N`: Limit the number of developers to process (useful for testing)
- `--developer_id ID`: Process only a specific developer by ID

### Examples

**Process a specific CSV file:**
```bash
python main.py --csv dev_data.csv
```

**Process only the first 10 developers (for testing):**
```bash
python main.py --limit 10
```

**Process a specific developer:**
```bash
python main.py --developer_id 12345
```

**Combine options:**
```bash
python main.py --csv dev_data.csv --limit 50
```

## Customization

### Custom Tailoring the Prompt and Script

This tool is designed to be fully customizable. You can tailor both the evaluation prompt (`prompt.txt`) and the script (`main.py`) to work with any evaluation criteria you need. Here's a comprehensive guide:

### Step 1: Customize the Evaluation Prompt

The evaluation criteria are defined in `prompt.txt`. You can completely rewrite this file to evaluate any criteria you need.

**Key Components of `prompt.txt`:**

1. **Placeholder for Resume Data**: The prompt must include `${dev_resume}` where the resume content will be inserted
   ```
   ${dev_resume}
   ```

2. **Field Definitions**: Define each evaluation field with:
   - A clear question or instruction
   - Field name (e.g., `Field_Name: my_field`)
   - Expected value format (e.g., `Field_Values: [Yes/No]`)

3. **Output Format**: Include a JSON example showing the expected output structure

**Example Custom Prompt Structure:**
```
You are an expert resume evaluator.

## Candidate Resume
${dev_resume}

## Evaluation Questions

Check if candidate has experience in [YOUR CRITERIA]
Field_Name: field1; Field_Values: [Length of experience in years]

Check if candidate has [YOUR CRITERIA]
Field_Name: field2; Field_Values: [Yes/No]

## Output Format
```json
{
  "field1": "[Length of experience in years]",
  "field2": "[Yes/No]"
}
```
```

### Step 2: Update the Script to Match Your Prompt

After modifying `prompt.txt`, you **must** update `main.py` to match your new fields:

1. **Locate the `create_evaluation_dataframe()` function** in `main.py` (around line 270)

2. **Update the `expected_fields` list** to match all field names from your prompt's JSON output:

```python
def create_evaluation_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Create a DataFrame from evaluation results with developer_id and parsed scores
    matching the current evaluation prompt (field1, field2, field3).
    
    Args:
        results (List[Dict]): List of evaluation results from evaluate_all_developers
        
    Returns:
        pd.DataFrame: DataFrame with developer_id and evaluation scores
    """
    # Define all expected fields from the prompt
    expected_fields = [
        'field1', 'field2', 'field3'  # Update this list to match your prompt fields
    ]
```

3. **Update the docstring** to reflect your new fields

### Step-by-Step Customization Example

**Scenario**: You want to evaluate candidates for a Data Science role with these criteria:
- Years of Python experience
- Has machine learning experience (Yes/No)
- Number of ML projects completed
- Preferred ML frameworks (comma-separated list)

**Step 1: Update `prompt.txt`:**
```markdown
You are an expert resume evaluator.

## Candidate Resume
${dev_resume}

## Evaluation Questions

Check years of Python programming experience.
Field_Name: python_years; Field_Values: [Length of experience in years]

Check if candidate has machine learning experience.
Field_Name: has_ml; Field_Values: [Yes/No]

Count the number of machine learning projects mentioned.
Field_Name: ml_projects; Field_Values: [Number of projects]

List ML frameworks/tools mentioned (comma-separated).
Field_Name: ml_frameworks; Field_Values: [comma separated list]

## Output Format
```json
{
  "python_years": "[Length of experience in years]",
  "has_ml": "[Yes/No]",
  "ml_projects": "[Number of projects]",
  "ml_frameworks": "[comma separated list]"
}
```
```

**Step 2: Update `main.py`:**
```python
def create_evaluation_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Create a DataFrame from evaluation results with developer_id and parsed scores
    matching the current evaluation prompt (python_years, has_ml, ml_projects, ml_frameworks).
    
    Args:
        results (List[Dict]): List of evaluation results from evaluate_all_developers
        
    Returns:
        pd.DataFrame: DataFrame with developer_id and evaluation scores
    """
    # Define all expected fields from the prompt
    expected_fields = [
        'python_years', 'has_ml', 'ml_projects', 'ml_frameworks'
    ]
    # ... rest of function remains the same
```

**Step 3: Test Your Changes:**
```bash
# Test with a single developer first
python main.py --limit 1

# Check the output CSV to verify all fields are present
# Review evaluation_results.json to see raw API responses
```

### Important Notes for Customization

1. **Field Names Must Match**: The field names in your prompt's JSON output must exactly match the `expected_fields` list in `main.py`

2. **JSON Format**: Ensure your prompt instructs the AI to output valid JSON. The script uses regex to extract JSON from markdown code blocks if present

3. **Error Handling**: If a field is missing from the API response, it will be set to "ERROR" in the CSV. Always test with a small dataset first

4. **Placeholder**: Never remove `${dev_resume}` from the prompt - this is where the actual resume content gets inserted

5. **Testing**: Always test with `--limit 1` or `--limit 5` before processing large datasets

### Current Evaluation Fields (Senior Technical Roles)

The current prompt evaluates:
- `a_XP`: Direct experience with batteries, energy storage, semiconductors, chips, or hardware systems (years)
- `b_XP`: Adjacent experience in hardware, embedded systems, manufacturing, etc. (years)
- `c_XP`: Research, systems-level, or first-principles engineering backgrounds (years)
- `t1`: Relevant keywords or experiences (comma-separated list)
- `t2`: Fit classification (Direct/Adjacent/Potential/None)
- `t3`: Education level (PhD/Post Grad/None)
- `t4`: Experience type (Industry/Academic Labs/None)

## Output Files

The tool generates two output files:

1. **`evaluation_results.json`**: Complete evaluation results with raw API responses
   ```json
   [
     {
       "developer_id": 1,
       "evaluation": "{...JSON response from OpenAI...}"
     }
   ]
   ```

2. **`evaluation_scores.csv`**: Parsed evaluation scores in tabular format
   ```csv
   developer_id,a_XP,b_XP,c_XP,t1,t2,t3,t4
   1,5,3,2,"batteries, semiconductors","Direct","PhD","Industry"
   ```

## Configuration

### Model Settings

In `main.py`, you can adjust the OpenAI API settings:

```python
response = client.chat.completions.create(
    model="gpt-4o",           # Change model here
    max_tokens=1000,          # Adjust max response length
    temperature=0.7           # Adjust creativity (0.0-1.0)
)
```

### Parallel Processing

Adjust the number of parallel workers in `evaluate_all_developers()`:
```python
results = evaluate_all_developers(
    df, 
    prompt_template, 
    max_developers=args.limit, 
    output_file="evaluation_results.json", 
    max_workers=5  # Change this number
)
```

## Cost Estimation

The tool provides cost and time estimates before processing:
- Estimated time per request: ~2 seconds (with parallel processing)
- Estimated cost per request: ~$0.01 USD (for GPT-4o)
- Actual costs may vary based on resume length and API pricing

## Error Handling

- **Missing API Key**: The tool will prompt you to set up your API key
- **Invalid CSV**: Clear error messages for file not found or parsing errors
- **API Errors**: Handles rate limits, authentication errors, and API failures gracefully
- **Parsing Errors**: Failed evaluations are marked as "ERROR" in the output CSV

## Project Structure

```
CustomGPT/
├── main.py              # Main evaluation script
├── api_call.py         # OpenAI API wrapper functions
├── prompt.txt          # Evaluation prompt template
├── requirements.txt    # Python dependencies
├── SETUP.md           # Detailed setup instructions
├── README.md          # This file
└── .env               # Your API key (not in git)
```

## Troubleshooting

### "OpenAI API key is required"
- Ensure your `.env` file exists with `OPENAI_API_KEY=sk-...`
- Or set the environment variable in your terminal
- See [SETUP.md](SETUP.md) for detailed instructions

### "CSV file not found"
- Check that your CSV file path is correct
- Use `--csv` flag to specify the full path if needed
- Ensure the CSV has `developer_id` and `resume_plain` columns

### "Rate limit exceeded"
- The tool uses parallel processing which may hit rate limits
- Reduce `max_workers` in the code
- Wait a few minutes and try again

### Parsing errors in output
- Check that `prompt.txt` outputs valid JSON
- Verify `expected_fields` in `main.py` match the prompt output
- Review `evaluation_results.json` to see raw API responses

## Contributing

When modifying the evaluation criteria, follow the **Customization** section above for detailed instructions. Quick checklist:

1. **Update `prompt.txt`** with your new evaluation criteria and field definitions
2. **Update `expected_fields`** in `create_evaluation_dataframe()` function in `main.py` to match your prompt fields
3. **Update the docstring** in `create_evaluation_dataframe()` to reflect new fields
4. **Test with a small dataset first** (`--limit 5` or `--limit 1`)
5. **Verify output CSV** has all expected columns and correct data types
6. **Check `evaluation_results.json`** to ensure API responses match your expected format

**Remember**: Field names in your prompt's JSON output must exactly match the `expected_fields` list in `main.py`!

## License

This project is for internal use. Please ensure you comply with OpenAI's usage policies and data privacy requirements when processing resumes.

## Support

For issues or questions:
1. Check the error messages for specific guidance
2. Review `SETUP.md` for API key setup
3. Verify your CSV format matches the expected structure
4. Test with a single developer first: `python main.py --limit 1`

