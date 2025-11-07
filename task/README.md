# Citation Verification Task

## Overview

This task requires you to verify the accuracy of author names in academic paper citations. You will be given a JSON file containing extracted citations from research papers, and your task is to query external databases (arXiv, DBLP, or Semantic Scholar) to verify if the author names are correct.


## Provided Files

- **`citations.json`**: A JSON file containing ~3,000 citations extracted from academic papers. Each citation includes:
  - `title`: Paper title
  - `authors`: List of author names (may be incomplete or incorrect)
  - `proceedings`: Conference/journal name
  - `year`: Publication year

- **`sample/`**: Directory containing the original XML files from which citations were extracted (for reference)

## Task Description

For each citation in `citations.json`, you need to:

1. **Query external databases** using the paper title to find the correct author information
2. **Compare** the authors listed in the JSON file with the authors found in the databases
3. **Identify discrepancies** such as:
   - Missing authors
   - Incorrect author names (spelling errors, name variations)
   - Wrong author order
   - Completely incorrect author lists

4. **Generate a report** documenting:
   - Which citations were verified
   - Which databases were used for each citation
   - Any discrepancies found
   - Corrected author information

## API Resources

You can use any combination of the following APIs to verify citations:

### 1. arXiv API

**Documentation**: https://arxiv.org/help/api/user-manual

**Python Library**: `arxiv` (install via `pip install arxiv`)

**Example Usage**:
```python
import arxiv

# Search for a paper by title
client = arxiv.Client()
search = arxiv.Search(
    query="Adversarial examples are a natural consequence of test error in noise",
    max_results=5,
    sort_by=arxiv.SortCriterion.Relevance
)

for result in client.results(search):
    print(f"Title: {result.title}")
    print(f"Authors: {[author.name for author in result.authors]}")
    print(f"Year: {result.published.year}")
    break
```

**Rate Limits**: Be respectful - add delays between requests (e.g., 1-3 seconds)

### 2. DBLP API

**Documentation**: https://dblp.org/faq/How+to+use+the+dblp+search+API.html

**Base URL**: `https://dblp.org/search/publ/api`

**Example Usage**:
```python
import requests
from urllib.parse import quote

# Search DBLP by title
title = "Statistical approaches to computer-assisted translation"
query = quote(title)
url = f"https://dblp.org/search/publ/api?q={query}&format=json&h=5"

response = requests.get(url)
data = response.json()

# Parse results
if 'result' in data and 'hits' in data['result']:
    for hit in data['result']['hits']['hit']:
        info = hit['info']
        print(f"Title: {info.get('title', 'N/A')}")
        print(f"Authors: {info.get('authors', {}).get('author', [])}")
        print(f"Year: {info.get('year', 'N/A')}")
```

**Rate Limits**: Add delays (1-2 seconds) between requests

### 3. Semantic Scholar API

**Documentation**: https://api.semanticscholar.org/api-docs/

**Base URL**: `https://api.semanticscholar.org/graph/v1`

**Example Usage**:
```python
import requests

# Search by title
title = "Adversarial examples are a natural consequence of test error in noise"
url = "https://api.semanticscholar.org/graph/v1/paper/search"
params = {
    "query": title,
    "limit": 5,
    "fields": "title,authors,year"
}

response = requests.get(url, params=params)
data = response.json()

# Parse results
if 'data' in data:
    for paper in data['data']:
        print(f"Title: {paper.get('title', 'N/A')}")
        authors = [author.get('name', '') for author in paper.get('authors', [])]
        print(f"Authors: {authors}")
        print(f"Year: {paper.get('year', 'N/A')}")
```

**Rate Limits**: 
- Free tier: 100 requests per 5 minutes
- Consider using API key for higher limits (optional)

## Expected Output Format

Create a JSON file `verification_results.json` with the following structure:

```json
[
  {
    "original_citation": {
      "title": "Paper title",
      "authors": ["Author 1", "Author 2"],
      "proceedings": "Conference name",
      "year": 2019
    },
    "verified_authors": {
      "source": "arxiv|dblp|semantic_scholar",
      "authors": ["Verified Author 1", "Verified Author 2"],
      "match_score": 0.95,
      "discrepancies": [
        {
          "type": "missing_author|incorrect_name|wrong_order",
          "original": "Author Name",
          "corrected": "Correct Author Name",
          "explanation": "Brief explanation"
        }
      ]
    },
    "verification_status": "verified|discrepancy_found|not_found"
  }
]
```

## Implementation Guidelines

### 1. Code Structure

Organize your code into modules:
- `parser.py`: Parse the citations.json file
- `api_clients.py`: Functions to query arXiv, DBLP, and Semantic Scholar
- `verifier.py`: Main logic to compare authors and identify discrepancies
- `main.py`: Entry point that orchestrates the verification process

### 2. Error Handling

- Handle network errors gracefully
- Implement retry logic for failed API requests
- Handle cases where papers are not found in any database
- Handle rate limiting appropriately

### 3. Author Name Matching

Consider these challenges when comparing author names:
- **Name variations**: "John Smith" vs "J. Smith" vs "John A. Smith"
- **Order differences**: Author order may vary
- **Missing authors**: Some authors might be missing from the original citation
- **Spelling errors**: Typos in names

You may want to use fuzzy string matching (e.g., `fuzzywuzzy` library) to handle name variations.

### 4. Progress Tracking

- Save progress periodically (e.g., after every 100 citations)
- Implement resume functionality in case the script is interrupted
- Log which citations were processed and which failed

### 5. Performance Considerations

- Use appropriate delays between API calls to respect rate limits
- Consider parallel processing (but be mindful of rate limits)
- Cache results to avoid re-querying the same papers

## Example Workflow

1. **Load citations**: Read `citations.json` and parse all citations
2. **For each citation**:
   - Extract the paper title
   - Query one or more APIs (start with arXiv, fallback to DBLP or Semantic Scholar)
   - Find the best matching paper (use title similarity)
   - Extract author information from the matched paper
   - Compare authors with the original citation
   - Document any discrepancies
3. **Generate report**: Create `verification_results.json` with all findings
4. **Create summary**: Generate a summary report (markdown or PDF) with statistics




## Tips

- **Start small**: Test your code on a subset of citations first (e.g., first 10-20)
- **Use multiple sources**: If one API doesn't have the paper, try another
- **Handle edge cases**: Papers with very common titles, papers not in any database, etc.
- **Document your approach**: Explain why you chose certain matching strategies
- **Be respectful**: Add appropriate delays between API calls

## Getting Started

1. Install required dependencies:
   ```bash
   pip install arxiv requests fuzzywuzzy python-Levenshtein
   ```

2. Read the `citations.json` file to understand the data structure

3. Start with a simple script that queries arXiv for one citation

4. Gradually build up to handle all citations with proper error handling


