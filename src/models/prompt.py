import json

def create_classification_prompt(mismatch_data):
    """
    Create a concise prompt for classifying author name mismatches.

    This function generates a prompt to help an LLM classify discrepancies
    between cited author names and their authoritative DBLP records.

    Args:
        mismatch_data: Dict containing mismatch information

    Returns:
        String prompt for LLM classification
    """
    
    # Extract key information for better context
    reference = mismatch_data.get("reference", {})
    dblp_match = mismatch_data.get("dblp_match", {})
    mismatches = mismatch_data.get("mismatches", [])
    error_classifications = mismatch_data.get("error_classifications", [])
    
    # Format authors lists for clarity
    citation_authors = reference.get("authors", [])
    if isinstance(citation_authors, str):
        citation_authors = [citation_authors]
    dblp_authors = dblp_match.get("authors", [])
    
    prompt = f"""You are classifying author name discrepancies in academic citations.

CRITICAL CONTEXT:

- DBLP names are ALWAYS authoritative (based on ORCID/author preference)
- Most differences are NOT typos - they are parser errors, nicknames, formatting issues, or cultural name variants
- TYPO is VERY RARE - use ONLY for clear 1–2 character spelling errors with NO truncation, NO missing parts
- TRUNCATION (name cut off) is ALWAYS PARSER_ERROR, never TYPO
- "Brocke" vs "Brockett", "Durre" vs "Durrett", "Amishra" vs "Bhattamishra" = TRUNCATION = PARSER_ERROR

CRITICAL RULES - ALWAYS CHECK FIRST:

1. If mismatch says "Author not found" or "author_not_found" → ALWAYS AUTHOR_MISSING (NEVER TYPO)
2. If mismatch says "author_order_wrong" or "Authors match but order differs" → ALWAYS AUTHOR_ORDER_ERROR (NEVER TYPO)
3. Consider parser errors FIRST before assuming typos

CLASSIFICATION CATEGORIES:

1. PARSER_ERROR - PDF/text extraction failures (COMMON)

   Pattern: Text incorrectly extracted from PDF, resulting in:
   - Names split across multiple author entries ("Zhixiang", "Eddie Xu" → "Zhixiang Eddie Xu")
   - Names merged incorrectly ("Steven Chu, Hong Hoi" → "Steven Chu-Hong Hoi")
   - Character position errors ("E Yves R Jean-Mary" → "Yves R. Jean-Mary")
   - Initials/punctuation misplaced
   - Name truncation creating different-looking name ("Kaiyuan" → "Karen")
   - Diacritics corrupted ("Gökhan" → "Gükhan", "Rémi" → "R'emi")
   - Last names truncated ("Wistuba" → "Wistub")

   Key indicators: Structural issues, name parts in wrong positions, truncation patterns

2. NICKNAME - Common short forms AND cultural name variants (COMMON)

   Pattern: Informal versions, shortened forms, OR English/Western names vs original names for the SAME person.
   Use NICKNAME especially for first-name mismatches where one name clearly looks like a short, familiar,
   or Anglicized form of the other.
   Examples: 
   - Nick/Nicholas, Liz/Elizabeth, Beth/Elizabeth, Bob/Robert, Rob/Robert, Rob/Robert J.
   - Dave/David, Mike/Michael, Bill/William, Misha/Michael, Tim/Timothy
   - Eric/Yunxuan, Henry/Henghui (English name vs Chinese name for same person)
   - Gbolahan/Femi (different forms of same cultural name)
   Key: The shorter or more informal name is plausibly a commonly used version of the longer or more formal
   given name, and the scholarly context suggests it is the same person (same co-authors, topic, venue, etc.).
   If DBLP uses the full given name and the citation uses a common short form, prefer NICKNAME over TYPO for
   first-name mismatches.

3. INITIAL_VS_FULL - Abbreviation differences (COMMON)

   Pattern: Initial vs spelled out, missing/added initials or prefixes
   Examples: "R Chen" → "Ricky Chen", "Shad Akhtar" → "Md. Shad Akhtar", "Quoc" → "Quoc V. Le"
   - Missing middle initial: "M Wistub" vs "Martin Wistuba" → INITIAL_VS_FULL (not TYPO)

4. MIDDLE_NAME - Reordering multi-part given names

   Pattern: Same name parts, different order
   Examples: "Shane Shixiang Gu" → "Shixiang Shane Gu"

5. LAST_NAME_ERROR - Surname parsing/formatting issues

   Pattern: Compound surnames split or merged incorrectly
   Examples: "Sabela Ramos" → "Sabela Ramos Garea" (compound surname truncated)

6. AUTHOR_MISSING - Authors omitted from citation

   Pattern: Citation has fewer authors OR includes non-person entities
   Examples: Missing co-authors, "Google Brain" listed as author name
   - "Author not found" mismatch → AUTHOR_MISSING

7. AUTHOR_ORDER_ERROR - Same authors, wrong sequence

   Pattern: All authors present but different order

8. TRANSLITERATION - Different romanization systems

   Pattern: Alternative spellings of same non-Latin name due to transliteration
   Examples: Different romanization systems for Arabic/Chinese/Korean names

9. TYPO - Small spelling error of 1–2 characters (VERY RARE - use extremely conservatively)

   Pattern: Only 1–2 characters differ, with no missing name parts, no truncation, and no structural issues.
   The shorter form must NOT simply be the longer form with parts removed or cut off.
   Examples:
   - "Paranjabe" → "Paranjape" (b→p), single letter substitution
   - "Jon" → "John" (missing h), one extra character in the longer form

   Prefer NICKNAME or INITIAL_VS_FULL for first-name mismatches where the shorter name is a common short form
   of the longer (e.g., Nick/Nicholas, Liz/Elizabeth, Rob/Robert). TYPO is for small spelling mistakes within
   the same name form, not for short/long versions of the same given name.
   
   CRITICAL: Do NOT use TYPO if:
   - Mismatch says "Author not found" or "author_not_found" → MUST BE AUTHOR_MISSING
   - Mismatch says "author_order_wrong" or "order differs" → MUST BE AUTHOR_ORDER_ERROR
   - Name is truncated (e.g., "Brocke" vs "Brockett", "Durre" vs "Durrett") → PARSER_ERROR
   - Name is cut off (e.g., "Amishra" vs "Bhattamishra", "Lemoyer" vs "Zettlemoyer") → PARSER_ERROR
   - Missing initials or name parts (e.g., "Edward E" vs "Edward Grefenstette") → PARSER_ERROR or INITIAL_VS_FULL
   - Could be nickname or a common short form → NICKNAME
   - Structural parsing issues → PARSER_ERROR
   - Multiple differences → PARSER_ERROR
   
   Rule: If the shorter name could be the longer name with parts removed, it's PARSER_ERROR, not TYPO
   Rule: "Author not found" mismatches are ALWAYS AUTHOR_MISSING, never TYPO

10. WRONG_PERSON - Genuinely different people (RARE)

    Pattern: Citation attributes work to completely wrong author
    Examples: Citing work by "Jun Deng" when it's actually "Jia Deng" (different scholar)
    Note: DBLP numbers (e.g., "0001") just help disambiguation, don't indicate wrong person

11. DEADNAME - Author's previous name (VERY RARE - use only with strong evidence)

    Pattern: Name the author has explicitly changed from and no longer uses
    Examples: "Keith Bonawitz" → "Kallista Bonawitz" (documented name change)
    WARNING: Do NOT assume based on name "gender". Use AMBIGUOUS if unsure.

12. NAME_CHANGE_OTHER - Marriage/cultural/religious name change

    Pattern: Documented name change for non-trans reasons
    Examples: Marriage surname changes

13. AMBIGUOUS - Insufficient information to classify

CLASSIFICATION WORKFLOW:

Step 1: Check mismatch type from automated detection FIRST (MANDATORY)
- "Author not found" OR "author_not_found" → MUST BE AUTHOR_MISSING (NEVER TYPO, NEVER PARSER_ERROR)
- "author_order_wrong" OR "Authors match but order differs" → MUST BE AUTHOR_ORDER_ERROR (NEVER TYPO)
- "first_name_mismatch" → likely NICKNAME, INITIAL_VS_FULL, or PARSER_ERROR (almost never TYPO)
- "last_name_mismatch" → likely PARSER_ERROR, LAST_NAME_ERROR, INITIAL_VS_FULL, or rarely TYPO

IMPORTANT: If the mismatch field contains "Author not found" or "author_order", use that category immediately. Do NOT classify as TYPO.

Step 2: Look for PARSER_ERROR patterns FIRST (most common)
- Name split/merged across entries → PARSER_ERROR
- Initials in wrong position → PARSER_ERROR
- TRUNCATION (name cut off) → PARSER_ERROR
  * "Brocke" vs "Brockett" = truncation = PARSER_ERROR
  * "Durre" vs "Durrett" = truncation = PARSER_ERROR
  * "Amishra" vs "Bhattamishra" = truncation = PARSER_ERROR
  * "Lemoyer" vs "Zettlemoyer" = truncation = PARSER_ERROR
- Diacritics corrupted → PARSER_ERROR
- Structural issues → PARSER_ERROR
- Missing name parts → PARSER_ERROR or INITIAL_VS_FULL

Step 3: Check for INITIAL_VS_FULL
- Missing middle initial/name → INITIAL_VS_FULL
- "M Wistub" vs "Martin Wistuba" → INITIAL_VS_FULL (missing initial AND truncation)

Step 4: Check for NICKNAME patterns
- Common nicknames or short forms (Nick/Nicholas, Liz/Elizabeth, Dave/David, Mike/Michael) → NICKNAME
- English vs cultural/original name (e.g., Eric/Yunxuan) → NICKNAME
- For first_name_mismatch, if one form clearly looks like a shorter or informal version of the other,
  you should strongly prefer NICKNAME over TYPO.

Step 5: TYPO only if ALL of these are true:
- Only 1–2 characters differ between the two forms
- NO truncation (shorter name is NOT a prefix of longer name)
- NO missing parts
- NO parsing issues
- Clear spelling error with no other explanation
- If shorter name could be longer name with parts removed → NOT TYPO, use PARSER_ERROR

OUTPUT FORMAT:

Provide a brief reasoning (max 15 words) explaining your classification, then output the category.

IMPORTANT: Write actual reasoning text, NOT placeholder text like "[brief explanation]". Write real explanation.

Format:
Reasoning: [write your actual explanation here]
RESULT: [CATEGORY]

Example output:
Reasoning: Last name truncated from Wistuba to Wistub during PDF extraction
RESULT: PARSER_ERROR

EXAMPLES:

Citation authors: ["Misha Laskin"]
DBLP authors: ["Michael Laskin"]
Mismatch: first_name_mismatch: Misha vs Michael
Reasoning: Misha is common nickname for Michael
RESULT: NICKNAME

Citation authors: ["Karen Yang"]
DBLP authors: ["Kaiyuan Yang 0001"]
Mismatch: first_name_mismatch: Karen vs Kaiyuan
Reasoning: Parser truncated Kaiyuan to Karen during PDF extraction
RESULT: PARSER_ERROR

Citation authors: ["M Wistub"]
DBLP authors: ["Martin Wistuba"]
Mismatch: last_name_mismatch: M Wistub vs Martin Wistuba
Reasoning: Missing initial Martin and last name truncated Wistuba to Wistub
RESULT: PARSER_ERROR

Citation authors: ["Paranjabe"]
DBLP authors: ["Paranjape"]
Mismatch: last_name_mismatch: Paranjabe vs Paranjape
Reasoning: Single letter typo, b changed to p, no truncation
RESULT: TYPO

Citation authors: ["Zhixiang", "Eddie Xu"]
DBLP authors: ["Zhixiang Eddie Xu"]
Mismatch: first_name_mismatch
Reasoning: Parser split single name into multiple author entries
RESULT: PARSER_ERROR

Citation authors: ["Quoc"]
DBLP authors: ["Quoc V. Le"]
Mismatch: first_name_mismatch: Quoc vs Quoc V. Le
Reasoning: Missing middle initial and last name
RESULT: INITIAL_VS_FULL

Citation authors: ["Philipp Koehn"]
DBLP authors: ["Hany Hassan", "Kareem Darwish"]
Mismatch: Author not found in DBLP: Philipp Koehn
Reasoning: Citation author not found in DBLP database
RESULT: AUTHOR_MISSING

Citation authors: ["A", "B", "C"]
DBLP authors: ["B", "A", "C"]
Mismatch: Authors match but order differs
Reasoning: Same authors present but in different order
RESULT: AUTHOR_ORDER_ERROR

DATA TO CLASSIFY:

Citation Reference:
Title: {reference.get('title', 'N/A')}
Authors (as parsed from citation): {json.dumps(citation_authors, indent=2)}

DBLP Match (authoritative):
Authors (from DBLP): {json.dumps(dblp_authors, indent=2)}

Automated Detection:
Mismatches detected: {json.dumps(mismatches, indent=2)}
Error classifications: {json.dumps(error_classifications, indent=2)}

Analyze the discrepancy between the citation authors and DBLP authors. Consider the automated mismatch detection output. Classify the discrepancy:"""

    
    return prompt