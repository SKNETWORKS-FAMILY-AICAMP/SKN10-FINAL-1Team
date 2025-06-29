import os
import re
import yaml
import boto3
from pocketflow import Node, BatchNode
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from crawl_github_files import crawl_github_files
from call_llm import call_llm
from dotenv import load_dotenv
import shutil

# Import for progress tracking
try:
    from accounts.models import ScanTask
except ImportError:
    # Fallback for when running as standalone
    ScanTask = None


def update_task_progress(task_id, stage, step):
    """Update task progress if task_id is provided"""
    if task_id and ScanTask:
        try:
            task = ScanTask.objects.get(id=task_id)
            task.update_progress(stage, step)
            print(f"✅ Progress updated: Task {task_id} - Stage: {stage}, Step: {step}, Progress: {task.progress}%")
        except ScanTask.DoesNotExist:
            print(f"Warning: Task {task_id} not found for progress update")
        except Exception as e:
            print(f"Warning: Failed to update task progress: {e}")
    else:
        print(f"⚠️ No task_id provided or ScanTask not available. task_id: {task_id}, ScanTask: {ScanTask}")


# Helper to get content for specific file indices
def get_content_for_indices(files_data, indices):
    content_map = {}
    for i in indices:
        if 0 <= i < len(files_data):
            path, content = files_data[i]
            content_map[f"{i} # {path}"] = (
                content  # Use index + path as key for context
            )
    return content_map


class FetchRepo(Node):
    def prep(self, shared):
        repo_url = shared.get("repo_url")
        local_dir = shared.get("local_dir")
        project_name = shared.get("project_name")

        if not project_name:
            # Basic name derivation from URL or directory
            if repo_url:
                from urllib.parse import urlparse
                from pathlib import Path
                path_parts = Path(urlparse(repo_url).path.strip('/')).parts
                if len(path_parts) >= 2:
                    project_name = path_parts[1].replace(".git", "")
                else:
                    # Fallback for unexpected URL formats
                    project_name = repo_url.split("/")[-1].replace(".git", "")
            else:
                project_name = os.path.basename(os.path.abspath(local_dir))
            shared["project_name"] = project_name

        # Get file patterns directly from shared
        include_patterns = shared["include_patterns"]
        exclude_patterns = shared["exclude_patterns"]
        max_file_size = shared["max_file_size"]

        return {
            "repo_url": repo_url,
            "local_dir": local_dir,
            "token": shared.get("github_token"),
            "include_patterns": include_patterns,
            "exclude_patterns": exclude_patterns,
            "max_file_size": max_file_size,
            "use_relative_paths": True,
        }

    def exec(self, prep_res):
        if not prep_res["repo_url"]:
            raise ValueError("Repository URL (--repo) is required.")

        print(f"Crawling repository: {prep_res['repo_url']}...")
        result = crawl_github_files(
            repo_url=prep_res["repo_url"],
            token=prep_res["token"],
            include_patterns=prep_res["include_patterns"],
            exclude_patterns=prep_res["exclude_patterns"],
            max_file_size=prep_res["max_file_size"],
            use_relative_paths=prep_res["use_relative_paths"],
        )

        # Convert dict to list of tuples: [(path, content), ...]
        files_list = list(result.get("files", {}).items())
        if len(files_list) == 0:
            raise (ValueError("Failed to fetch files"))
        print(f"Fetched {len(files_list)} files.")
        return files_list

    def post(self, shared, prep_res, exec_res):
        shared["files"] = exec_res  # List of (path, content) tuples
        # Update progress - Step 1: Fetching Repository
        update_task_progress(shared.get("task_id"), "fetching", 1)


class IdentifyAbstractions(Node):
    def prep(self, shared):
        print("--- RUNNING MODIFIED IdentifyAbstractions.prep ---")
        files_data = shared["files"]
        project_name = shared["project_name"]
        language = shared.get("language", "english")
        use_cache = shared.get("use_cache", True)
        max_abstraction_num = shared.get("max_abstraction_num", 10)
        # New parameter for context size limit, can be added to shared/args later
        max_context_chars = shared.get("max_context_chars", 100000)
        print(f"--- Max context characters: {max_context_chars} ---")

        # --- Start: Added for debugging context size ---
        full_context_for_debug = ""
        for i, (path, content) in enumerate(files_data):
            full_context_for_debug += f"--- File Index {i}: {path} ---\n{content}\n\n"
        print(f"--- Full context size (before truncation): {len(full_context_for_debug)} chars ---")
        # --- End: Added for debugging context size ---

        def create_llm_context(files_data, max_chars):
            context = ""
            file_info = []  # Store tuples of (index, path)
            included_files_count = 0
            for i, (path, content) in enumerate(files_data):
                entry = f"--- File Index {i}: {path} ---\n{content}\n\n"
                if len(context) + len(entry) > max_chars:
                    print(
                        f"Warning: Context size limit ({max_chars} chars) reached. "
                        f"Stopping at file index {i-1}. "
                        f"{len(files_data) - included_files_count} files were not included in the context."
                    )
                    break
                context += entry
                file_info.append((i, path))
                included_files_count += 1
            return context, file_info

        context, file_info = create_llm_context(files_data, max_context_chars)
        print(f"--- Truncated context size: {len(context)} chars ---")

        # The file listing for the prompt should only include files that are actually in the context
        file_listing_for_prompt = "\n".join(
            [f"- {idx} # {path}" for idx, path in file_info]
        )
        return (
            context,
            file_listing_for_prompt,
            len(file_info), # BUG FIX: Use the count of files actually included in the context
            project_name,
            language,
            use_cache,
            max_abstraction_num,
        )

    def exec(self, prep_res):
        (
            context,
            file_listing_for_prompt,
            file_count,
            project_name,
            language,
            use_cache,
            max_abstraction_num,
        ) = prep_res  # Unpack all parameters

        # --- Start LLM Prompt for Abstractions ---
        prompt = f"""
Analyze the following codebase for the project '{project_name}'.

Available files (total {file_count}):
{file_listing_for_prompt}

Full context of all files:
{context}

Based on the provided codebase, identify the key abstractions that are central to understanding this project. 
These abstractions should represent the core components, modules, or concepts.

Desired output format is a YAML list of objects, where each object has:
- 'name': A concise name for the abstraction (in {language}).
- 'description': A brief explanation of what this abstraction represents and its role (in {language}).
- 'file_indices': A list of integer file indices that are most relevant to this abstraction. Choose from the file list above.

Return at most {max_abstraction_num} key abstractions.

Example for a different project (simple web server):
```yaml
- name: "HTTP 요청 핸들러 (HTTP Request Handler)"
  description: "수신 HTTP 요청을 처리하고 적절한 응답을 생성하는 구성 요소입니다. (Component that processes incoming HTTP requests and generates appropriate responses.)"
  file_indices: [0, 2]
- name: "라우팅 설정 (Routing Configuration)"
  description: "URL 경로를 특정 요청 핸들러 함수에 매핑하는 규칙을 정의합니다. (Defines rules for mapping URL paths to specific request handler functions.)"
  file_indices: [1]
```

Your response should be only the YAML list, enclosed in triple backticks (```yaml ... ```).
Ensure the output is valid YAML.
"""
        # --- End LLM Prompt ---

        print(f"Identifying abstractions for {project_name} (in {language})...")
        llm_response = call_llm(prompt, use_cache=use_cache)
        print(f"LLM response for abstractions:\n{llm_response}")

        abstractions = []
        try:
            # Try to parse as YAML first
            # Remove backticks and 'yaml' language identifier if present
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```yaml"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            abstractions = yaml.safe_load(cleaned_response)
            if not isinstance(abstractions, list):
                abstractions = [] # Ensure it's a list
                print("Warning: LLM response for abstractions was not a YAML list.")

        except yaml.YAMLError as e:
            print(f"YAML parsing failed for abstractions: {e}. Falling back to regex.")
            # Fallback to regex if YAML parsing fails (basic example)
            # This regex is very basic and might need to be adjusted based on common LLM output variations.
            # It looks for items starting with '- name:' and tries to capture name, description, and file_indices.
            pattern = re.compile(
                r"- name: (?P<name>.*?)\n\s*description: (?P<description>.*?)\n\s*file_indices: (?P<file_indices>.*?)(?=\n- name:|$)",
                re.DOTALL,
            )
            for match in pattern.finditer(llm_response):
                try:
                    name = match.group("name").strip().replace('"', '')
                    description = match.group("description").strip().replace('"', '')
                    # Process file_indices carefully, as it might be a string like "[0, 1]"
                    indices_str = match.group("file_indices").strip()
                    # Remove brackets and split by comma, then convert to int
                    file_indices = [
                        int(i.strip())
                        for i in indices_str.strip("[]").split(",")
                        if i.strip().isdigit()
                    ]
                    abstractions.append({
                        "name": name,
                        "description": description,
                        "file_indices": file_indices,
                    })
                except Exception as regex_err:
                    print(f"Error parsing abstraction with regex: {regex_err}")
        
        if not abstractions:
            print("Warning: Could not parse any abstractions from LLM response.")
            # As a last resort, if no abstractions are parsed, we might return an empty list
            # or a default structure if that's preferable to failing the flow.
            # For now, we'll return an empty list, which downstream nodes should handle.

        print(f"Identified {len(abstractions)} abstractions.")
        return abstractions

    def post(self, shared, prep_res, exec_res):
        shared["abstractions"] = exec_res
        # Update progress - Step 2: Identifying Abstractions
        update_task_progress(shared.get("task_id"), "identifying", 2)


class AnalyzeRelationships(Node):
    def prep(self, shared):
        print("--- RUNNING AnalyzeRelationships.prep ---")
        abstractions = shared["abstractions"]
        files_data = shared["files"]
        project_name = shared["project_name"]
        language = shared.get("language", "english")
        use_cache = shared.get("use_cache", True)
        num_abstractions = len(abstractions)

        context = "Identified Abstractions:\n"
        all_relevant_indices = set()
        abstraction_info_for_prompt = []
        for i, abstr in enumerate(abstractions):
            file_indices_str = ", ".join(map(str, abstr["file_indices"]))
            info_line = f"- Index {i}: {abstr['name']} (Relevant file indices: [{file_indices_str}])\n  Description: {abstr['description']}"
            context += info_line + "\n"
            abstraction_info_for_prompt.append(f"{i} # {abstr['name']}")
            all_relevant_indices.update(abstr["file_indices"])

        print(f"--- All relevant indices: {all_relevant_indices} ---")

        context += "\nRelevant File Snippets (Referenced by Index and Path):\n"
        relevant_files_content_map = get_content_for_indices(
            files_data, sorted(list(all_relevant_indices))
        )
        file_context_str = "\n\n".join(
            f"--- File: {idx_path} ---\n{content}"
            for idx_path, content in relevant_files_content_map.items()
        )
        context += file_context_str

        return (
            context,
            abstraction_info_for_prompt,
            num_abstractions,
            project_name,
            language,
            use_cache,
        )

    def exec(self, prep_res):
        (
            context,
            abstraction_info_for_prompt,
            num_abstractions,
            project_name,
            language,
            use_cache,
        ) = prep_res

        abstraction_listing = "\n".join(abstraction_info_for_prompt)

        print(f"Analyzing relationships using LLM...")

        language_instruction = ""
        lang_hint = ""
        list_lang_note = ""
        if language.lower() != "english":
            language_instruction = f"IMPORTANT: Generate the `summary` and relationship `label` fields in **{language.capitalize()}** language. Do NOT use English for these fields.\n\n"
            lang_hint = f" (in {language.capitalize()})"
            list_lang_note = f" (Names might be in {language.capitalize()})"

        prompt = f"""
Based on the following abstractions and relevant code snippets from the project `{project_name}`:

List of Abstraction Indices and Names{list_lang_note}:
{abstraction_listing}

Context (Abstractions, Descriptions, Code):
{context}

{language_instruction}Please provide:
1. A high-level `summary` of the project's main purpose and functionality in a few beginner-friendly sentences{lang_hint}. Use markdown formatting with **bold** and *italic* text to highlight important concepts.
2. A list (`relationships`) describing the key interactions between these abstractions. For each relationship, specify:
    - `from_abstraction`: Index of the source abstraction (e.g., `0 # AbstractionName1`)
    - `to_abstraction`: Index of the target abstraction (e.g., `1 # AbstractionName2`)
    - `label`: A brief label for the interaction **in just a few words**{lang_hint} (e.g., "Manages", "Inherits", "Uses").
    Ideally the relationship should be backed by one abstraction calling or passing parameters to another.
    Simplify the relationship and exclude those non-important ones.

IMPORTANT: Make sure EVERY abstraction is involved in at least ONE relationship (either as source or target). Each abstraction index must appear at least once across all relationships.

Format the output as YAML:

```yaml
summary: |
  A brief, simple explanation of the project{lang_hint}.
  Can span multiple lines with **bold** and *italic* for emphasis.
relationships:
  - from_abstraction: 0 # AbstractionName1
    to_abstraction: 1 # AbstractionName2
    label: "Manages"{lang_hint}
  - from_abstraction: 2 # AbstractionName3
    to_abstraction: 0 # AbstractionName1
    label: "Provides config"{lang_hint}
  # ... other relationships
```

Now, provide the YAML output:
"""
        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0))

        yaml_str = response.strip().split("```yaml")[1].split("```")[0].strip()
        relationships_data = yaml.safe_load(yaml_str)

        if not isinstance(relationships_data, dict) or not all(
            k in relationships_data for k in ["summary", "relationships"]
        ):
            raise ValueError(
                "LLM output is not a dict or missing keys ('summary', 'relationships')"
            )
        if not isinstance(relationships_data["summary"], str):
            raise ValueError("summary is not a string")
        if not isinstance(relationships_data["relationships"], list):
            raise ValueError("relationships is not a list")

        validated_relationships = []
        for rel in relationships_data["relationships"]:
            if not isinstance(rel, dict) or not all(
                k in rel for k in ["from_abstraction", "to_abstraction", "label"]
            ):
                raise ValueError(
                    f"Missing keys (expected from_abstraction, to_abstraction, label) in relationship item: {rel}"
                )
            if not isinstance(rel["label"], str):
                raise ValueError(f"Relationship label is not a string: {rel}")

            try:
                from_idx = int(str(rel["from_abstraction"]).split("#")[0].strip())
                to_idx = int(str(rel["to_abstraction"]).split("#")[0].strip())
                if not (
                    0 <= from_idx < num_abstractions and 0 <= to_idx < num_abstractions
                ):
                    raise ValueError(
                        f"Invalid index in relationship: from={from_idx}, to={to_idx}. Max index is {num_abstractions-1}."
                    )
                validated_relationships.append(
                    {
                        "from": from_idx,
                        "to": to_idx,
                        "label": rel["label"],
                    }
                )
            except (ValueError, TypeError):
                raise ValueError(f"Could not parse indices from relationship: {rel}")

        print("Generated project summary and relationship details.")
        return {
            "summary": relationships_data["summary"],
            "details": validated_relationships,
        }

    def post(self, shared, prep_res, exec_res):
        shared["relationships"] = exec_res
        # Update progress - Step 3: Analyzing Relationships
        update_task_progress(shared.get("task_id"), "analyzing", 3)


class OrderChapters(Node):
    def prep(self, shared):
        abstractions = shared["abstractions"]  # Name/description might be translated
        relationships = shared["relationships"]  # Summary/label might be translated
        project_name = shared["project_name"]  # Get project name
        language = shared.get("language", "english")  # Get language
        use_cache = shared.get("use_cache", True)  # Get use_cache flag, default to True

        # Prepare context for the LLM
        abstraction_info_for_prompt = []
        for i, a in enumerate(abstractions):
            abstraction_info_for_prompt.append(
                f"- {i} # {a['name']}"
            )  # Use potentially translated name
        abstraction_listing = "\n".join(abstraction_info_for_prompt)

        # Use potentially translated summary and labels
        summary_note = ""
        if language.lower() != "english":
            summary_note = (
                f" (Note: Project Summary might be in {language.capitalize()})"
            )

        context = f"Project Summary{summary_note}:\n{relationships['summary']}\n\n"
        context += "Relationships (Indices refer to abstractions above):\n"
        for rel in relationships["details"]:
            from_name = abstractions[rel["from"]]["name"]
            to_name = abstractions[rel["to"]]["name"]
            # Use potentially translated 'label'
            context += f"- From {rel['from']} ({from_name}) to {rel['to']} ({to_name}): {rel['label']}\n"  # Label might be translated

        list_lang_note = ""
        if language.lower() != "english":
            list_lang_note = f" (Names might be in {language.capitalize()})"

        return (
            abstraction_listing,
            context,
            len(abstractions),
            project_name,
            list_lang_note,
            use_cache,
        )  # Return use_cache

    def exec(self, prep_res):
        (
            abstraction_listing,
            context,
            num_abstractions,
            project_name,
            list_lang_note,
            use_cache,
        ) = prep_res  # Unpack use_cache
        print("Determining chapter order using LLM...")
        # No language variation needed here in prompt instructions, just ordering based on structure
        # The input names might be translated, hence the note.
        prompt = f"""
Given the following project abstractions and their relationships for the project ```` {project_name} ````:

Abstractions (Index # Name){list_lang_note}:
{abstraction_listing}

Context about relationships and project summary:
{context}

If you are going to make a tutorial for ```` {project_name} ````, what is the best order to explain these abstractions, from first to last?
Ideally, first explain those that are the most important or foundational, perhaps user-facing concepts or entry points. Then move to more detailed, lower-level implementation details or supporting concepts.

Output the ordered list of abstraction indices, including the name in a comment for clarity. Use the format `idx # AbstractionName`.

```yaml
- 2 # FoundationalConcept
- 0 # CoreClassA
- 1 # CoreClassB (uses CoreClassA)
- ...
```

Now, provide the YAML output:
"""
        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0)) # Use cache only if enabled and not retrying

        # --- Validation ---
        yaml_str = response.strip().split("```yaml")[1].split("```")[0].strip()
        ordered_indices_raw = yaml.safe_load(yaml_str)

        if not isinstance(ordered_indices_raw, list):
            raise ValueError("LLM output is not a list")

        ordered_indices = []
        seen_indices = set()
        for entry in ordered_indices_raw:
            try:
                if isinstance(entry, int):
                    idx = entry
                elif isinstance(entry, str) and "#" in entry:
                    idx = int(entry.split("#")[0].strip())
                else:
                    idx = int(str(entry).strip())

                if not (0 <= idx < num_abstractions):
                    raise ValueError(
                        f"Invalid index {idx} in ordered list. Max index is {num_abstractions-1}."
                    )
                if idx in seen_indices:
                    raise ValueError(f"Duplicate index {idx} found in ordered list.")
                ordered_indices.append(idx)
                seen_indices.add(idx)

            except (ValueError, TypeError):
                raise ValueError(
                    f"Could not parse index from ordered list entry: {entry}"
                )

        # Check if all abstractions are included
        if len(ordered_indices) != num_abstractions:
            raise ValueError(
                f"Ordered list length ({len(ordered_indices)}) does not match number of abstractions ({num_abstractions}). Missing indices: {set(range(num_abstractions)) - seen_indices}"
            )

        print(f"Determined chapter order (indices): {ordered_indices}")
        return ordered_indices  # Return the list of indices

    def post(self, shared, prep_res, exec_res):
        # exec_res is already the list of ordered indices
        shared["chapter_order"] = exec_res
        # Update progress - Step 4: Ordering Chapters
        update_task_progress(shared.get("task_id"), "ordering", 4)


class WriteChapters(BatchNode):
    def prep(self, shared):
        chapter_order = shared["chapter_order"]  # List of indices
        abstractions = shared[
            "abstractions"
        ]  # List of {"name": str, "description": str, "files": [int]}
        files_data = shared["files"]  # List of (path, content) tuples
        project_name = shared["project_name"]
        language = shared.get("language", "english")
        use_cache = shared.get("use_cache", True)  # Get use_cache flag, default to True

        # Get already written chapters to provide context
        # We store them temporarily during the batch run, not in shared memory yet
        # The 'previous_chapters_summary' will be built progressively in the exec context
        self.chapters_written_so_far = (
            []
        )  # Use instance variable for temporary storage across exec calls

        # Create a complete list of all chapters
        all_chapters = []
        chapter_filenames = {}  # Store chapter filename mapping for linking
        for i, abstraction_index in enumerate(chapter_order):
            if 0 <= abstraction_index < len(abstractions):
                chapter_num = i + 1
                chapter_name = abstractions[abstraction_index][
                    "name"
                ]  # Potentially translated name
                # Create safe filename (from potentially translated name)
                safe_name = "".join(
                    c if c.isalnum() else "_" for c in chapter_name
                ).lower()
                filename = f"{i+1:02d}_{safe_name}.md"
                # Format with link (using potentially translated name)
                all_chapters.append(f"{chapter_num}. [{chapter_name}]({filename})")
                # Store mapping of chapter index to filename for linking
                chapter_filenames[abstraction_index] = {
                    "num": chapter_num,
                    "name": chapter_name,
                    "filename": filename,
                }

        # Create a formatted string with all chapters
        full_chapter_listing = "\n".join(all_chapters)

        items_to_process = []
        for i, abstraction_index in enumerate(chapter_order):
            if 0 <= abstraction_index < len(abstractions):
                abstraction_details = abstractions[
                    abstraction_index
                ]  # Contains potentially translated name/desc
                # Use 'files' (list of indices) directly
                related_file_indices = abstraction_details.get("files", [])
                # Get content using helper, passing indices
                related_files_content_map = get_content_for_indices(
                    files_data, related_file_indices
                )

                # Get previous chapter info for transitions (uses potentially translated name)
                prev_chapter = None
                if i > 0:
                    prev_idx = chapter_order[i - 1]
                    prev_chapter = chapter_filenames[prev_idx]

                # Get next chapter info for transitions (uses potentially translated name)
                next_chapter = None
                if i < len(chapter_order) - 1:
                    next_idx = chapter_order[i + 1]
                    next_chapter = chapter_filenames[next_idx]

                items_to_process.append(
                    {
                        "chapter_num": i + 1,
                        "abstraction_index": abstraction_index,
                        "abstraction_details": abstraction_details,  # Has potentially translated name/desc
                        "related_files_content_map": related_files_content_map,
                        "project_name": shared["project_name"],  # Add project name
                        "full_chapter_listing": full_chapter_listing,  # Add the full chapter listing (uses potentially translated names)
                        "chapter_filenames": chapter_filenames,  # Add chapter filenames mapping (uses potentially translated names)
                        "prev_chapter": prev_chapter,  # Add previous chapter info (uses potentially translated name)
                        "next_chapter": next_chapter,  # Add next chapter info (uses potentially translated name)
                        "language": language,  # Add language for multi-language support
                        "use_cache": use_cache, # Pass use_cache flag
                        # previous_chapters_summary will be added dynamically in exec
                    }
                )
            else:
                print(
                    f"Warning: Invalid abstraction index {abstraction_index} in chapter_order. Skipping."
                )

        print(f"Preparing to write {len(items_to_process)} chapters...")
        return items_to_process  # Iterable for BatchNode

    def exec(self, item):
        # This runs for each item prepared above
        abstraction_name = item["abstraction_details"][
            "name"
        ]  # Potentially translated name
        abstraction_description = item["abstraction_details"][
            "description"
        ]  # Potentially translated description
        chapter_num = item["chapter_num"]
        project_name = item.get("project_name")
        language = item.get("language", "english")
        use_cache = item.get("use_cache", True) # Read use_cache from item
        print(f"Writing chapter {chapter_num} for: {abstraction_name} using LLM...")

        # Prepare file context string from the map
        file_context_str = "\n\n".join(
            f"--- File: {idx_path.split('# ')[1] if '# ' in idx_path else idx_path} ---\n{content}"
            for idx_path, content in item["related_files_content_map"].items()
        )

        # Get summary of chapters written *before* this one
        # Use the temporary instance variable
        previous_chapters_summary = "\n---\n".join(self.chapters_written_so_far)

        # Add language instruction and context notes only if not English
        language_instruction = ""
        concept_details_note = ""
        structure_note = ""
        prev_summary_note = ""
        instruction_lang_note = ""
        mermaid_lang_note = ""
        code_comment_note = ""
        link_lang_note = ""
        tone_note = ""
        if language.lower() != "english":
            lang_cap = language.capitalize()
            language_instruction = f"IMPORTANT: Write this ENTIRE tutorial chapter in **{lang_cap}**. Some input context (like concept name, description, chapter list, previous summary) might already be in {lang_cap}, but you MUST translate ALL other generated content including explanations, examples, technical terms, and potentially code comments into {lang_cap}. DO NOT use English anywhere except in code syntax, required proper nouns, or when specified. The entire output MUST be in {lang_cap}.\n\n"
            concept_details_note = f" (Note: Provided in {lang_cap})"
            structure_note = f" (Note: Chapter names might be in {lang_cap})"
            prev_summary_note = f" (Note: This summary might be in {lang_cap})"
            instruction_lang_note = f" (in {lang_cap})"
            mermaid_lang_note = f" (Use {lang_cap} for labels/text if appropriate)"
            code_comment_note = f" (Translate to {lang_cap} if possible, otherwise keep minimal English for clarity)"
            link_lang_note = (
                f" (Use the {lang_cap} chapter title from the structure above)"
            )
            tone_note = f" (appropriate for {lang_cap} readers)"

        prompt = f"""
{language_instruction}Write a very beginner-friendly tutorial chapter (in Markdown format) for the project `{project_name}` about the concept: "{abstraction_name}". This is Chapter {chapter_num}.

Concept Details{concept_details_note}:
- Name: {abstraction_name}
- Description:
{abstraction_description}

Complete Tutorial Structure{structure_note}:
{item["full_chapter_listing"]}

Context from previous chapters{prev_summary_note}:
{previous_chapters_summary if previous_chapters_summary else "This is the first chapter."}

Relevant Code Snippets (Code itself remains unchanged):
{file_context_str if file_context_str else "No specific code snippets provided for this abstraction."}

Instructions for the chapter (Generate content in {language.capitalize()} unless specified otherwise):
- Start with a clear heading (e.g., `# Chapter {chapter_num}: {abstraction_name}`). Use the provided concept name.

- If this is not the first chapter, begin with a brief transition from the previous chapter{instruction_lang_note}, referencing it with a proper Markdown link using its name{link_lang_note}.

- Begin with a high-level motivation explaining what problem this abstraction solves{instruction_lang_note}. Start with a central use case as a concrete example. The whole chapter should guide the reader to understand how to solve this use case. Make it very minimal and friendly to beginners.

- If the abstraction is complex, break it down into key concepts. Explain each concept one-by-one in a very beginner-friendly way{instruction_lang_note}.

- Explain how to use this abstraction to solve the use case{instruction_lang_note}. Give example inputs and outputs for code snippets (if the output isn't values, describe at a high level what will happen{instruction_lang_note}).

- Each code block should be BELOW 10 lines! If longer code blocks are needed, break them down into smaller pieces and walk through them one-by-one. Aggresively simplify the code to make it minimal. Use comments{code_comment_note} to skip non-important implementation details. Each code block should have a beginner friendly explanation right after it{instruction_lang_note}.

- Describe the internal implementation to help understand what's under the hood{instruction_lang_note}. First provide a non-code or code-light walkthrough on what happens step-by-step when the abstraction is called{instruction_lang_note}. It's recommended to use a simple sequenceDiagram with a dummy example - keep it minimal with at most 5 participants to ensure clarity. If participant name has space, use: `participant QP as Query Processing`. {mermaid_lang_note}.

- Then dive deeper into code for the internal implementation with references to files. Provide example code blocks, but make them similarly simple and beginner-friendly. Explain{instruction_lang_note}.

- IMPORTANT: When you need to refer to other core abstractions covered in other chapters, ALWAYS use proper Markdown links like this: [Chapter Title](filename.md). Use the Complete Tutorial Structure above to find the correct filename and the chapter title{link_lang_note}. Translate the surrounding text.

- Use mermaid diagrams to illustrate complex concepts (```mermaid``` format). {mermaid_lang_note}.

- Heavily use analogies and examples throughout{instruction_lang_note} to help beginners understand.

- End the chapter with a brief conclusion that summarizes what was learned{instruction_lang_note} and provides a transition to the next chapter{instruction_lang_note}. If there is a next chapter, use a proper Markdown link: [Next Chapter Title](next_chapter_filename){link_lang_note}.

- Ensure the tone is welcoming and easy for a newcomer to understand{tone_note}.

- Output *only* the Markdown content for this chapter.

Now, directly provide a super beginner-friendly Markdown output (DON'T need ```markdown``` tags):
"""
        chapter_content = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0)) # Use cache only if enabled and not retrying
        # Basic validation/cleanup
        actual_heading = f"# Chapter {chapter_num}: {abstraction_name}"  # Use potentially translated name
        if not chapter_content.strip().startswith(f"# Chapter {chapter_num}"):
            # Add heading if missing or incorrect, trying to preserve content
            lines = chapter_content.strip().split("\n")
            if lines and lines[0].strip().startswith(
                "#"
            ):  # If there's some heading, replace it
                lines[0] = actual_heading
                chapter_content = "\n".join(lines)
            else:  # Otherwise, prepend it
                chapter_content = f"{actual_heading}\n\n{chapter_content}"

        # Add the generated content to our temporary list for the next iteration's context
        self.chapters_written_so_far.append(chapter_content)

        return chapter_content  # Return the Markdown string (potentially translated)

    def post(self, shared, prep_res, exec_res_list):
        # exec_res_list contains the generated Markdown for each chapter, in order
        shared["chapters"] = exec_res_list
        # Update progress - Step 5: Writing Chapters
        update_task_progress(shared.get("task_id"), "writing", 5)


class CombineTutorial(Node):
    def prep(self, shared):
        project_name = shared["project_name"]
        output_base_dir = shared.get("output_dir", "output")  # Default output dir
        output_path = os.path.join(output_base_dir, project_name)
        repo_url = shared.get("repo_url")  # Get the repository URL

        # Get potentially translated data
        relationships_data = shared[
            "relationships"
        ]  # {"summary": str, "details": [{"from": int, "to": int, "label": str}]} -> summary/label potentially translated
        chapter_order = shared["chapter_order"]  # indices
        abstractions = shared[
            "abstractions"
        ]  # list of dicts -> name/description potentially translated
        chapters_content = shared[
            "chapters"
        ]  # list of strings -> content potentially translated

        # --- Generate Mermaid Diagram ---
        mermaid_lines = ["flowchart TD"]
        for i, abstr in enumerate(abstractions):
            node_id = f"A{i}"
            sanitized_name = abstr["name"].replace('"', "")
            node_label = sanitized_name
            mermaid_lines.append(f'    {node_id}["{node_label}"]')

        for rel in relationships_data["details"]:
            from_node_id = f"A{rel['from']}"
            to_node_id = f"A{rel['to']}"
            edge_label = rel["label"].replace('"', "").replace("\n", " ")
            max_label_len = 30
            if len(edge_label) > max_label_len:
                edge_label = edge_label[: max_label_len - 3] + "..."
            mermaid_lines.append(f'    {from_node_id} -- "{edge_label}" --> {to_node_id}')

        mermaid_diagram = "\n".join(mermaid_lines)

        # --- Prepare index.md content ---
        index_content = f"# Tutorial: {project_name}\n\n"
        index_content += f"{relationships_data['summary']}\n\n"
        index_content += f"**Source Repository:** [{repo_url}]({repo_url})\n\n"
        index_content += "```mermaid\n"
        index_content += mermaid_diagram + "\n"
        index_content += "```\n\n"
        index_content += f"## Chapters\n\n"

        chapter_files = []
        for i, abstraction_index in enumerate(chapter_order):
            if 0 <= abstraction_index < len(abstractions) and i < len(chapters_content):
                abstraction_name = abstractions[abstraction_index]["name"]
                safe_name = "".join(c if c.isalnum() else "_" for c in abstraction_name).lower()
                filename = f"{i+1:02d}_{safe_name}.md"
                index_content += f"{i+1}. [{abstraction_name}]({filename})\n"

                chapter_content = chapters_content[i]
                if not chapter_content.endswith("\n\n"):
                    chapter_content += "\n\n"
                chapter_content += f"---\n\nGenerated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)"

                chapter_files.append({"filename": filename, "content": chapter_content})
            else:
                print(f"Warning: Mismatch between chapter order, abstractions, or content at index {i} (abstraction index {abstraction_index}). Skipping file generation for this entry.")

        index_content += f"\n\n---\n\nGenerated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)"

        return {
            "output_path": output_path,
            "project_name": project_name,
            "repo_url": repo_url,
            "index_content": index_content,
            "chapter_files": chapter_files,
            "s3_bucket_name": shared.get("s3_bucket_name"),
            "aws_access_key_id": shared.get("aws_access_key_id"),
            "aws_secret_access_key": shared.get("aws_secret_access_key"),
            "aws_region_name": shared.get("aws_region_name"),
            "user_id": shared.get("user_id"),
            "github_user_name": shared.get("github_user_name"),
        }

    def exec(self, prep_res):
        output_path = prep_res["output_path"]
        project_name = prep_res["project_name"]
        repo_url = prep_res.get("repo_url")  # Add null check for repo_url
        index_content = prep_res["index_content"]
        chapter_files = prep_res["chapter_files"]

        print(f"Combining tutorial into directory: {output_path}")
        # Rely on Node's built-in retry/fallback
        os.makedirs(output_path, exist_ok=True)

        # Write index.md
        index_filepath = os.path.join(output_path, "index.md")
        with open(index_filepath, "w", encoding="utf-8") as f:
            f.write(index_content)
        print(f"  - Wrote {index_filepath}")

        # Write chapter files
        for chapter_info in chapter_files:
            chapter_filepath = os.path.join(output_path, chapter_info["filename"])
            with open(chapter_filepath, "w", encoding="utf-8") as f:
                f.write(chapter_info["content"])
            print(f"  - Wrote {chapter_filepath}")

        # --- S3 Upload Logic for Generated Tutorial --- #
        s3_bucket_name = prep_res.get("s3_bucket_name")
        aws_access_key_id = prep_res.get("aws_access_key_id")
        aws_secret_access_key = prep_res.get("aws_secret_access_key")
        aws_region_name = prep_res.get("aws_region_name")
        github_user_name = prep_res.get("github_user_name")
        project_name = prep_res.get("project_name")
        repo_url = prep_res.get("repo_url")

        # Extract branch name from repo_url (assuming format is https://github.com/user/repo/tree/branch)
        branch_name = 'main' # default
        if repo_url and '/tree/' in repo_url:
            branch_name = repo_url.split('/tree/')[-1]

        if s3_bucket_name and aws_access_key_id and aws_secret_access_key and aws_region_name:
            print(f"S3에 튜토리얼 업로드 시작: {s3_bucket_name}")
            try:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region_name
                )

                # Upload index and chapter files
                all_files_to_upload = [index_filepath] + [os.path.join(output_path, chapter['filename']) for chapter in chapter_files]

                for local_path in all_files_to_upload:
                    file_name = os.path.basename(local_path)
                    s3_object_key = f"user_github_repos/{github_user_name}/{project_name}/{branch_name}/llm_tutorial/{file_name}"
                    
                    print(f"Uploading '{file_name}' to S3 -> {s3_object_key}")
                    s3_client.upload_file(local_path, s3_bucket_name, s3_object_key)
                
                print(f"Tutorial S3 upload complete.")
            except Exception as e:
                print(f"S3 upload failed: {e}")

        return output_path  # Return the final path

    def post(self, shared, prep_res, exec_res):
        shared["final_output_dir"] = exec_res
        # Update progress - Step 6: Combining Tutorial
        update_task_progress(shared.get("task_id"), "combining", 6)


class UploadToPinecone(Node):
    def prep(self, shared):
        print("--- RUNNING UploadToPinecone.prep ---")
        # Load environment variables from .env file
        load_dotenv()

        final_output_dir = shared.get("final_output_dir")
        repo_url = shared.get("repo_url")
        project_name = shared.get("project_name")
        github_user_name = shared.get("github_user_name")
        user_id = shared.get("user_id")  # 로그인 유저 ID 추가

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

        # Debug information
        print(f"DEBUG: final_output_dir = {final_output_dir}")
        print(f"DEBUG: repo_url = {repo_url}")
        print(f"DEBUG: project_name = {project_name}")
        print(f"DEBUG: github_user_name = {github_user_name}")
        print(f"DEBUG: user_id = {user_id}")
        print(f"DEBUG: PINECONE_API_KEY = {'Set' if pinecone_api_key else 'Not Set'}")
        print(f"DEBUG: PINECONE_ENVIRONMENT = {pinecone_environment}")

        if not all([final_output_dir, repo_url, project_name, github_user_name, pinecone_api_key, pinecone_environment]):
            missing_items = []
            if not final_output_dir: missing_items.append("final_output_dir")
            if not repo_url: missing_items.append("repo_url")
            if not project_name: missing_items.append("project_name")
            if not github_user_name: missing_items.append("github_user_name")
            if not pinecone_api_key: missing_items.append("PINECONE_API_KEY")
            if not pinecone_environment: missing_items.append("PINECONE_ENVIRONMENT")
            
            print(f"Skipping Pinecone upload due to missing configuration or data: {', '.join(missing_items)}")
            return None

        # Extract branch name from repo_url
        branch_name = 'main'  # Default
        if '/tree/' in repo_url:
            branch_name = repo_url.split('/tree/')[-1]

        # Extract owner from repo_url (could be username or organization)
        from urllib.parse import urlparse
        try:
            parsed_url = urlparse(repo_url)
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 2:
                owner = path_parts[0]  # This could be username or organization
            else:
                owner = github_user_name  # Fallback to github_user_name
        except Exception:
            owner = github_user_name  # Fallback to github_user_name

        return {
            "final_output_dir": final_output_dir,
            "repo_url": repo_url,
            "project_name": project_name,
            "github_user_name": github_user_name,
            "branch_name": branch_name,
            "user_id": user_id,  # 로그인 유저 ID 추가
            "owner": owner,  # 실제 레포지터리 소유자 (username 또는 organization)
            "pinecone_api_key": pinecone_api_key,
            "pinecone_environment": pinecone_environment,
        }

    def exec(self, prep_res):
        if prep_res is None:
            return "Pinecone upload skipped."

        from pinecone import Pinecone, ServerlessSpec
        from openai import OpenAI
        import uuid
        import urllib.parse

        # Unpack prep results
        final_output_dir = prep_res["final_output_dir"]
        project_name = prep_res["project_name"]
        github_user_name = prep_res["github_user_name"]
        branch_name = prep_res["branch_name"]
        user_id = prep_res["user_id"]  # 로그인 유저 ID 추가
        owner = prep_res["owner"]  # 실제 레포지터리 소유자
        pinecone_api_key = prep_res["pinecone_api_key"]
        pinecone_environment = prep_res["pinecone_environment"]

        try:
            # 1. Initialize Pinecone
            pc = Pinecone(api_key=pinecone_api_key)
            index_name = "code-documents"  # 언더스코어를 하이픈으로 변경

            # Correct common region name typos
            if pinecone_environment == "us-east1":
                print("Correcting pinecone_environment from 'us-east1' to 'us-east-1'")
                pinecone_environment = "us-east-1"

            # 2. Check if index exists, create if not
            if index_name not in pc.list_indexes().names():
                print(f"Creating Pinecone index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=3072,  # Dimension for 'text-embedding-3-large'
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region=pinecone_environment)
                )
            
            index = pc.Index(index_name)

            # 3. Initialize OpenAI client
            # The client will automatically use the OPENAI_API_KEY from your environment
            client = OpenAI()
            embedding_model = "text-embedding-3-large"

            # 4. Read generated tutorial files
            md_files = [f for f in os.listdir(final_output_dir) if f.endswith('.md')]
            
            vectors_to_upsert = []
            namespace = str(user_id) if user_id else "anonymous"  # 네임스페이스를 로그인 유저명으로 변경

            for filename in md_files:
                file_path = os.path.join(final_output_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 5. Create metadata with new structure
                original_document_id = f"{project_name}_{uuid.uuid4().hex}"
                metadata = {
                    'github_user_repo': f"{owner}/{project_name}",  # 실제 레포지터리 소유자/레포지터리명
                    'branch_name': branch_name,  # 브랜치명
                    'original_filename': filename,  # 원본 파일 이름
                    'original_document_id': original_document_id,  # 원본 파일 ID
                    'text': content  # 원본 텍스트
                }

                # 6. Embed content using OpenAI
                response = client.embeddings.create(
                    input=content,
                    model=embedding_model
                )
                embedding = response.data[0].embedding

                # 7. Prepare vector for upsert by encoding the filename to be ASCII-safe
                safe_filename = urllib.parse.quote(filename)
                vector_id = f"{safe_filename}_{uuid.uuid4().hex}"
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })

            # 8. Upsert to Pinecone
            if vectors_to_upsert:
                print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone index '{index_name}' with namespace '{namespace}'...")
                index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                print("Upsert complete.")
            else:
                print("No documents to upsert.")

            return f"Successfully uploaded {len(vectors_to_upsert)} documents to Pinecone."

        except Exception as e:
            print(f"An error occurred during Pinecone upload: {e}")
            # Potentially re-raise or handle the error as needed
            raise e

    def post(self, shared, prep_res, exec_res):
        print(exec_res)

        # Delete local tutorial files after successful upload
        if isinstance(exec_res, str) and exec_res.startswith("Successfully uploaded"):
            final_output_dir = prep_res.get("final_output_dir")
            if final_output_dir and os.path.exists(final_output_dir):
                try:
                    shutil.rmtree(final_output_dir)
                    print(f"Successfully deleted local tutorial directory: {final_output_dir}")
                except Exception as e:
                    print(f"Error deleting local tutorial directory {final_output_dir}: {e}")

        # Update progress - Step 7: Uploading to Pinecone
        update_task_progress(shared.get("task_id"), "uploading", 7)