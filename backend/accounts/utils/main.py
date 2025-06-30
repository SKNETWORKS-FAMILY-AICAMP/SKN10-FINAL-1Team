import dotenv
import os
import argparse
import sys

# Add the parent directory to sys.path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # backend directory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the function that creates the flow
try:
    from accounts.utils.flow import create_tutorial_flow
except ImportError:
    # Fallback for when running as main module
    pass
from .flow import create_tutorial_flow

dotenv.load_dotenv()

# Default file patterns
DEFAULT_INCLUDE_PATTERNS = {
    "*.py", "*.js", "*.jsx", "*.ts", "*.tsx", "*.go", "*.java", "*.pyi", "*.pyx",
    "*.c", "*.cc", "*.cpp", "*.h", "*.md", "*.rst", "*Dockerfile",
    "*Makefile", "*.yaml", "*.yml",
}

DEFAULT_EXCLUDE_PATTERNS = {
    "assets/*", "data/*", "images/*", "public/*", "static/*", "temp/*",
    "*docs/*",
    "*venv/*",
    "*.venv/*",
    "*test*",
    "*tests/*",
    "*examples/*",
    "v1/*",
    "*dist/*",
    "*build/*",
    "*experimental/*",
    "*deprecated/*",
    "*misc/*",
    "*legacy/*",
    ".git/*", ".github/*", ".next/*", ".vscode/*",
    "*obj/*",
    "*bin/*",
    "*node_modules/*",
    "*.log"
}

# --- Main Function ---
def main(repo_url=None, project_name=None, github_token=None, output_dir="output", 
         include_patterns=None, exclude_patterns=None, max_file_size=100000,
         language="english", use_cache=True, max_abstractions=10, max_context_chars=100000,
         s3_bucket=None, s3_access_key=None, s3_secret_key=None, s3_region=None,
         user_id=None, github_user_name=None, task_id=None):
    """
    Main function for tutorial generation. Can be called with direct parameters
    instead of command line arguments.
    """
    
    # Import here to avoid circular imports
    from accounts.models import ScanTask
    
    # Update task status to running if task_id is provided
    if task_id:
        try:
            task = ScanTask.objects.get(id=task_id)
            task.status = 'running'
            task.save()
        except ScanTask.DoesNotExist:
            print(f"Warning: Task {task_id} not found")
    
    # If no repo_url provided, try to parse from command line (for backward compatibility)
    if repo_url is None:
        parser = argparse.ArgumentParser(description="Generate a tutorial for a GitHub codebase or local directory.")
        parser.add_argument("--repo", help="URL of the public GitHub repository.", required=True)
        parser.add_argument("-n", "--name", help="Project name (optional, derived from repo/directory if omitted).")
        parser.add_argument("-t", "--token", help="GitHub personal access token (optional, reads from GITHUB_TOKEN env var if not provided).")
        parser.add_argument("-o", "--output", default="output", help="Base directory for output (default: ./output).")
        parser.add_argument("-i", "--include", nargs="+", help="Include file patterns (e.g. '*.py' '*.js'). Defaults to common code files if not specified.")
        parser.add_argument("-e", "--exclude", nargs="+", help="Exclude file patterns (e.g. 'tests/*' 'docs/*'). Defaults to test/build directories if not specified.")
        parser.add_argument("-s", "--max-size", type=int, default=100000, help="Maximum file size in bytes (default: 100000, about 100KB).")
        parser.add_argument("--language", default="english", help="Language for the generated tutorial (default: english)")
        parser.add_argument("--no-cache", action="store_true", help="Disable LLM response caching (default: caching enabled)")
        parser.add_argument("--max-abstractions", type=int, default=10, help="Maximum number of abstractions to identify (default: 10)")
        parser.add_argument("--max-context-chars", type=int, default=100000, help="Maximum number of characters for the LLM context (default: 100000).")
        parser.add_argument("--s3-bucket", help="S3 bucket name for uploading the generated tutorial.")
        parser.add_argument("--s3-access-key", help="AWS Access Key ID for S3 upload.")
        parser.add_argument("--s3-secret-key", help="AWS Secret Access Key for S3 upload.")
        parser.add_argument("--s3-region", help="AWS region for the S3 bucket.")
        parser.add_argument("--user-id", help="User identifier for constructing S3 path.")
        parser.add_argument("--github-user-name", help="GitHub username for constructing S3 path.")
        parser.add_argument("--task-id", help="Task ID for progress tracking.")

        args = parser.parse_args()

        # Use command line arguments
        repo_url = args.repo
        project_name = args.name
        github_token = args.token or os.environ.get('GITHUB_TOKEN')
        output_dir = args.output
        include_patterns = set(args.include) if args.include else DEFAULT_INCLUDE_PATTERNS
        exclude_patterns = set(args.exclude) if args.exclude else DEFAULT_EXCLUDE_PATTERNS
        max_file_size = args.max_size
        language = args.language
        use_cache = not args.no_cache
        max_abstractions = args.max_abstractions
        max_context_chars = args.max_context_chars
        s3_bucket = args.s3_bucket
        s3_access_key = args.s3_access_key
        s3_secret_key = args.s3_secret_key
        s3_region = args.s3_region
        user_id = args.user_id
        github_user_name = args.github_user_name
        task_id = args.task_id
    else:
        # Use provided parameters, with defaults
        if include_patterns is None:
            include_patterns = DEFAULT_INCLUDE_PATTERNS
        if exclude_patterns is None:
            exclude_patterns = DEFAULT_EXCLUDE_PATTERNS
        if github_token is None:
            github_token = os.environ.get('GITHUB_TOKEN')

    if not github_token:
        print("Warning: No GitHub token provided. You might hit rate limits for public repositories.")

    # Initialize the shared dictionary with inputs
    shared = {
        "repo_url": repo_url,
        "project_name": project_name, # Can be None, FetchRepo will derive it
        "github_token": github_token,
        "output_dir": output_dir, # Base directory for CombineTutorial output

        # Add include/exclude patterns and max file size
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
        "max_file_size": max_file_size,

        # Add language for multi-language support
        "language": language,
        
        # Add use_cache flag (inverse of no-cache flag)
        "use_cache": use_cache,
        
        # Add max_abstraction_num parameter
        "max_abstraction_num": max_abstractions,

        # Add context character limit
        "max_context_chars": max_context_chars,

        # Add S3 details to shared context if provided
        "s3_bucket_name": s3_bucket,
        "aws_access_key_id": s3_access_key,
        "aws_secret_access_key": s3_secret_key,
        "aws_region_name": s3_region,
        "user_id": user_id,
        "github_user_name": github_user_name,
        
        # Add task_id for progress tracking
        "task_id": task_id,

        # Outputs will be populated by the nodes
        "files": [],
        "abstractions": [],
        "relationships": {},
        "chapter_order": [],
        "chapters": [],
        "final_output_dir": None
    }

    # Display starting message with repository and language
    print(f"Starting tutorial generation for: {repo_url} in {language.capitalize()} language")
    print(f"LLM caching: {'Disabled' if not use_cache else 'Enabled'}")
    print(f"Task ID for progress tracking: {task_id}")

    try:
        # Create the flow instance
        tutorial_flow = create_tutorial_flow()

        # Run the flow
        tutorial_flow.run(shared)
        
        # Mark task as completed if task_id is provided
        if task_id:
            try:
                task = ScanTask.objects.get(id=task_id)
                task.mark_completed()
            except ScanTask.DoesNotExist:
                print(f"Warning: Task {task_id} not found for completion")
                
    except Exception as e:
        pass

if __name__ == "__main__":
    main() 