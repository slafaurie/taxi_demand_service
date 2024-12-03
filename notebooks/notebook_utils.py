def insert_parent_in_path():
    import sys
    from pathlib import Path
    # Add the project root directory to Python path
    project_root = Path.cwd().parent
    sys.path.append(str(project_root))