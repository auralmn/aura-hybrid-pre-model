#!/usr/bin/env python3
"""
Script to fix incorrect imports in the Aura codebase.
Changes 'from X import' to 'from X import' for proper package structure.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix imports in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace 'from X import' with 'from X import'
        content = re.sub(r'from src\.([a-zA-Z_][a-zA-Z0-9_.]*)', r'from \1', content)
        
        # Replace 'import X' with 'import X'
        content = re.sub(r'import src\.([a-zA-Z_][a-zA-Z0-9_.]*)', r'import \1', content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function to fix imports across the codebase."""
    src_dir = Path(__file__).parent
    
    # Find all Python files in src/ directory
    python_files = list(src_dir.rglob('*.py'))
    
    fixed_count = 0
    processed_count = 0
    
    print(f"Found {len(python_files)} Python files")
    print("Fixing imports...")
    
    for filepath in python_files:
        processed_count += 1
        if fix_imports_in_file(filepath):
            fixed_count += 1
            print(f"✓ Fixed: {filepath.relative_to(src_dir)}")
    
    print("\n" + "="*60)
    print(f"Summary:")
    print(f"  Processed: {processed_count} files")
    print(f"  Fixed:     {fixed_count} files")
    print("="*60)

if __name__ == "__main__":
    main()
