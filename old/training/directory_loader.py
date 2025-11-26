"""
Directory loader for auto-discovery and loading of training files
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class DirectoryLoader:
    """Auto-discovery and loading of training files from directories"""
    
    def __init__(self, base_directory: str):
        self.base_directory = Path(base_directory)
        self.supported_extensions = {'.jsonl', '.json', '.txt', '.csv', '.tsv'}
        
    def discover_files(self, pattern: str = "*", recursive: bool = True) -> List[Path]:
        """Discover all supported training files in directory"""
        files = []
        
        if recursive:
            # Search recursively through subdirectories
            for ext in self.supported_extensions:
                pattern_with_ext = f"**/{pattern}{ext}"
                files.extend(self.base_directory.glob(pattern_with_ext))
        else:
            # Search only in current directory
            for ext in self.supported_extensions:
                pattern_with_ext = f"{pattern}{ext}"
                files.extend(self.base_directory.glob(pattern_with_ext))
        
        return sorted(files)
    
    def categorize_files(self) -> Dict[str, List[Path]]:
        """Categorize files by type based on filename patterns"""
        all_files = self.discover_files()
        categories = {
            'historical': [],
            'emotional': [],
            'conversational': [],
            'causal': [],
            'socratic': [],
            'general': []
        }
        
        for file_path in all_files:
            filename = file_path.name.lower()
            
            # Categorize based on filename patterns
            if any(keyword in filename for keyword in ['hist', 'historical', 'history', 'era', 'ancient', 'medieval']):
                categories['historical'].append(file_path)
            elif any(keyword in filename for keyword in ['emotion', 'movie', 'scene', 'feeling', 'amygdala']):
                categories['emotional'].append(file_path)
            elif any(keyword in filename for keyword in ['conv', 'chat', 'dialogue', 'conversation']):
                categories['conversational'].append(file_path)
            elif any(keyword in filename for keyword in ['causal', 'cause', 'effect', 'reasoning']):
                categories['causal'].append(file_path)
            elif any(keyword in filename for keyword in ['socratic', 'question', 'answer', 'qa']):
                categories['socratic'].append(file_path)
            else:
                categories['general'].append(file_path)
        
        return categories
    
    def load_file_preview(self, file_path: Path, max_lines: int = 3) -> Dict[str, Any]:
        """Load a preview of the file to understand its structure"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [f.readline().strip() for _ in range(max_lines) if f.readline()]
            
            # Try to parse as JSON/JSONL
            sample_data = []
            for line in lines:
                if line:
                    try:
                        data = json.loads(line)
                        sample_data.append(data)
                    except:
                        # Not JSON, treat as text
                        sample_data.append({'text': line})
            
            return {
                'file': str(file_path),
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'sample_data': sample_data,
                'estimated_records': self.estimate_record_count(file_path)
            }
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'size_mb': 0,
                'sample_data': [],
                'estimated_records': 0
            }
    
    def estimate_record_count(self, file_path: Path) -> int:
        """Estimate number of records in file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    return sum(1 for _ in f)
                elif file_path.suffix == '.json':
                    data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    else:
                        return 1
                else:
                    # For text files, estimate by lines
                    f.seek(0)
                    return sum(1 for _ in f)
        except:
            return 0
    
    def generate_loading_report(self) -> Dict[str, Any]:
        """Generate comprehensive loading report"""
        categories = self.categorize_files()
        
        # Calculate statistics
        total_files = sum(len(files) for files in categories.values())
        total_size = 0
        file_details = []
        
        for category, files in categories.items():
            category_size = 0
            for file_path in files:
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    category_size += size_mb
                    total_size += size_mb
                    
                    # Get file preview
                    preview = self.load_file_preview(file_path)
                    preview['category'] = category
                    file_details.append(preview)
                except Exception as e:
                    file_details.append({
                        'file': str(file_path),
                        'category': category,
                        'error': str(e),
                        'size_mb': 0,
                        'sample_data': [],
                        'estimated_records': 0
                    })
        
        # Update categories with size info
        for category, files in categories.items():
            categories[category] = {
                'files': [str(f) for f in files],
                'count': len(files),
                'total_size_mb': sum(f.stat().st_size / (1024 * 1024) for f in files if f.exists())
            }
        
        return {
            'base_directory': str(self.base_directory),
            'total_files': total_files,
            'total_size_mb': total_size,
            'categories': categories,
            'file_details': file_details
        }


def discover_and_preview(directory: str) -> Dict[str, Any]:
    """Discover and preview files in directory"""
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return {}
    
    loader = DirectoryLoader(directory)
    report = loader.generate_loading_report()
    
    print("📁 DIRECTORY DISCOVERY REPORT")
    print("=" * 50)
    print(f"📂 Base Directory: {report['base_directory']}")
    print(f"📊 Total Files: {report['total_files']}")
    
    if report['total_files'] == 0:
        print("⚠️  No supported files found (.jsonl, .json, .txt, .csv, .tsv)")
        return report
    
    print("\n🗂️  Categories:")
    for category, info in report['categories'].items():
        if info['count'] > 0:
            print(f"   📋 {category}: {info['count']} files ({info['total_size_mb']:.1f} MB)")
            for file_path in info['files'][:3]:  # Show first 3 files
                print(f"      • {Path(file_path).name}")
            if info['count'] > 3:
                print(f"      • ... and {info['count'] - 3} more")
    
    print("\n📖 File Previews:")
    for detail in report['file_details'][:5]:  # Show first 5 file previews
        print(f"   🔍 {Path(detail['file']).name} ({detail['category']})")
        print(f"      Size: {detail['size_mb']:.1f} MB | Est. Records: {detail['estimated_records']}")
        if detail.get('sample_data'):
            print(f"      Sample: {detail['sample_data'][0] if detail['sample_data'] else 'N/A'}")
        if detail.get('error'):
            print(f"      Error: {detail['error']}")
    
    return report
