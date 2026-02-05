"""
Utility to split a large markdown file by H1 headers.
This creates separate files per chapter/section, which can improve:
- Incremental updates (only changed chapters re-ingested)
- Source attribution (clearer file names in citations)
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging


def sanitize_filename(name: str) -> str:
    """
    Convert a header to a safe filename.
    
    Args:
        name: Header text
        
    Returns:
        Safe filename string
    """
    # Remove non-alphanumeric characters except spaces and dashes
    name = re.sub(r'[^\w\s-]', '', name)
    # Replace spaces with underscores
    name = name.strip().replace(' ', '_')
    # Limit length
    if len(name) > 50:
        name = name[:50]
    return name.lower()


def split_by_h1(content: str) -> List[Tuple[str, str]]:
    """
    Split markdown content by H1 headers.
    
    Args:
        content: Full markdown text
        
    Returns:
        List of (title, section_content) tuples
    """
    # Pattern to match H1 headers: # Title or #Title (at line start)
    h1_pattern = r'(?:^|\n)#[ \t]+([^\n]+)(?:\n|$)'
    
    # Find all H1 headers
    matches = list(re.finditer(h1_pattern, content))
    
    if not matches:
        logging.warning("No H1 headers found. Treating as single section.")
        return [("full_document", content)]
    
    sections = []
    
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start_pos = match.start()
        
        # Find end position (start of next H1 or end of content)
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        section_content = content[start_pos:end_pos].strip()
        
        sections.append((title, section_content))
    
    return sections


def split_markdown_file(
    input_path: str,
    output_dir: str,
    prefix: str = "",
    dry_run: bool = False,
) -> int:
    """
    Split a markdown file into multiple files by H1 headers.
    
    Args:
        input_path: Path to input markdown file
        output_dir: Directory for output files
        prefix: Optional prefix for filenames
        dry_run: If True, only show what would be done
        
    Returns:
        Number of sections created
    """
    logging.info(f"Reading: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    logging.info(f"File size: {len(content)} characters")
    
    # Split by H1
    sections = split_by_h1(content)
    
    logging.info(f"Found {len(sections)} H1 sections")
    
    if dry_run:
        print("\nDRY RUN - Would create files:\n")
    
    # Create output directory
    if not dry_run:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Write sections
    for i, (title, section_content) in enumerate(sections, 1):
        safe_name = sanitize_filename(title)
        
        if prefix:
            filename = f"{prefix}_{i:03d}_{safe_name}.md"
        else:
            filename = f"{i:03d}_{safe_name}.md"
        
        output_path = Path(output_dir) / filename
        
        if dry_run:
            print(f"  {filename}")
            print(f"    Title: {title}")
            print(f"    Size: {len(section_content)} chars")
            print()
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(section_content)
            logging.info(f"Created: {output_path}")
    
    return len(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Split a large markdown file by H1 headers"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input markdown file path",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for split files",
    )
    parser.add_argument(
        "--prefix", "-p",
        default="",
        help="Prefix for output filenames (e.g., 'chapter')",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without creating files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.INFO)
    
    # Validate input
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Run split
    count = split_markdown_file(
        input_path=args.input,
        output_dir=args.output,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )
    
    if args.dry_run:
        print(f"\nWould create {count} files in: {args.output}")
    else:
        logging.info(f"Successfully created {count} files in: {args.output}")
        print(f"\nNext steps:")
        print(f"  1. Update config.yaml:")
        print(f'     data_path: "{args.output}"')
        print(f"  2. Run: python scripts/ingest.py")


if __name__ == "__main__":
    main()
