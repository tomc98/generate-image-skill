#!/usr/bin/env python3
"""
Image Generation Script using Google's Nano Banana 2 (Gemini 3.1 Flash Image) model.

Usage:
    python3 generate.py "A sunset over mountains" --name "sunset" [--aspect 16:9]
    python3 generate.py "Make this image more vibrant" --name "vibrant-photo" --reference photo.png
    python3 generate.py "Combine these styles" --name "merged" --reference img1.png img2.png
"""

import argparse
import base64
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Configuration
MODEL = "gemini-3.1-flash-image-preview"  # Nano Banana 2
API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
DEFAULT_OUTPUT_DIR = Path(os.environ.get("IMAGE_OUTPUT_DIR", Path.cwd()))

VALID_ASPECTS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


def get_api_key() -> str:
    """Get API key from environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        print("\nTo fix this, add to ~/.claude/settings.json:", file=sys.stderr)
        print('  "env": { "GEMINI_API_KEY": "your-key-here" }', file=sys.stderr)
        print("\nGet your API key at: https://aistudio.google.com/apikey", file=sys.stderr)
        sys.exit(1)
    return api_key


def load_reference_image(image_path: Path) -> dict:
    """Load and encode a reference image for the API."""
    if not image_path.exists():
        print(f"Error: Reference image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Determine MIME type
    suffix = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/png")

    # Read and encode
    image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": image_data
        }
    }


def generate_image(prompt: str, aspect_ratio: str = "1:1", reference_images: Optional[List[Path]] = None) -> bytes:
    """Call the Gemini API to generate an image."""
    api_key = get_api_key()

    # Build parts list - reference images first, then text prompt
    parts = []

    if reference_images:
        for img_path in reference_images:
            parts.append(load_reference_image(img_path))

    parts.append({"text": prompt})

    # Build request payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio
            }
        }
    }

    # Make API request
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    request = urllib.request.Request(
        API_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else "No details"
        print(f"Error: API request failed with status {e.code}", file=sys.stderr)
        print(f"Details: {error_body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Error: Network error - {e.reason}", file=sys.stderr)
        sys.exit(1)

    # Extract image data from response
    try:
        candidates = result.get("candidates", [])
        if not candidates:
            print("Error: No candidates in response", file=sys.stderr)
            print(f"Response: {json.dumps(result, indent=2)}", file=sys.stderr)
            sys.exit(1)

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            if "inlineData" in part:
                image_data = part["inlineData"]
                if image_data.get("mimeType", "").startswith("image/"):
                    return base64.b64decode(image_data["data"])

        print("Error: No image data found in response", file=sys.stderr)
        print(f"Parts received: {[list(p.keys()) for p in parts]}", file=sys.stderr)
        sys.exit(1)

    except (KeyError, IndexError) as e:
        print(f"Error: Unexpected response format - {e}", file=sys.stderr)
        print(f"Response: {json.dumps(result, indent=2)}", file=sys.stderr)
        sys.exit(1)


def save_image(image_bytes: bytes, output_path: Path) -> Path:
    """Save image bytes to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    return output_path


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    # Replace spaces with hyphens, remove invalid characters
    import re
    name = name.lower().strip()
    name = re.sub(r'[^\w\s-]', '', name)  # Remove non-word chars except spaces/hyphens
    name = re.sub(r'[\s_]+', '-', name)   # Replace spaces/underscores with hyphens
    name = re.sub(r'-+', '-', name)       # Collapse multiple hyphens
    name = name.strip('-')                # Remove leading/trailing hyphens
    return name[:50] if name else "image" # Limit length, fallback to "image"


def generate_filename(name: Optional[str] = None) -> str:
    """Generate a filename, using provided name or timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if name:
        safe_name = sanitize_filename(name)
        return f"{safe_name}-{timestamp}.png"
    return f"generated-{timestamp}.png"


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Google's Nano Banana 2 model"
    )
    parser.add_argument(
        "prompt",
        help="The image generation prompt"
    )
    parser.add_argument(
        "--name",
        help="Descriptive name for the image file (e.g., 'sunset-mountains')"
    )
    parser.add_argument(
        "--aspect",
        default="1:1",
        choices=VALID_ASPECTS,
        help="Aspect ratio (default: 1:1)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (overrides --name)"
    )
    parser.add_argument(
        "--reference", "-r",
        type=Path,
        nargs="+",
        help="Reference image(s) for editing or style guidance (up to 14 images)"
    )

    args = parser.parse_args()

    # Validate reference images
    reference_images = None
    if args.reference:
        if len(args.reference) > 14:
            print("Error: Maximum 14 reference images allowed", file=sys.stderr)
            sys.exit(1)
        reference_images = args.reference

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = DEFAULT_OUTPUT_DIR / generate_filename(args.name)

    # Generate and save
    print(f"Generating image with prompt: {args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}")
    print(f"Aspect ratio: {args.aspect}")
    if reference_images:
        print(f"Reference images: {len(reference_images)}")

    image_bytes = generate_image(args.prompt, args.aspect, reference_images)
    saved_path = save_image(image_bytes, output_path)

    print(f"\nImage saved to: {saved_path}")

    # Output wikilink format for Obsidian
    if "Archive/Files" in str(saved_path):
        relative_path = f"Archive/Files/{saved_path.name}"
        print(f"\nEmbed with: ![[{relative_path}]]")
        print(f"Or with width: ![[{relative_path}|500]]")

    return str(saved_path)


if __name__ == "__main__":
    main()
