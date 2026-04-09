import argparse
import json
from pathlib import Path


DEFAULT_PROMPT = "A ball navigates through a maze from the top-left corner to the bottom-right corner."


def build_train_cos(
    metadata_path: str,
    output_path: str,
    search_video_field: str,
    prompt: str = DEFAULT_PROMPT,
) -> None:
    records = []
    with open(metadata_path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            metadata = json.loads(line)
            walk_video = metadata.get("walk_video")
            search_video = metadata.get(search_video_field)
            if not isinstance(walk_video, str) or not walk_video:
                raise ValueError(f"{metadata_path}:{line_no} is missing walk_video")
            if not isinstance(search_video, str) or not search_video:
                raise ValueError(
                    f"{metadata_path}:{line_no} is missing {search_video_field}; "
                    "generate the search videos first or choose the correct field."
                )

            records.append({
                "video": walk_video,
                "search_video": search_video,
                "prompt": prompt,
            })

    output = Path(output_path)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build train_cos.json from metadata.jsonl")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--search-video-field",
        type=str,
        default="bfs_video",
        help="Metadata field to use as search_video, e.g. bfs_video or unidirectional_bfs_video",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt string")
    args = parser.parse_args()

    build_train_cos(
        metadata_path=args.metadata,
        output_path=args.output,
        search_video_field=args.search_video_field,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
