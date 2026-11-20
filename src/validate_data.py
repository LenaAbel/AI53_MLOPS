from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Required Core checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow (`Image.load()` or equivalent);
      5. no split or class is empty.

    The report must not contain timestamps because it is a DVC output.
    """
    sorted_classes = sorted(EXPECTED_CLASSES)
    splits_report = {}
    total_images = 0

    for folder, expected_count in EXPECTED_SPLITS.items():
        split_path = raw_root / folder

        # 1. train, validation and test folders exist
        if not split_path.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_path}")

        # 2. every split contains exactly the three expected classes
        classes = set()
        for item in split_path.iterdir():
            if item.is_dir():
                classes.add(item.name)

        if classes != EXPECTED_CLASSES:
            raise RuntimeError(
                f"Unexpected classes in {folder}: expected {sorted_classes}, found {sorted(classes)}"
            )

        # 3. image counts, 4. Pillow decode & 5. no class is empty
        split_count = 0
        class_counts = {}

        for class_name in sorted_classes:
            class_path = split_path / class_name
            class_count = 0

            for file_path in class_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    # 4. Pillow can read all the pixels in each image
                    try:
                        with Image.open(file_path) as img:
                            img.load()
                    except Exception as exc:
                        raise OSError(f"Corrupted image file: {file_path}") from exc

                    class_count += 1

            # 5. no class is empty
            if class_count == 0:
                raise RuntimeError(f"Empty class directory: {class_path}")

            class_counts[class_name] = class_count
            split_count += class_count

        # 3. image counts match the exact course archives
        if split_count != expected_count:
            raise RuntimeError(
                f"Image count mismatch in {folder}: expected {expected_count}, found {split_count}"
            )

        splits_report[folder] = {
            "images": split_count,
            "class_counts": class_counts,
        }
        total_images += split_count

    # 6. deterministic report (no timestamps, sorted classes)
    return {
        "status": "PASS",
        "classes": sorted_classes,
        "splits": splits_report,
        "total_images": total_images,
    }


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
