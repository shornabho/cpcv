"""
Face enrollment tool.

    python enroll.py build   — build face_db.pkl from known_faces/<name>/*.jpg
    python enroll.py review  — label unknown_faces/ images interactively
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from recognizer import _embedding


def build(known_dir: Path, db_path: Path) -> None:
    db: dict[str, list] = {}
    total_embeddings = 0

    for person_dir in sorted(known_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        embeddings = []
        photos = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png"))
        for img_path in photos:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [{name}] skip (unreadable): {img_path.name}")
                continue
            emb = _embedding(img)
            if emb is None:
                print(f"  [{name}] skip (no face detected): {img_path.name}")
                continue
            embeddings.append(emb)
            total_embeddings += 1

        if embeddings:
            db[name] = embeddings
            print(f"  {name}: {len(embeddings)} embedding(s) from {len(photos)} photo(s)")
        else:
            print(f"  {name}: no usable faces — skipped")

    with open(db_path, "wb") as f:
        pickle.dump(db, f)

    print(f"\nSaved {db_path} — {len(db)} person(s), {total_embeddings} total embedding(s)")


def _render_panel(base: np.ndarray, name_buf: str, status: str) -> np.ndarray:
    """Rebuild the display panel with the current typing state."""
    bar = np.full((52, base.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, f"Name: {name_buf}_",
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, "Enter=save  Esc=skip  D=delete  Bksp=erase",
                (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
    if status:
        cv2.putText(bar, status, (base.shape[1] - 160, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)
    return np.vstack([base, bar])


def review(unknown_dir: Path, known_dir: Path, db_path: Path) -> None:
    images = sorted(unknown_dir.glob("*.jpg")) + sorted(unknown_dir.glob("*.png"))
    if not images:
        print(f"No images found in {unknown_dir}")
        return

    moved = 0
    total = len(images)
    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            img_path.unlink()
            continue

        # Scale up to a minimum 400px on the longest side for legibility.
        h, w = img.shape[:2]
        scale = max(1, 400 // max(h, w, 1))
        base = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        title = f"[{idx}/{total}] {img_path.name}"

        # All interaction happens inside the cv2 window — no terminal input(),
        # which avoids cv2.waitKey() corrupting the terminal's line discipline.
        name_buf = ""
        action = "skip"

        while True:
            cv2.imshow(title, _render_panel(base, name_buf, ""))
            key = cv2.waitKey(50) & 0xFF

            if key == 255:  # no key
                continue
            elif key == 27:  # Esc → skip
                action = "skip"
                break
            elif key == ord('d') and not name_buf:  # D with empty buffer → delete
                action = "delete"
                break
            elif key in (8, 127):  # Backspace
                name_buf = name_buf[:-1]
            elif key == 13:  # Enter → save if we have a name
                if name_buf.strip():
                    action = "save"
                    break
            elif 32 <= key <= 126:  # printable ASCII
                name_buf += chr(key)

        cv2.destroyAllWindows()

        if action == "delete":
            img_path.unlink()
            print(f"  [{idx}/{total}] deleted")
        elif action == "save":
            name = name_buf.strip()
            dest_dir = known_dir / name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img_path), str(dest_dir / img_path.name))
            print(f"  [{idx}/{total}] → known_faces/{name}/")
            moved += 1
        else:
            print(f"  [{idx}/{total}] skipped")

    if moved:
        print(f"\nMoved {moved} image(s). Rebuilding face database...")
        build(known_dir, db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Face enrollment tool")
    parser.add_argument("command", choices=["build", "review"])
    args = parser.parse_args()

    known_dir = Path(config.KNOWN_FACES_DIR)
    unknown_dir = Path(config.UNKNOWN_FACES_DIR)
    db_path = Path(config.FACE_DB_PATH)

    if args.command == "build":
        if not known_dir.exists():
            print(f"'{known_dir}' not found. Create it and add subfolders per person.")
            sys.exit(1)
        print(f"Building face DB from {known_dir}/...")
        build(known_dir, db_path)

    elif args.command == "review":
        if not unknown_dir.exists() or not any(unknown_dir.iterdir()):
            print(f"No unknown faces at '{unknown_dir}'.")
            sys.exit(0)
        review(unknown_dir, known_dir, db_path)


if __name__ == "__main__":
    main()
