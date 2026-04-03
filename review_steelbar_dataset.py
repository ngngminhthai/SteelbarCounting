import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk


DATASET_ROOT = Path(__file__).resolve().parent / "steelbar_dataset"
IMAGE_DIR = DATASET_ROOT / "images"
LABEL_DIR = DATASET_ROOT / "labels"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
POINT_RADIUS = 5
MAX_CANVAS_SIZE = (1200, 800)


class SteelbarReviewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Steelbar Dataset Reviewer")
        self.root.geometry("1280x920")

        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(
                f"No image/label pairs found under {IMAGE_DIR} and {LABEL_DIR}"
            )

        self.index = 0
        self.photo_image = None
        self.point_cache: dict[Path, list[tuple[float, float]]] = {}

        self.status_var = tk.StringVar()
        self.point_size_var = tk.IntVar(value=POINT_RADIUS)
        self.show_points_var = tk.BooleanVar(value=True)
        self.only_labeled_var = tk.BooleanVar(value=False)
        self.jump_var = tk.StringVar()

        self._build_ui()
        self._bind_keys()
        self.refresh_view()

    def _load_samples(self) -> list[dict]:
        image_paths = sorted(
            path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        samples = []
        for image_path in image_paths:
            label_path = LABEL_DIR / f"{image_path.stem}.json"
            if label_path.exists():
                samples.append({"image": image_path, "label": label_path})
        return samples

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")

        ttk.Button(controls, text="Prev", command=self.prev_sample).pack(side="left")
        ttk.Button(controls, text="Next", command=self.next_sample).pack(side="left", padx=(8, 16))

        ttk.Checkbutton(
            controls,
            text="Show points",
            variable=self.show_points_var,
            command=self.refresh_view,
        ).pack(side="left")

        ttk.Checkbutton(
            controls,
            text="Only labeled",
            variable=self.only_labeled_var,
            command=self._apply_filter,
        ).pack(side="left", padx=(12, 0))

        ttk.Label(controls, text="Point size").pack(side="left", padx=(16, 4))
        ttk.Spinbox(
            controls,
            from_=1,
            to=20,
            width=4,
            textvariable=self.point_size_var,
            command=self.refresh_view,
        ).pack(side="left")

        ttk.Label(controls, text="Jump").pack(side="left", padx=(16, 4))
        self.jump_entry = ttk.Entry(controls, textvariable=self.jump_var, width=8)
        self.jump_entry.pack(side="left")
        self.jump_entry.bind("<Return>", lambda _event: self.jump_to_index())
        ttk.Button(controls, text="Go", command=self.jump_to_index).pack(side="left", padx=(4, 0))

        self.image_panel = ttk.Label(self.root, anchor="center")
        self.image_panel.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=8,
        )
        status_bar.pack(fill="x", side="bottom")

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda _event: self.prev_sample())
        self.root.bind("<Right>", lambda _event: self.next_sample())
        self.root.bind("<Up>", lambda _event: self.change_point_size(1))
        self.root.bind("<Down>", lambda _event: self.change_point_size(-1))
        self.root.bind("l", lambda _event: self.toggle_labeled_filter())
        self.root.bind("p", lambda _event: self.toggle_points())
        self.root.bind("g", lambda _event: self.focus_jump())

    def _read_points(self, label_path: Path) -> list[tuple[float, float]]:
        if label_path in self.point_cache:
            return self.point_cache[label_path]

        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            messagebox.showerror("Invalid label file", f"{label_path}\n\n{exc}")
            return []

        points = payload.get("points", [])
        valid_points = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                valid_points.append((float(point[0]), float(point[1])))
        self.point_cache[label_path] = valid_points
        return valid_points

    def _draw_points(self, image: Image.Image, points: list[tuple[float, float]]) -> Image.Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        radius = max(1, int(self.point_size_var.get()))
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="red", width=2)
            draw.line((x - radius - 2, y, x + radius + 2, y), fill="yellow", width=1)
            draw.line((x, y - radius - 2, x, y + radius + 2), fill="yellow", width=1)
        return annotated

    def _resize_for_display(self, image: Image.Image) -> Image.Image:
        display = image.copy()
        display.thumbnail(MAX_CANVAS_SIZE, Image.Resampling.LANCZOS)
        return display

    def _current_samples(self) -> list[dict]:
        if not self.only_labeled_var.get():
            return self.samples

        filtered = []
        for sample in self.samples:
            if self._read_points(sample["label"]):
                filtered.append(sample)
        return filtered

    def _apply_filter(self) -> None:
        current = self._current_samples()
        if not current:
            self.only_labeled_var.set(False)
            messagebox.showinfo("No labeled samples", "No samples match the current filter.")
            return

        self.index = min(self.index, len(current) - 1)
        self.refresh_view()

    def refresh_view(self) -> None:
        current = self._current_samples()
        if not current:
            self.image_panel.configure(image="", text="No samples available")
            self.status_var.set("No samples available")
            return

        self.index %= len(current)
        sample = current[self.index]
        points = self._read_points(sample["label"])

        image = Image.open(sample["image"]).convert("RGB")
        if self.show_points_var.get():
            image = self._draw_points(image, points)
        display = self._resize_for_display(image)

        self.photo_image = ImageTk.PhotoImage(display)
        self.image_panel.configure(image=self.photo_image)

        self.status_var.set(
            f"{self.index + 1}/{len(current)} | {sample['image'].name} | "
            f"points: {len(points)} | image: {sample['image']} | label: {sample['label']}"
        )

    def next_sample(self) -> None:
        self.index += 1
        self.refresh_view()

    def prev_sample(self) -> None:
        self.index -= 1
        self.refresh_view()

    def jump_to_index(self) -> None:
        raw = self.jump_var.get().strip()
        if not raw:
            return

        current = self._current_samples()
        try:
            value = int(raw)
        except ValueError:
            messagebox.showerror("Invalid index", f"Expected an integer, got: {raw}")
            return

        if not 1 <= value <= len(current):
            messagebox.showerror("Out of range", f"Enter a value between 1 and {len(current)}")
            return

        self.index = value - 1
        self.refresh_view()

    def change_point_size(self, delta: int) -> None:
        new_value = max(1, min(20, self.point_size_var.get() + delta))
        self.point_size_var.set(new_value)
        self.refresh_view()

    def toggle_labeled_filter(self) -> None:
        self.only_labeled_var.set(not self.only_labeled_var.get())
        self._apply_filter()

    def toggle_points(self) -> None:
        self.show_points_var.set(not self.show_points_var.get())
        self.refresh_view()

    def focus_jump(self) -> None:
        self.jump_entry.focus_set()
        self.jump_entry.selection_range(0, "end")


def main() -> None:
    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Expected dataset folders at {IMAGE_DIR} and {LABEL_DIR}"
        )

    root = tk.Tk()
    app = SteelbarReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
