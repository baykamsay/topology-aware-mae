"""Dataset utilities for loading image folders entirely into RAM."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

logger = logging.getLogger(__name__)


class InMemoryImageFolder(ImageFolder):
    """ImageFolder variant that preloads images into memory.

    This is intended for relatively small datasets where disk I/O becomes the
    training bottleneck. Images are loaded once at initialisation and kept in
    memory so that subsequent epochs only perform transformation work.

    Args:
        root: Root directory path.
        transform: A function/transform that takes in an image and returns a
            transformed version.  See :class:`torchvision.transforms`.
        target_transform: A function/transform that takes in the target and
            transforms it.
        loader: A callable that takes a path to an image and returns the loaded
            image.
        is_valid_file: A callable that takes a path and returns True if the
            file is ok to use. This is passed to the underlying ImageFolder.
        preload_transform: Optional transform applied once when loading each
            image into memory (e.g. converting to RGB). Defaults to ``None``.
        copy_into_memory: Whether to make a defensive copy of the loaded image
            before storing it. Defaults to ``True``.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        preload_transform: Optional[Callable[[Any], Any]] = None,
        copy_into_memory: bool = True,
        preload_workers: Optional[int] = None,
    ) -> None:
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.preload_transform = preload_transform
        self.copy_into_memory = copy_into_memory
        self._image_data = [None] * len(self.samples)  # type: ignore[var-annotated]

        worker_count = self._resolve_worker_count(preload_workers)
        logger.info(
            "Preloading %d images from %s using %d worker(s)",
            len(self.samples),
            root,
            worker_count,
        )

        if worker_count == 1:
            for idx, (path, _) in enumerate(self.samples):
                self._image_data[idx] = self._load_single(path)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(self._load_single, path): idx
                    for idx, (path, _) in enumerate(self.samples)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        self._image_data[idx] = future.result()
                    except Exception as exc:  # pragma: no cover - propagate to caller
                        logger.error(
                            "Failed to preload image %s: %s",
                            self.samples[idx][0],
                            exc,
                        )
                        raise

        logger.info(
            "Loaded %d images from %s into memory (classes: %d)",
            len(self._image_data),
            root,
            len(self.classes),
        )

    def _resolve_worker_count(self, preload_workers: Optional[int]) -> int:
        if preload_workers is not None:
            return max(1, preload_workers)
        cpu_count = os.cpu_count() or 1
        return max(1, min(32, cpu_count))

    def _load_single(self, path: str) -> Any:
        image = self.loader(path)

        if hasattr(image, "load"):
            image.load()  # type: ignore[call-arg]

        if self.preload_transform is not None:
            image = self.preload_transform(image)
        elif isinstance(image, Image.Image) and self.copy_into_memory:
            image = image.copy()

        return image

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self._image_data[index]
        target = self.targets[index]

        # Defensive copy so that downstream transforms cannot mutate the cache.
        if isinstance(image, Image.Image):
            image = image.copy()

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def extra_repr(self) -> str:
        base_repr = super().extra_repr()
        return base_repr + f"\n    in_memory={len(self._image_data)} images"
