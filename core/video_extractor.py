"""Video frame extraction module.

Supports:
- Local video files (mp4, avi, mkv, mov, etc.)
- Online video URLs (via yt-dlp)
- Auto-extraction (every N frames / every M seconds)
- Scene-change detection (only extract when frame differs enough)
- Perceptual hash dedup (skip near-duplicate frames)
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from loguru import logger


class VideoFrameExtractor:
    """Extract frames from video files or online video URLs."""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.total_frames: int = 0
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.duration_sec: float = 0.0
        self._prev_gray: Optional[np.ndarray] = None
        self._extracted_hashes: list[bytes] = []

    # ------------------------------------------------------------------
    # Open / Close
    # ------------------------------------------------------------------

    def open(self, video_path: str) -> bool:
        """Open a video file for frame extraction.

        Args:
            video_path: Path to local video file.

        Returns:
            True if opened successfully.
        """
        self.close()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return False

        self.video_path = video_path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0
        self._prev_gray = None
        self._extracted_hashes = []

        logger.info(
            f"Video opened: {video_path} "
            f"({self.width}x{self.height}, {self.fps:.1f}fps, "
            f"{self.total_frames} frames, {self.duration_sec:.1f}s)"
        )
        return True

    def close(self):
        """Close the current video."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_path = None
        self._prev_gray = None

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    # ------------------------------------------------------------------
    # Frame navigation
    # ------------------------------------------------------------------

    def seek_frame(self, frame_idx: int) -> bool:
        """Seek to a specific frame index."""
        if not self.is_opened():
            return False
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def read_frame(self) -> Optional[np.ndarray]:
        """Read current frame as BGR numpy array."""
        if not self.is_opened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def read_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """Seek and read a specific frame."""
        if not self.seek_frame(frame_idx):
            return None
        return self.read_frame()

    # ------------------------------------------------------------------
    # Auto extraction
    # ------------------------------------------------------------------

    def extract_auto(
        self,
        output_dir: str,
        mode: str = "interval",
        interval_frames: int = 30,
        interval_seconds: float = 1.0,
        scene_threshold: float = 30.0,
        dedup: bool = True,
        dedup_threshold: int = 8,
        max_frames: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[str]:
        """Auto-extract frames from the video.

        Args:
            output_dir: Directory to save extracted frames.
            mode: Extraction mode - "interval" (every N frames/seconds)
                  or "scene" (scene change detection).
            interval_frames: Extract every N frames (for mode="interval", step_type="frames").
            interval_seconds: Extract every M seconds (for mode="interval", step_type="seconds").
            scene_threshold: MSE threshold for scene change (for mode="scene").
                             Lower = more sensitive. Typical range: 10-50.
            dedup: Whether to skip near-duplicate frames.
            dedup_threshold: Hamming distance threshold for perceptual hash dedup.
                             Lower = stricter. Typical range: 5-12.
            max_frames: Maximum number of frames to extract. 0 = no limit.
            progress_callback: Called with (current_frame, total_frames).

        Returns:
            List of saved image file paths.
        """
        if not self.is_opened():
            logger.error("No video opened")
            return []

        os.makedirs(output_dir, exist_ok=True)
        saved_paths: list[str] = []
        self._prev_gray = None
        self._extracted_hashes = []

        # Calculate step size
        if mode == "interval":
            step = max(1, interval_frames)
        else:
            step = 1  # scene mode: check every frame

        frame_idx = 0
        saved_count = 0

        self.seek_frame(0)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            should_save = False

            if mode == "interval":
                if frame_idx % step == 0:
                    should_save = True
            elif mode == "scene":
                if self._is_scene_change(frame, scene_threshold):
                    should_save = True

            if should_save and dedup:
                if self._is_duplicate(frame, dedup_threshold):
                    should_save = False

            if should_save:
                path = self._save_frame(frame, output_dir, frame_idx)
                if path:
                    saved_paths.append(path)
                    saved_count += 1
                    if max_frames > 0 and saved_count >= max_frames:
                        break

            if progress_callback and frame_idx % 100 == 0:
                progress_callback(frame_idx, self.total_frames)

            frame_idx += 1

        logger.info(f"Extracted {len(saved_paths)} frames to {output_dir}")
        return saved_paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_scene_change(self, frame: np.ndarray, threshold: float) -> bool:
        """Detect scene change using MSE between consecutive frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))  # downscale for speed

        if self._prev_gray is None:
            self._prev_gray = gray
            return True  # First frame is always a "scene change"

        mse = np.mean((gray.astype(float) - self._prev_gray.astype(float)) ** 2)
        self._prev_gray = gray
        return mse > threshold

    def _is_duplicate(self, frame: np.ndarray, threshold: int) -> bool:
        """Check if a frame is a near-duplicate using color histogram comparison.

        Uses correlation of HSV histograms instead of average hash,
        which handles both content-rich and solid-color frames well.

        Optimized: only compares against recent hashes (sliding window)
        to avoid O(n²) cost on long videos.
        """
        h = self._color_histogram_hash(frame)
        corr_threshold = 1.0 - threshold / 32.0

        # Only compare against the most recent hashes (sliding window)
        # to avoid O(n²) cost on long videos.  64 recent hashes is enough
        # to catch consecutive duplicates while staying fast.
        recent = self._extracted_hashes[-64:]
        for existing in recent:
            if self._hist_correlation(h, existing) > corr_threshold:
                return True
        self._extracted_hashes.append(h)
        return False

    @staticmethod
    def _color_histogram_hash(frame: np.ndarray) -> bytes:
        """Compute a compact color histogram fingerprint of a frame.

        Uses HSV histograms (H: 18 bins, S: 8 bins, V: 8 bins) plus
        the mean BGR values to distinguish both color-rich and solid-color frames.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Compute histogram on H, S, V channels
        hist_h = cv2.calcHist([hsv], [0], None, [18], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        # Normalize
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        cv2.normalize(hist_v, hist_v)
        # Also include mean BGR to distinguish solid-color frames
        mean_bgr = frame.mean(axis=(0, 1))  # [B, G, R]
        # Combine into bytes
        combined = np.concatenate([
            hist_h.flatten(), hist_s.flatten(), hist_v.flatten(),
            mean_bgr / 255.0,  # normalize to [0, 1]
        ])
        return combined.tobytes()

    @staticmethod
    def _hist_correlation(h1: bytes, h2: bytes) -> float:
        """Compute correlation between two histogram fingerprints."""
        a = np.frombuffer(h1, dtype=np.float64)
        b = np.frombuffer(h2, dtype=np.float64)
        if len(a) != len(b):
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0 if norm_a == norm_b else 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def save_current_frame(self, output_dir: str, frame_idx: Optional[int] = None) -> Optional[str]:
        """Save the current (or specified) frame to disk.

        Preserves the current playback position after saving.

        Args:
            output_dir: Directory to save the frame.
            frame_idx: Specific frame index (0-based). If None, saves the
                current playback position.

        Returns:
            Path to saved image, or None on failure.
        """
        if not self.is_opened():
            return None

        saved_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        target_idx = frame_idx if frame_idx is not None else saved_pos

        self.seek_frame(target_idx)
        ret, frame = self.cap.read()
        self.seek_frame(saved_pos)

        if not ret or frame is None:
            return None

        os.makedirs(output_dir, exist_ok=True)
        return self._save_frame(frame, output_dir, target_idx)

    def _save_frame(self, frame: np.ndarray, output_dir: str, frame_idx: int) -> Optional[str]:
        """Save a frame as a JPEG file."""
        video_name = Path(self.video_path).stem if self.video_path else "frame"
        # Remove yt-dlp stream suffix like ".f100026" from the video name
        # e.g. "video.f100026" -> "video", "video.f30016.f100026" -> "video"
        video_name = re.sub(r'\.f\d+', '', video_name)
        # Also remove common unwanted characters
        video_name = re.sub(r'[^\w\-.]', '_', video_name)
        filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path

    # ------------------------------------------------------------------
    # Online video support
    # ------------------------------------------------------------------

    # Supported platforms (yt-dlp extractors)
    SUPPORTED_PLATFORMS = {
        "抖音": ["douyin.com", "iesdouyin.com", "v.douyin.com"],
        "B站": ["bilibili.com", "b23.tv", "biligc.com"],
        "快手": ["kuaishou.com", "v.kuaishou.com", "gifshow.com"],
        "西瓜视频": ["ixigua.com"],
        "微博": ["weibo.com", "weibo.cn", "m.weibo.cn"],
        "小红书": ["xiaohongshu.com", "xhslink.com"],
        "知乎": ["zhihu.com", "zhuanlan.zhihu.com"],
        "AcFun": ["acfun.cn"],
        "TikTok": ["tiktok.com", "vm.tiktok.com"],
        "YouTube": ["youtube.com", "youtu.be"],
        "花椒直播": ["huajiao.com"],
    }

    # Platforms that may need cookies for access
    COOKIE_PLATFORMS = {"抖音", "B站", "小红书", "微博"}

    @staticmethod
    def is_ytdlp_available() -> bool:
        """Check if yt-dlp is installed."""
        try:
            result = subprocess.run(
                ["yt-dlp", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def is_youget_available() -> bool:
        """Check if you-get is installed (fallback for platforms like Kuaishou)."""
        try:
            result = subprocess.run(
                ["you-get", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @classmethod
    def detect_platform(cls, url: str) -> str:
        """Detect which platform a URL belongs to.

        Automatically extracts URL from sharing text before detection.

        Returns:
            Platform name, or "其他" if unknown.
        """
        # Extract clean URL first (handles sharing text)
        clean_url = cls.extract_url(url)
        url_lower = clean_url.lower()
        for platform, domains in cls.SUPPORTED_PLATFORMS.items():
            for domain in domains:
                if domain in url_lower:
                    return platform
        return "其他"

    @classmethod
    def get_supported_platforms(cls) -> list[str]:
        """Return list of supported platform names."""
        return list(cls.SUPPORTED_PLATFORMS.keys())

    @staticmethod
    def extract_url(text: str) -> str:
        """Extract a valid video URL from sharing text.

        Many Chinese platforms share videos with surrounding text like:
        "8.28 W@M.Wm ... https://v.douyin.com/xxx/ 复制此链接..."
        This method strips the extra text and returns just the URL.

        Args:
            text: Raw sharing text or a plain URL.

        Returns:
            Extracted URL string, or the original text if no URL found.
        """
        import re
        # Match http:// or https:// URLs
        urls = re.findall(r'https?://[^\s<>\"]+', text)
        if urls:
            # Return the first URL found, strip trailing punctuation
            url = urls[0].rstrip("/:：，。！？、")
            # Ensure trailing slash for some short URLs (douyin, b23.tv)
            if re.match(r'https?://[a-z0-9.-]+\.(com|tv|cn|io)/[a-zA-Z0-9_-]+$', url):
                url += "/"
            return url
        return text.strip()

    @staticmethod
    def download_video(
        url: str,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        proxy: str = "",
        cookie_file: str = "",
        cookie_browser: str = "",
    ) -> Optional[str]:
        """Download a video from URL.

        Tries yt-dlp first, then you-get as fallback.
        Supports proxy and cookie authentication for restricted platforms.
        Automatically extracts URLs from sharing text.
        For platforms requiring cookies (Douyin, Bilibili, etc.), automatically
        tries to extract cookies from the user's browser.

        Args:
            url: Video URL or sharing text (Douyin, Bilibili, YouTube, Kuaishou, etc.)
            output_dir: Directory to save the video. Uses temp dir if None.
            progress_callback: Called with status text.
            proxy: HTTP/SOCKS proxy URL (e.g. "http://127.0.0.1:7890").
            cookie_file: Path to cookies.txt file for authenticated access.
            cookie_browser: Browser name for auto cookie extraction
                           (e.g. "chrome", "edge", "firefox", "safari").

        Returns:
            Path to downloaded video file, or None on failure.
        """
        # Extract URL from sharing text if needed
        url = VideoFrameExtractor.extract_url(url)
        logger.info(f"Downloading: {url}")

        # Try yt-dlp first
        if VideoFrameExtractor.is_ytdlp_available():
            # First attempt: use provided cookie_file or cookie_browser
            result = VideoFrameExtractor._download_with_ytdlp(
                url, output_dir, progress_callback, proxy, cookie_file, cookie_browser
            )
            if result:
                return result

            # If cookie-related error and no cookies were provided, auto-retry
            # with browser cookies for platforms that need them
            platform = VideoFrameExtractor.detect_platform(url)
            if platform in VideoFrameExtractor.COOKIE_PLATFORMS and not cookie_file and not cookie_browser:
                if progress_callback:
                    progress_callback("需要Cookie，尝试从浏览器自动获取...")
                # Try common browsers in order
                for browser in ["chrome", "edge", "firefox"]:
                    if progress_callback:
                        progress_callback(f"尝试从 {browser} 获取Cookie...")
                    result = VideoFrameExtractor._download_with_ytdlp(
                        url, output_dir, progress_callback, proxy, "", browser
                    )
                    if result:
                        return result

        # Fallback to you-get for unsupported platforms (e.g. Kuaishou)
        if VideoFrameExtractor.is_youget_available():
            result = VideoFrameExtractor._download_with_youget(
                url, output_dir, progress_callback, proxy
            )
            if result:
                return result

        # Both failed
        if not VideoFrameExtractor.is_ytdlp_available() and not VideoFrameExtractor.is_youget_available():
            logger.error("No video download tool available. Install: pip install yt-dlp you-get")
        return None

    @staticmethod
    def _download_with_ytdlp(
        url: str,
        output_dir: str,
        progress_callback: Optional[Callable[[str], None]],
        proxy: str = "",
        cookie_file: str = "",
        cookie_browser: str = "",
    ) -> Optional[str]:
        """Download video using yt-dlp."""
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="yolo_studio_")

        os.makedirs(output_dir, exist_ok=True)
        output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

        try:
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", output_template,
                "--no-playlist",
                "--no-check-certificates",
            ]

            # Proxy support
            if proxy:
                cmd += ["--proxy", proxy]

            # Cookie support - prefer browser cookies, then file cookies
            if cookie_browser:
                cmd += ["--cookies-from-browser", cookie_browser]
            elif cookie_file and os.path.isfile(cookie_file):
                cmd += ["--cookies", cookie_file]

            cmd.append(url)

            if progress_callback:
                progress_callback("正在下载视频 (yt-dlp)...")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                cwd=output_dir,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                logger.error(f"yt-dlp failed: {stderr}")
                # If it's a sign-in issue, provide helpful message
                if "Sign in" in stderr or "login" in stderr.lower() or "cookie" in stderr.lower() or "Fresh cookies" in stderr:
                    if progress_callback:
                        progress_callback("需要Cookie: 请在浏览器中登录抖音后重试，或手动配置Cookie文件")
                return None

            # Find the downloaded file
            # Prefer merged mp4 (no .fXXXXX suffix), then largest video file
            video_exts = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv"}
            candidates = [
                f for f in Path(output_dir).iterdir()
                if f.suffix.lower() in video_exts
            ]

            if not candidates:
                logger.error("Downloaded video file not found")
                return None

            # Prefer files without yt-dlp stream suffix (.f12345.mp4)
            merged = [f for f in candidates if not re.match(r'.+\.f\d+\.', f.name)]
            if merged:
                # Pick the largest merged file
                best = max(merged, key=lambda f: f.stat().st_size)
            else:
                # All files have stream suffix, pick the largest
                best = max(candidates, key=lambda f: f.stat().st_size)

            # Clean up other partial files
            for f in candidates:
                if f != best:
                    try:
                        f.unlink()
                        logger.debug(f"Removed partial file: {f.name}")
                    except OSError:
                        pass

            logger.info(f"Downloaded video: {best}")
            return str(best)

        except subprocess.TimeoutExpired:
            logger.error("Video download timed out (10min limit)")
            return None
        except Exception as e:
            logger.error(f"yt-dlp download failed: {e}")
            return None

    @staticmethod
    def _download_with_youget(
        url: str,
        output_dir: str,
        progress_callback: Optional[Callable[[str], None]],
        proxy: str = "",
    ) -> Optional[str]:
        """Download video using you-get (fallback for Kuaishou, etc.)."""
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="yolo_studio_")

        os.makedirs(output_dir, exist_ok=True)

        try:
            cmd = ["you-get", "-o", output_dir]

            # Proxy support
            if proxy:
                cmd += ["--http-proxy", proxy]

            cmd.append(url)

            if progress_callback:
                progress_callback("正在下载视频 (you-get)...")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )

            if result.returncode != 0:
                logger.error(f"you-get failed: {result.stderr}")
                return None

            # Find the downloaded file - pick the largest video file
            video_exts = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv"}
            candidates = [
                f for f in Path(output_dir).iterdir()
                if f.suffix.lower() in video_exts
            ]
            if not candidates:
                logger.error("Downloaded video file not found (you-get)")
                return None
            best = max(candidates, key=lambda f: f.stat().st_size)
            logger.info(f"Downloaded video (you-get): {best}")
            return str(best)

        except subprocess.TimeoutExpired:
            logger.error("you-get download timed out")
            return None
        except Exception as e:
            logger.error(f"you-get download failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        """Get video info as a dict."""
        return {
            "path": self.video_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_sec": self.duration_sec,
            "duration_str": self._format_duration(self.duration_sec),
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
