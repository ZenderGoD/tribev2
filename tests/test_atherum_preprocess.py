import unittest
from unittest import mock

from tribev2.atherum_preprocess import (
    build_image_to_video_command,
    build_prepare_video_command,
    image_to_video,
    prepare_video,
)


class AtherumPreprocessTests(unittest.TestCase):
    def test_builds_image_to_video_command_for_static_creative(self):
        command = build_image_to_video_command(
            input_path="/tmp/input.png",
            output_path="/tmp/output.mp4",
            duration_sec=4,
            width=640,
            height=360,
        )

        self.assertEqual(command[:5], ["ffmpeg", "-y", "-loop", "1", "-i"])
        vf = command[command.index("-vf") + 1]
        self.assertIn("scale=640:360:force_original_aspect_ratio=decrease", vf)
        self.assertIn("pad=640:360:(ow-iw)/2:(oh-ih)/2", vf)
        self.assertEqual(command[-1], "/tmp/output.mp4")

    def test_builds_video_prepare_command_with_trim_audio_strip_and_fps(self):
        command = build_prepare_video_command(
            input_path="/tmp/input.mov",
            output_path="/tmp/output.mp4",
            max_duration_sec=60,
            fps=4,
            strip_audio=True,
        )

        self.assertEqual(command[:4], ["ffmpeg", "-y", "-i", "/tmp/input.mov"])
        self.assertIn("-t", command)
        self.assertIn("60", command)
        self.assertIn("-an", command)
        self.assertIn("fps=4", command)
        self.assertEqual(command[-1], "/tmp/output.mp4")

    def test_wrappers_run_subprocess_with_check_enabled(self):
        with mock.patch("subprocess.run") as run:
            image_to_video("/tmp/image.png", "/tmp/image.mp4")
            prepare_video("/tmp/raw.mp4", "/tmp/tribe.mp4")

        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertTrue(call.kwargs["check"])
            self.assertTrue(call.kwargs["capture_output"])


if __name__ == "__main__":
    unittest.main()
