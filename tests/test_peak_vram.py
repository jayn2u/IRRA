import unittest
from unittest.mock import patch

from processor.processor import _peak_vram_bytes, _reset_peak_vram_stats


class FakePeakTensor:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class PeakVramMeasurementTest(unittest.TestCase):
    @patch("processor.processor.torch.cuda.reset_peak_memory_stats")
    @patch("processor.processor.torch.cuda.is_available", return_value=True)
    def test_resets_cuda_peak_stats(self, _is_available, reset):
        self.assertTrue(_reset_peak_vram_stats())
        reset.assert_called_once_with()

    @patch("processor.processor.torch.cuda.reset_peak_memory_stats")
    @patch("processor.processor.torch.cuda.is_available", return_value=False)
    def test_skips_reset_without_cuda(self, _is_available, reset):
        self.assertFalse(_reset_peak_vram_stats())
        reset.assert_not_called()

    @patch("processor.processor.torch.tensor")
    @patch("processor.processor.torch.cuda.max_memory_reserved", return_value=20)
    @patch("processor.processor.torch.cuda.max_memory_allocated", return_value=12)
    @patch("processor.processor.torch.cuda.is_available", return_value=True)
    def test_reads_local_peak_bytes(self, _available, _allocated, _reserved,
                                    tensor):
        tensor.return_value = FakePeakTensor([12, 20])

        self.assertEqual(_peak_vram_bytes(), (12, 20))

    @patch("processor.processor.torch.distributed.all_reduce")
    @patch("processor.processor.torch.tensor")
    @patch("processor.processor.torch.cuda.max_memory_reserved", return_value=20)
    @patch("processor.processor.torch.cuda.max_memory_allocated", return_value=12)
    @patch("processor.processor.torch.cuda.is_available", return_value=True)
    def test_reduces_distributed_peaks_with_max(self, _available, _allocated,
                                                _reserved, tensor, all_reduce):
        peaks = FakePeakTensor([12, 20])
        tensor.return_value = peaks

        def replace_with_global_max(value, op):
            self.assertIs(op, __import__("torch").distributed.ReduceOp.MAX)
            value.values = [18, 24]

        all_reduce.side_effect = replace_with_global_max

        self.assertEqual(_peak_vram_bytes(distributed=True), (18, 24))

    @patch("processor.processor.torch.cuda.is_available", return_value=False)
    def test_omits_peaks_without_cuda(self, _available):
        self.assertIsNone(_peak_vram_bytes())


if __name__ == "__main__":
    unittest.main()
