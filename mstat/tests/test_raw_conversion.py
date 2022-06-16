import pytest
import os
from dependencies.file_conversion.RAWConversion import rawToNumpy

class TestRAWConversion:
    bad_path = 'fake/path/'
    real_file = 'test_file.raw'
    good_path = f'tests/{real_file}'
    def test_bad_path(self):
        with pytest.raises(OSError):
            rawToNumpy(self.bad_path)

    def test_good_path(self):
        try:
            rawToNumpy(self.good_path)
        except OSError as exc:
            assert False, f"Opening a valid RAW file raised an exception {exc}"

    def test_bin_number(self):
        mzs, _, _ = rawToNumpy(self.good_path)
        assert len(mzs) == 10799

    def test_intens_number(self):
        _, intens, _ = rawToNumpy(self.good_path)
        assert len(intens) == 10799

    def test_file_name(self):
        _, _, metadata = rawToNumpy(self.good_path)
        assert metadata[0] == self.real_file

    def test_low_lim(self):
        _, _, metadata = rawToNumpy(self.good_path)
        assert float(metadata[-3]) == 100.0

    def test_up_lim(self):
        _, _, metadata = rawToNumpy(self.good_path)
        assert float(metadata[-2]) == 1000.0

    def test_num_scans(self):
        _, _, metadata = rawToNumpy(self.good_path)
        assert int(metadata[-1]) == 71

