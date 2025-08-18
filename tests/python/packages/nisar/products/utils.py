import numpy as np

from nisar.products.utils import to_bytes


class TestToBytes:
    def test_str(self):
        s = "¥§©ÅΣ"
        b = to_bytes(s)
        assert b == s.encode("utf-8")
        assert b.shape == ()
        assert np.issubdtype(b.dtype, np.bytes_)

    def test_arraylike(self):
        s = ["¥§©ÅΣ", "adsf"]
        b = to_bytes(s)
        assert b.shape == (2,)
        assert np.issubdtype(b.dtype, np.bytes_)
        assert b[0] == s[0].encode("utf-8")
        assert b[1] == s[1].encode("utf-8")