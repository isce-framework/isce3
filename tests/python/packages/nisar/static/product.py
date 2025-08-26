from nisar.static.product import build_hdf5_dataset_creation_kwds_dict


class TestBuildHdf5DatasetCreationKwdsDict:
    def test_no_chunking(self):
        kwds = build_hdf5_dataset_creation_kwds_dict(
            chunk_size=(-1, -1),
            compression_enabled=True,
            compression_type="gzip",
            compression_level=9,
            shuffle=True,
        )
        assert kwds == {}

    def test_no_compression(self):
        kwds = build_hdf5_dataset_creation_kwds_dict(
            chunk_size=(512, 512),
            compression_enabled=False,
            compression_type="gzip",
            compression_level=9,
            shuffle=True,
        )
        assert kwds == {"chunks": (512, 512)}

    def test_gzip_compression(self):
        kwds = build_hdf5_dataset_creation_kwds_dict(
            chunk_size=(512, 512),
            compression_enabled=True,
            compression_type="gzip",
            compression_level=9,
            shuffle=True,
        )
        assert kwds == {
            "chunks": (512, 512),
            "compression": "gzip",
            "shuffle": True,
            "compression_opts": 9,
        }

    def test_lzf_compression(self):
        kwds = build_hdf5_dataset_creation_kwds_dict(
            chunk_size=(512, 512),
            compression_enabled=True,
            compression_type="lzf",
            compression_level=9,
            shuffle=False,
        )
        assert kwds == {
            "chunks": (512, 512),
            "compression": "lzf",
            "shuffle": False,
        }
