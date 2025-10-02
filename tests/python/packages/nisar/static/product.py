from nisar.static.product import build_hdf5_dataset_creation_kwds_dict


class TestBuildHdf5DatasetCreationKwdsDict:
    def test_no_chunking(self):
        kwds = build_hdf5_dataset_creation_kwds_dict(
            dataset_shape=(1024, 1024),
            chunk_size=(-1, -1),
            compression_enabled=True,
            compression_type="gzip",
            compression_level=4,
            shuffle=True,
        )
        assert kwds == {"compression": "gzip", "shuffle": True, "compression_opts": 4}

    def test_no_compression(self):
        kwds = build_hdf5_dataset_creation_kwds_dict(
            dataset_shape=(1024, 1024),
            chunk_size=(512, 512),
            compression_enabled=False,
            compression_type="gzip",
            compression_level=4,
            shuffle=True,
        )
        assert kwds == {"chunks": (512, 512)}

    def test_gzip_compression(self):
        kwds = build_hdf5_dataset_creation_kwds_dict(
            dataset_shape=(1024, 1024),
            chunk_size=(512, 512),
            compression_enabled=True,
            compression_type="gzip",
            compression_level=4,
            shuffle=True,
        )
        assert kwds == {
            "chunks": (512, 512),
            "compression": "gzip",
            "shuffle": True,
            "compression_opts": 4,
        }

    def test_lzf_compression(self):
        kwds = build_hdf5_dataset_creation_kwds_dict(
            dataset_shape=(1024, 1024),
            chunk_size=(512, 512),
            compression_enabled=True,
            compression_type="lzf",
            compression_level=4,
            shuffle=False,
        )
        assert kwds == {
            "chunks": (512, 512),
            "compression": "lzf",
            "shuffle": False,
        }

    def test_clip_chunk_size(self):
        kwds = build_hdf5_dataset_creation_kwds_dict(
            dataset_shape=[256, 256],
            chunk_size=[512, 512],
            compression_enabled=True,
            compression_type="gzip",
            compression_level=4,
            shuffle=True,
        )
        assert kwds["chunks"] == (256, 256)
