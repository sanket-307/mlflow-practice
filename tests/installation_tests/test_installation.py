def test_import_module():
    try:
        import mlhousing.ingest_data
        import mlhousing.preprocess
        import mlhousing.train
        import mlhousing.score

    except ImportError:
        assert False, "Failed to import module: mlhousing.ingest_data"
