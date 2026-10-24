from src.utils import paths


def test_project_root_contains_markers():
    assert (paths.PROJECT_ROOT / "be" / "main.py").is_file()
    assert (paths.PROJECT_ROOT / "src" / "inference" / "predictor.py").is_file()


def test_default_model_path_points_to_weights():
    assert paths.DEFAULT_MODEL_PATH == paths.WEIGHTS_DIR / "best.pt"


def test_dataset_yamls_exist_and_use_relative_paths():
    for yaml_path in (paths.KAGGLE_DATA_YAML, paths.CUSTOM_DATA_YAML):
        assert yaml_path.is_file(), yaml_path
        content = yaml_path.read_text(encoding="utf-8")
        assert "images/train" in content
        assert "images/val" in content
        assert "D:/" not in content
        assert "D:\\" not in content


def test_default_data_yaml_is_kaggle():
    assert paths.DATA_YAML == paths.KAGGLE_DATA_YAML
