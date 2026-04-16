from PIL import Image

from cnnClassifier.utils.data_utils import build_grouped_split_dataframe


def _save_solid_image(path, color):
    Image.new("RGB", (8, 8), color=color).save(path)


def test_grouped_split_dataframe_removes_exact_duplicates(tmp_path):
    dataset_dir = tmp_path / "dataset"
    cancer_dir = dataset_dir / "adenocarcinoma"
    normal_dir = dataset_dir / "normal"
    cancer_dir.mkdir(parents=True)
    normal_dir.mkdir(parents=True)

    _save_solid_image(cancer_dir / "cancer_a.png", (10, 10, 10))
    _save_solid_image(cancer_dir / "cancer_b.png", (10, 10, 10))
    _save_solid_image(cancer_dir / "cancer_unique.png", (20, 20, 20))
    _save_solid_image(normal_dir / "normal_a.png", (30, 30, 30))
    _save_solid_image(normal_dir / "normal_b.png", (40, 40, 40))

    train_df, validation_df, summary = build_grouped_split_dataframe(
        dataset_dir=dataset_dir,
        validation_split=0.5,
        seed=42,
    )

    assert len(train_df) + len(validation_df) == 4
    assert summary["adenocarcinoma"]["duplicates_removed"] == 1
