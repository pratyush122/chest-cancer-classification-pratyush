import pandas as pd

from scripts.train_ecg_model import build_balanced_training_dataframe


def test_build_balanced_training_dataframe_softens_minority_oversampling():
    train_df = pd.DataFrame(
        {
            "filepath": [f"file_{index}.png" for index in range(360)],
            "label": (
                ["Abnormal heartbeat"] * 176
                + ["History of MI"] * 70
                + ["Normal Person"] * 114
            ),
        }
    )

    balanced_train_df, target_counts = build_balanced_training_dataframe(train_df)
    balanced_counts = balanced_train_df["label"].value_counts().to_dict()

    assert target_counts == {
        "Abnormal heartbeat": 176,
        "History of MI": 114,
        "Normal Person": 141,
    }
    assert balanced_counts == target_counts
