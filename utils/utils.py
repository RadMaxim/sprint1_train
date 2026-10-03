import pandas as pd


def remove_duplicates(
    data: pd.DataFrame
) -> pd.DataFrame:

    rows_before = len(data)

    data = (
        data
        .drop_duplicates()
        .copy()
    )

    print(
        f"Duplicates removed: "
        f"{rows_before - len(data)}"
    )

    return data


def cast_types(
    data: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> pd.DataFrame:

    data["senior_citizen"] = (
        data["senior_citizen"]
        .astype("int8")
    )

    data["target"] = (
        data["target"]
        .astype("int8")
    )

    for col in numeric_columns:
        data[col] = (
            data[col]
            .astype("float32")
        )

    for col in categorical_columns:
        data[col] = (
            data[col]
            .astype("category")
        )

    return data


def fill_missing_values(
    data: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> pd.DataFrame:

    for col in categorical_columns:

        if data[col].isna().any():

            mode_value = (
                data[col]
                .mode(dropna=True)
                .iloc[0]
            )

            data[col] = (
                data[col]
                .fillna(mode_value)
            )

            print(
                f"{col}: "
                f"filled with mode = "
                f"{mode_value}"
            )

    for col in numeric_columns:

        if data[col].isna().any():

            median_value = (
                data[col]
                .median()
            )

            data[col] = (
                data[col]
                .fillna(median_value)
            )

            print(
                f"{col}: "
                f"filled with median = "
                f"{median_value:.2f}"
            )

    return data


def validate_missing_values(
    data: pd.DataFrame
) -> None:

    missing_count = (
        data.isna()
        .sum()
        .sum()
    )

    print(
        f"Missing values: "
        f"{missing_count}"
    )

    if missing_count > 0:
        raise ValueError(
            "Dataset contains missing values"
        )


def remove_outliers_iqr(
    data: pd.DataFrame,
    numeric_columns: list[str],
    threshold: float = 1.5,
) -> pd.DataFrame:

    outlier_mask = pd.Series(
        False,
        index=data.index,
    )

    for col in numeric_columns:

        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)

        iqr = q3 - q1

        lower_bound = (
            q1 - threshold * iqr
        )

        upper_bound = (
            q3 + threshold * iqr
        )

        col_outliers = (
            (data[col] < lower_bound)
            |
            (data[col] > upper_bound)
        )

        print(
            f"{col}: "
            f"{col_outliers.sum()} outliers, "
            f"bounds: "
            f"{lower_bound:.2f} - "
            f"{upper_bound:.2f}"
        )

        outlier_mask |= col_outliers

    total_outliers = int(
        outlier_mask.sum()
    )

    print(
        f"Total rows with outliers: "
        f"{total_outliers}"
    )

    return (
        data.loc[
            ~outlier_mask
        ]
        .copy()
    )