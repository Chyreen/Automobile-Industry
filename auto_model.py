import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


CURRENT_YEAR = 2021
SAMPLE_SIZE = 30000
RANDOM_STATE = 42
N_SPLITS = 5


class ReplaceYearForNewVehicles:
    def __init__(self, current_year: int = 2021):
        self.current_year = current_year

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["year_of_registration"] = df.apply(
            lambda row: self.current_year
            if row["vehicle_condition"] == "NEW"
            else row["year_of_registration"],
            axis=1,
        )
        return df


class MapRegistrationCodesToYears:
    def __init__(self):
        self.a_dict = {
            "7": 2007, "8": 2008, "10": 2010, "13": 2013, "17": 2017,
            "55": 2005, "57": 2007, "59": 2009, "63": 2013, "64": 2014,
            "65": 2015, "66": 2016, "68": 2018,
        }
        self.alphabets = {
            "A": 1963, "B": 1964, "C": 1965, "D": 1966, "E": 1967, "F": 1968,
            "G": 1969, "H": 1970, "J": 1971, "K": 1972, "L": 1973, "M": 1974,
            "N": 1975, "P": 1976, "R": 1977, "S": 1978, "T": 1979, "V": 1980,
            "W": 1981, "X": 1982, "Y": 1983,
        }
        self.D = {str(regcode): year for regcode, year in zip(range(2, 24), range(2002, 2024))}
        self.D2 = {str(regcode): year for regcode, year in zip(range(51, 73), range(2002, 2024))}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        mask_lt_1904 = df["year_of_registration"] < 1904
        df.loc[mask_lt_1904, "year_of_registration"] = df.loc[mask_lt_1904, "reg_code"].map(self.a_dict)

        mask_null = df["year_of_registration"].isna()
        df.loc[mask_null, "year_of_registration"] = df.loc[mask_null, "reg_code"].map(self.alphabets)

        mask_null = df["year_of_registration"].isna()
        df.loc[mask_null, "year_of_registration"] = df.loc[mask_null, "reg_code"].map(self.D)

        mask_null = df["year_of_registration"].isna()
        df.loc[mask_null, "year_of_registration"] = df.loc[mask_null, "reg_code"].map(self.D2)

        used_mask = df["vehicle_condition"] == "USED"
        used_mode = df.loc[used_mask, "year_of_registration"].mode()
        if not used_mode.empty:
            df.loc[used_mask, "year_of_registration"] = df.loc[
                used_mask, "year_of_registration"
            ].fillna(used_mode.iloc[0])

        specific_condition = (
            (df["year_of_registration"] == 1909)
            & (df["reg_code"] == "9")
            & (df["standard_make"] == "Hyundai")
            & (df["standard_model"] == "i10")
        )
        df.loc[specific_condition, "year_of_registration"] = 2009

        return df


class DataPreprocessor:
    def __init__(self, current_year: int = 2021):
        self.current_year = current_year

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data.drop_duplicates(inplace=True)

        old_new = {
            "standard_colour": "colour",
            "standard_make": "make",
            "standard_model": "model",
            "vehicle_condition": "condition",
            "year_of_registration": "year",
            "body_type": "body",
            "crossover_car_and_van": "crossover",
            "fuel_type": "fuel",
        }
        data.rename(columns=old_new, inplace=True)

        extreme = data["price"] == 9999999
        price_mode_by_year_make = data.groupby(["make", "year"])["price"].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        data.loc[extreme, "price"] = price_mode_by_year_make[extreme]

        data["age"] = self.current_year - data["year"]
        data = data[data["mileage"] != 999999].copy()

        data["status"] = "non_scrab"
        data.loc[
            (data["mileage"] >= (10000 * data["age"])) & (data["age"] > 10),
            "status"
        ] = "scrab"

        categorical_cols = ["colour", "fuel", "body"]

        for col in categorical_cols:
            data[col] = data.groupby(["year"])[col].transform(
                lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else pd.NA)
            )

        for col in categorical_cols:
            data[col] = data.groupby(["age", "make"])[col].transform(
                lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else pd.NA)
            )

        for col in categorical_cols:
            data[col] = data[col].fillna("Unknown")

        subset = data[(data["condition"] == "USED") & (data["mileage"] > 0)]
        mean_mileage = subset["mileage"].mean()

        data.loc[
            (data["condition"] == "USED") & ((data["mileage"] == 0) | (data["mileage"].isna())),
            "mileage"
        ] = mean_mileage

        data["mileage"] = data["mileage"].fillna(data["mileage"].median())

        # Keep state column creation for comparison if you want to inspect it,
        # but we will NOT use it in the model features below.
        data["state"] = "modern"
        data.loc[(data["age"] >= 45) & (data["age"] <= 117), "state"] = "vin_ant"
        data.loc[(data["age"] > 20) & (data["age"] <= 45), "state"] = "classics"

        drop_cols = ["public_reference", "reg_code", "year", "crossover"]
        existing_drop_cols = [c for c in drop_cols if c in data.columns]
        data.drop(columns=existing_drop_cols, inplace=True)

        return data


def stratified_sample(df: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    df = df.copy()

    top_makes = df["standard_make"].value_counts().nlargest(20).index
    df["make_for_sampling"] = df["standard_make"].where(df["standard_make"].isin(top_makes), "Other")
    df["condition_for_sampling"] = df["vehicle_condition"].fillna("Unknown")
    df["sample_strata"] = df["make_for_sampling"].astype(str) + "_" + df["condition_for_sampling"].astype(str)

    frac = min(sample_size / len(df), 1.0)

    sampled = (
        df.groupby("sample_strata", group_keys=False)
        .apply(lambda x: x.sample(max(1, int(round(len(x) * frac))), random_state=random_state))
        .reset_index(drop=True)
    )

    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    sampled = sampled.drop(columns=["make_for_sampling", "condition_for_sampling", "sample_strata"], errors="ignore")
    return sampled


def load_and_clean_data(csv_path: str, sample_size: int = 30000) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if sample_size is not None and len(df) > sample_size:
        df = stratified_sample(df, sample_size=sample_size, random_state=RANDOM_STATE)

    df = ReplaceYearForNewVehicles(current_year=CURRENT_YEAR).fit_transform(df)
    df = MapRegistrationCodesToYears().fit_transform(df)
    df = DataPreprocessor(current_year=CURRENT_YEAR).fit_transform(df)

    df = df.dropna(subset=["price", "age"])
    df = df[(df["age"] >= 0) & (df["age"] <= 117)]
    df = df[df["price"] > 0]

    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    return model


def make_cv_strata(df: pd.DataFrame) -> pd.Series:
    top_makes = df["make"].value_counts().nlargest(15).index
    make_group = df["make"].where(df["make"].isin(top_makes), "Other")
    price_bins = pd.qcut(df["price"], q=5, duplicates="drop").astype(str)
    strata = make_group.astype(str) + "_" + df["condition"].astype(str) + "_" + price_bins
    return strata


def train_with_stratified_cv(df: pd.DataFrame):
    # NOTE: state removed here on purpose
    feature_cols = [
        "mileage",
        "make",
        "model",
        "condition",
        "colour",
        "body",
        "fuel",
        "age",
        "status",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y_log = np.log1p(df["price"])

    model = build_pipeline(X)
    strata = make_cv_strata(df)

    strata_counts = strata.value_counts()
    rare = strata_counts[strata_counts < N_SPLITS].index
    strata = strata.where(~strata.isin(rare), "RARE_GROUP")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        model,
        X,
        y_log,
        cv=skf.split(X, strata),
        scoring={"r2": "r2"},
        n_jobs=1,
        return_train_score=False
    )

    r2_scores = cv_results["test_r2"]

    print("CV R2 scores:", r2_scores)
    print("Average CV R2:", r2_scores.mean())
    print("CV R2 std:", r2_scores.std())

    model.fit(X, y_log)
    return model


if __name__ == "__main__":
    df_clean = load_and_clean_data("adverts.csv", sample_size=SAMPLE_SIZE)
    print("Cleaned shape:", df_clean.shape)

    model = train_with_stratified_cv(df_clean)

    joblib.dump(model, "auto_model.pkl")
    print("Saved as auto_model.pkl")