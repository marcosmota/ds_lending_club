import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(
        self,
        train_size=0.5,
        test_size=0.2,
        val_size=0.15,
        calibration_size=0.15,
        random_state=42,
    ):
        if train_size + test_size + val_size + calibration_size != 1.0:
            raise ValueError(
                "The sum of train, test, and validation proportions must equal 1.0"
            )
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.calibration_size = calibration_size
        self.random_state = random_state

    def split_data(self, X, y):
        data = X.copy()
        data["target"] = y

        train_data = data.sample(frac=self.train_size, random_state=self.random_state)
        temp_data = data.drop(train_data.index)

        test_val_ratio = self.test_size / (self.test_size + self.val_size)
        test_data = temp_data.sample(
            frac=test_val_ratio, random_state=self.random_state
        )
        val_data = temp_data.drop(test_data.index)
        return train_data, val_data, test_data


class TimeSplitVerifier:
    def __init__(self, df, date_col):
        """
        Classe para verificar se o dataset tem mudanças temporais significativas e determinar o tipo de split.

        :param df: DataFrame contendo os dados.
        :param date_col: Nome da coluna de data.
        """
        self.df = df.copy()
        self.date_col = date_col
        self.threshold = 0.05

        if self.date_col not in self.df.columns:
            raise ValueError(
                f" A coluna '{self.date_col}' não foi encontrada no DataFrame."
            )

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.sort_values(by=self.date_col)

        print(f" Coluna de data '{self.date_col}' identificada e processada.")

    def run_tests(self):
        """
        Executa testes estatísticos para verificar se há mudanças na distribuição ao longo do tempo.
        """
        mid_point = len(self.df) // 2
        early_data = self.df.iloc[:mid_point]
        late_data = self.df.iloc[mid_point:]

        print("Running stats tests...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        ks_results = {}
        for col in numeric_cols:
            stat, p_value = ks_2samp(early_data[col], late_data[col])
            ks_results[col] = p_value

        mw_results = {}
        for col in numeric_cols:
            stat, p_value = mannwhitneyu(early_data[col], late_data[col])
            mw_results[col] = p_value

        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        chi2_results = {}
        for col in categorical_cols:
            contingency_table = pd.crosstab(
                self.df[col], self.df[self.date_col].dt.year
            )
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            chi2_results[col] = p_value

        ks_significant = any(p < self.threshold for p in ks_results.values())
        mw_significant = any(p < self.threshold for p in mw_results.values())
        chi2_significant = any(p < self.threshold for p in chi2_results.values())

        report_df = self._generate_report(ks_results, mw_results, chi2_results)

        print(report_df)

        if ks_significant or mw_significant or chi2_significant:
            print(
                "[!] Mudança estatística detectada! Recommended use `TimeSeriesSplit` to avoid data leakage."
            )
            return "TimeSeriesSplit"
        else:
            print(
                "Sem mudança significativa. Pode usar `train_test_split` com embaralhamento."
            )
            return "train_test_split"

    def _generate_report(self, ks_results, mw_results, chi2_results):
        report_data = {
            "Test": [],
            "Significance": [],
            "Threshold": [],
            "Result": [],
        }

        for col, p_value in ks_results.items():
            report_data["Test"].append(f"KS Test for {col}")
            report_data["Significance"].append(p_value)
            report_data["Threshold"].append(self.threshold)
            report_data["Result"].append(
                "Significant" if p_value < self.threshold else "Not Significant"
            )

        for col, p_value in mw_results.items():
            report_data["Test"].append(f"Mann-Whitney U Test for {col}")
            report_data["Significance"].append(p_value)
            report_data["Threshold"].append(self.threshold)
            report_data["Result"].append(
                "Significant" if p_value < self.threshold else "Not Significant"
            )

        for col, p_value in chi2_results.items():
            report_data["Test"].append(f"Chi-Square Test for {col}")
            report_data["Significance"].append(p_value)
            report_data["Threshold"].append(self.threshold)
            report_data["Result"].append(
                "Significant" if p_value < self.threshold else "Not Significant"
            )

        return pd.DataFrame(report_data)
