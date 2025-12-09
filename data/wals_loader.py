import pandas as pd

class WALSLoader:
    def __init__(self, data_path="./data"):
        self.data_path = data_path
        self.languages = None
        self.parameters = None
        self.values = None
        self.codes = None

    def load_csvs(self):
        """Load all WALS CSV files into pandas DataFrames"""
        self.languages = pd.read_csv(f"{self.data_path}/languages.csv")
        self.parameters = pd.read_csv(f"{self.data_path}/parameters.csv")
        self.values = pd.read_csv(f"{self.data_path}/values.csv")
        self.codes = pd.read_csv(f"{self.data_path}/codes.csv")
        print("WALS CSVs loaded successfully!")

    def get_features_for_languages(self, selected_languages=None, selected_parameters=None):
        """
        Returns a numeric feature matrix (languages x features) for selected languages
        and parameters. Handles merges and pivoting automatically.
        """
        # Start from values CSV
        df = self.values.copy()

        # Filter languages
        if selected_languages is not None:
            df = df[df['Language_ID'].isin(selected_languages)]

        # Filter parameters
        if selected_parameters is not None:
            df = df[df['Parameter_ID'].isin(selected_parameters)]

        # Merge with codes to get descriptions
        df = df.merge(self.codes, left_on='Code_ID', right_on='ID', how='left')

        # Ensure Parameter_ID column has consistent name
        if 'Parameter_ID_x' in df.columns:
            df = df.rename(columns={'Parameter_ID_x': 'Parameter_ID'})

        # Pivot: languages as rows, parameters as columns
        feature_matrix = df.pivot(index='Language_ID', columns='Parameter_ID', values='Description')

        # Convert descriptions to numeric codes (factorize)
        feature_matrix_numeric = feature_matrix.apply(lambda col: pd.factorize(col)[0])

        # Optional: merge language names if you want readable index
        if 'languages' in dir(self):
            feature_matrix_numeric = feature_matrix_numeric.merge(
                self.languages[['ID', 'Name']], left_index=True, right_on='ID'
            ).set_index('Name')

        # Ensure all column names are strings for sklearn
        feature_matrix_numeric.columns = feature_matrix_numeric.columns.astype(str)

        return feature_matrix_numeric
