from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from TaxiFareModel.encoders import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse


class Trainer():
    def __init__(self, X_train, X_test, y_train, y_test):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                                ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
                "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
                'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                            remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                            ('linear_model', LinearRegression())])
        return pipe

    def run(self, pipe):
        """set and train the pipeline"""
        pipe.fit(self.X_train, self.y_train)
        return pipe

    def evaluate(self, pipeline):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        return rmse


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    # set X and y
    y = df['fare_amount']
    X = df.drop('fare_amount', axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # train
    trainer = Trainer(X_train, X_val, y_train, y_val)
    pipeline = trainer.set_pipeline()
    pipe = trainer.run(pipeline)
    # evaluate
    rmse = trainer.evaluate(pipe)
    print(rmse)
