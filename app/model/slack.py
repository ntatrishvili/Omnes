import pandas as pd


class Slack:
    def __init__(self, flow_in: pd.DataFrame, flow_out: pd.DataFrame):
        self.flow_in = flow_in
        self.flow_out = flow_out
