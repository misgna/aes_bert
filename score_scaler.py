class ScoreScaler:
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def min_max_scaler(self, score):
        return (score - self.minimum) / (self.maximum - self.minimum)
    
    def inverse_scaler(self, score):
        return round(score * (self.maximum - self.minimum) + self.minimum)