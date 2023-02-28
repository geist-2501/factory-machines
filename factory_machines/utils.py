from scipy.ndimage.filters import uniform_filter1d


class LinearDecay:
    def __init__(self, start_value, final_value, max_steps):
        self.start_value = start_value
        self.final_value = final_value
        self.max_steps = max_steps

    def get(self, step):
        step = min(step, self.max_steps)
        upper = self.start_value * (self.max_steps - step)
        lower = self.final_value * step
        return (upper + lower) / self.max_steps


def smoothen(data):
    return uniform_filter1d(data, size=30)
