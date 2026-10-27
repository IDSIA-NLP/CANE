class BaseInputFormatter:
    def format(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")
