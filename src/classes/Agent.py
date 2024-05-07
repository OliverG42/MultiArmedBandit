class Agent:
    def __init__(self):
        self.name = self.__class__.__name__
        self._initialized = False

    def choose_lever(self, arm_state):
        raise NotImplementedError("The choose_lever method hasn't been implemented!")

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._init_args = args
        instance._init_kwargs = kwargs
        return instance

    def reset(self):
        if self._initialized:
            # noinspection PyArgumentList
            self.__init__(*self._init_args, **self._init_kwargs)

    def _initialize(self):
        self._initialized = True
