class AgentNotFound(Exception):
    """Raised when an agent ID is requested but not registered"""


class WrapperNotFound(Exception):
    """Raised when a wrapper ID is requested but not registered"""
