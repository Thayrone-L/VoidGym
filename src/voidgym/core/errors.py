"""Hierarquia de erros da v2, mantendo compatibilidade com tipos Python comuns."""


class VoidGymError(Exception):
    """Erro de domínio da VoidGym com mensagem pronta para operador."""


class VoidGymValueError(VoidGymError, ValueError):
    """Entrada incompatível com o contrato da VoidGym."""


class VoidGymTypeError(VoidGymError, TypeError):
    """Tipo de entrada incompatível com o contrato da VoidGym."""


class VoidGymRuntimeError(VoidGymError, RuntimeError):
    """Falha operacional da VoidGym."""
