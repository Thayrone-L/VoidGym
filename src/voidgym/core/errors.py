"""Exceções públicas da v2, apresentadas ao operador em português."""


class VoidGymError(Exception):
    """Erro esperado da aplicação VoidGym."""


class ConfigError(VoidGymError):
    """Configuração inválida ou incompatível."""


class ExperienceStoreError(VoidGymError):
    """Falha de leitura ou gravação no armazém de experiência."""
