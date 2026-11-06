# Regras de código da v2

- Módulos Python até 600 linhas; funções e métodos até 80 linhas.
- Código JavaScript e CSS legível, sem minificação, com linhas de até 300 caracteres.
- Configuração da campanha centralizada em `core/config.py` e caminhos de runtime em
  `core/paths.py`.
- Variáveis de ambiente, quando necessárias, devem ser lidas na camada de configuração.
- Erros de domínio devem ter mensagem clara em português.
- Mudanças em cada prompt atualizam `docs/v2/arquitetura.md` e registram decisões relevantes em
  ADRs de `docs/v2/decisoes/`.
- A checagem Ruff mais rígida é aplicada explicitamente ao pacote v2 para não alterar as regras
  de lint da v1.
