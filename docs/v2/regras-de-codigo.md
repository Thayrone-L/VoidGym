# Regras de código da v2

- Cada módulo Python em `src/voidgym` tem no máximo 600 linhas; funções têm no máximo 80 linhas.
- JavaScript e CSS ficam legíveis, uma declaração por linha; nenhuma linha passa de 300 caracteres.
- Configuração vive em `core/config.py`; caminhos de runtime em `core/paths.py`; só `config.py` lê variáveis de ambiente.
- Exceções da v2 herdam de `VoidGymError` e apresentam mensagem em português.
- Toda alteração atualiza `arquitetura.md` e registra decisões novas como ADR curto em `decisoes/`.
- Ruff aplica complexidade C901 (limite 12), imports e convenções de nomes à v2 sem alterar a v1.
