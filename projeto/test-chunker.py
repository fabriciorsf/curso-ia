from utils.semantic_chunker import SemanticChunker


little_text = """
    A inteligência artificial transformou-se em um dos pilares mais importantes da tecnologia moderna.
    """
small_text = """
    A inteligência artificial transformou-se em um dos pilares mais importantes da tecnologia moderna.

    No ambiente de trabalho, essas ferramentas automatizam tarefas repetitivas e otimizam a análise de grandes volumes de dados.
    """
medium_text = """
    A inteligência artificial transformou-se em um dos pilares mais importantes da tecnologia moderna.
    Ela está presente em sistemas de navegação, assistentes virtuais e diagnósticos médicos avançados.
    Essa evolução rápida redefine a forma como a sociedade consome informação e resolve problemas complexos cotidianos.
    """
large_text = """
    A inteligência artificial transformou-se em um dos pilares mais importantes da tecnologia moderna.
    Ela está presente em sistemas de navegação, assistentes virtuais e diagnósticos médicos avançados.
    Essa evolução rápida redefine a forma como a sociedade consome informação e resolve problemas complexos cotidianos.

    No ambiente de trabalho, essas ferramentas automatizam tarefas repetitivas e otimizam a análise de grandes volumes de dados.
    Os profissionais agora podem focar em atividades estratégicas e criativas que exigem sensibilidade humana.
    No entanto, essa transição exige que a força de trabalho desenvolva novas habilidades constantemente.

    Apesar dos benefícios evidentes, o avanço dessa tecnologia levanta debates éticos profundos e urgentes.
    Questões sobre a privacidade dos dados pessoais e a segurança da informação preocupam especialistas no mundo todo.
    É fundamental criar regulamentações claras para evitar vieses e garantir o uso responsável dos algoritmos.

    O futuro dessa inovação promete uma integração ainda maior com a nossa rotina diária.
    Espera-se que novos modelos resolvam desafios globais, como as mudanças climáticas e a escassez de recursos.
    O equilíbrio entre o desenvolvimento tecnológico e o bem-estar humano ditará o sucesso dessa jornada.
    """


def measure_chunks(small_text, chunker):
    chunks = chunker.create_chunks(small_text)
    print(f"Total de chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: '{chunk}'")


chunker = SemanticChunker(max_tokens=300)

print("=====LITTLE TEXT=====")
measure_chunks(little_text, chunker)

print("=====SMALL TEXT=====")
measure_chunks(small_text, chunker)

print("=====MEDIUM TEXT=====")
measure_chunks(medium_text, chunker)

print("=====LARGE TEXT=====")
measure_chunks(large_text, chunker)
