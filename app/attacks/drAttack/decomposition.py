def decompose_prompt(prompt):

    words = prompt.split()

    chunks = []

    chunk_size = 2

    for i in range(0, len(words), chunk_size):

        chunks.append(
            " ".join(words[i:i+chunk_size])
        )

    return chunks