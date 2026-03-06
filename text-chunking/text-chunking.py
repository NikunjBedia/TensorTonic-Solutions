def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    output=[]
    for i in range(0,len(tokens),chunk_size-overlap):
        output.append(tokens[i:i+chunk_size])
        if i+chunk_size>len(tokens)-1:
            break

    return output
    