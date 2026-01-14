def parse_cg_file(file_path):
    """
    Parses a .cg file to extract nodes and edges.

    Args:
        file_path (str): Path to the .cg file.

    Returns:
        tuple: A tuple containing a list of nodes and a list of edges.
    """
    nodes = []
    edges = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        mode = None

        for line in lines:
            line = line.strip()
            if line.startswith('<NODES>'):
                mode = 'nodes'
                continue
            elif line.startswith('<EDGES>'):
                mode = 'edges'
                continue
            elif not line or line.startswith('//'):
                continue

            if mode == 'nodes':
                nodes.append(line)
            elif mode == 'edges':
                if '<->' in line:  # Handle bidirectional edges
                    src, tgt = map(str.strip, line.split('<->'))
                    edges.append((src, tgt))
                    edges.append((tgt, src))
                elif '->' in line:  # Handle directed edges
                    src, tgt = map(str.strip, line.split('->'))
                    edges.append((src, tgt))

    return nodes, edges