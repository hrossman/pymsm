import base64
from IPython.display import Image, display


def state_diagram(graph):
    """Plot a state diagram for a graph. See http://mermaid-js.github.io/mermaid/#/Tutorials?id=jupyter-integration-with-mermaid-js

    Example:
        state_diagram(
            '''stateDiagram-v2
            s1 : (1) Primary surgery
            s2: (2) Disease recurrence
            s3: (3) Death
            s1 --> s2
            s1 --> s3
            s2 --> s3
            ''')
    """
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    img = Image(url="https://mermaid.ink/img/" + base64_string)
    display(img)
